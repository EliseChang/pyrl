"""
The rats' TaskSwitch stage curriculum on the per-timestep engagement cost (version 4 of the
engagement line), 16 Oct 2026.

What this is. `pagan_hist_stages.py` is the same 10-stage ladder on the *lump-cost* effort network
(`pagan_hist_effort2`); this file is the ladder on `pagan_hist_tscost`, where engagement is charged
per timestep and withdrawal has its own output. Where the two `tscost` runs so far hand off from
the nose-poke pre-training straight to the full task, this one hands off to the ladder, so it is
the arm that asks whether the graded stage structure protects engagement where the straight
hand-off does not.

The ladder, read from the protocol (learning_rats_RNNs/Protocol/@TaskSwitch4new/TrainingSection.m,
StimulusSection.m) and the pooled trial tables of 208 rats (stage_statistics.csv):

  stage  what is on                                                   promotion (rats)
  1-2    LOC task only, easiest evidence (gamma 4), no FRQ pulses      >150 trials, >0.75 correct
  3-4    LOC and FRQ tasks in performance-gated blocks, each pure      both tasks >0.7
         (the other feature's pulses absent), gamma 4
  5-6    both features present, the irrelevant one uninformative       both >0.7
         (gamma 0 = 50/50 pulses)
  7      irrelevant becomes informative: gamma in {1, 4}; quadrant     >0.68
         anti-bias on
  8      irrelevant in {1, 2.5, 4}                                     >0.68
  9      relevant harder too: {2.5, 4}; irrelevant {1, 2.5, 4}         >0.68
  10     both in {1, 2.5, 4}: the full task                            (end)

gamma -> pulse strength: 4 -> 0.25 (easiest), 2.5 -> 0.35, 1 -> 0.45, 0 (uninformative) -> 0.5.

This is a port of the share's **version 2** ladder (13 Oct), which adds three things over the
version 1 still committed in this repo as `pagan_hist_stages.py`: the lab's side anti-bias in
stages 1-6 (per task, as in `pagan_hist_antibias`), the quadrant anti-bias in stage 10 of every
arm (so `short` and `full` differ only in whether stages 7-9 exist), and a cap on block length at
twice the rats' median for the stage - the protocol relies on a trainer to end a block a stuck rat
never ends, and three networks of the first probe sat in one task for 30,000-90,000 trials.
`pagan_hist_stages.py` is deliberately left alone: it is the committed record of the `st55` effort2
runs.

Blocks. From stage 3 the task is set by performance-gated blocks (task_switch_min_perf): the
context switches when the running accuracy over the last 30 attempted trials reaches the stage's
switch threshold (0.8 for stages 3-6, 0.7 from 7) and the block is at least 20 trials long, or at
twice the rats' median block length regardless.

Promotion. Checked at every checkpoint on the running accuracy of *training* trials per task (the
last 150 attempted trials of the stage): a stage is left when at least 150 trials were attempted
in it and every task on in that stage is above the threshold. No demotion, so stage lengths are
the network's own, not the rats' medians.

Quadrant anti-bias (stages 7+). Per task, a kernel-weighted hit rate per quadrant of the stimulus
matrix (relevant side x irrelevant side), kernel exp(-lag/tau) with tau 50, normalised over the
four and rounded to 2 decimals; the next quadrant is drawn from softmax(-fracs x beta x 1.5) for
LOC and softmax(-fracs x beta) for FRQ, beta 5. The protocol also scales the reward by
4 x p(quadrant) (water volume), kept here behind PAGAN_QAB_REWARD; the delay is not modelled.

-----------------------------------------------------------------------------------------
Two things this file exists to get right, both of which the abandoned `pagan_hist_stages3.py`
(the same ladder on `pagan_hist_fixcost`) got wrong.
-----------------------------------------------------------------------------------------

1. HOOK CHAINING. pyrl binds one function per hook name out of the model file's namespace
   (pyrl/model.py), and a `def` here shadows whatever the star import brought in - the harness
   does no chaining. `pagan_hist_tscost` defines `set_validation`, `record_outcome`, `update` and
   `terminate`, so each has to be handled explicitly:

     set_validation   chains already, via the hasattr guard below.
     record_outcome   calls _eff.record_outcome FIRST and UNCONDITIONALLY, before this module's
                      own `if _st['validating']: return`. The two filters are complementary, not
                      overlapping: tscost summarises the *validation* batch and early-returns when
                      not validating, while the ladder tracks *training*-trial accuracy for
                      promotion and the anti-bias kernels and early-returns when validating. Both
                      must see every trial. stages3 shadowed this outright, which silenced its
                      parent's diagnostics completely.
     update           calls _eff.update(perf) exactly once and never _curr.update or _base.update:
                      tscost's update owns the nose-poke phase clock and the duration schedule. A
                      second call to either makes the pre-training phases pass twice as fast, or
                      gives the schedule a second independent clock - which is precisely how
                      stages3 ended up with the stage ladder and the stimulus length advancing on
                      unrelated clocks.
     terminate        routes through _eff.terminate, NOT _curr.terminate, so tscost's guard
                      ("never stop at criterion while the duration schedule is still running")
                      survives. stages3 used _curr.terminate and silently dropped it.

2. THE DURATION SCHEDULE RUNS IN STAGE 1 ONLY, ONCE IN ALL OF TRAINING. No new machinery is
   needed for the schedule itself: tscost.update already pins the presentation window to
   PAGAN_PRESTIM (20 ms) through the pre-training, sets it to PAGAN_STIM0 (50 ms) at a one-shot
   hand-off, and then grants one +50 ms opportunity per checkpoint. At the hand-off stage() is
   ORDER[0] = 1, so the schedule runs in Stage 1 by construction. Confining it there is one guard
   in update(): no stage may be left while `_base.stimulus < _base.stimulus_max`. Gating on the
   window rather than on `stage() == 1` is what makes it correct rather than merely convenient -
   the schedule cannot outlive Stage 1, and once the window reaches 1,300 ms tscost's own
   `stimulus >= stimulus_max` branch makes the schedule a permanent no-op, so "once" needs no
   latch. With PAGAN_RAMP=off the window is at full length from the hand-off and the guard never
   fires.

   Timing: pre-training 30 updates, then 25 growth opportunities at one per checkpoint = 125
   updates, so Stage 1 holds until about update 155 of 2,800. If the schedule ever stalls, the
   seed stays in Stage 1 for the whole run - accepted deliberately, and interpretable (the ladder
   never started because engagement never scaled).

Promotion counting needs no adaptation for the decline action, which is worth saying because it is
easy to assume otherwise: a trial counts as attempted only when `status['choice']` is set, which
excludes all four of tscost's non-choice outcomes (declined on the first timestep, declined inside
the window, declined after it, timed out), and `hit` is read from `status['correct']` rather than
from `reward`, so it is immune both to the per-step cost subtraction and to the quadrant reward
scaling. A network that declines 90% of trials therefore takes about ten times as many checkpoints
to accumulate its 150 attempted trials, but is never blocked from promotion.

One interaction documented rather than fixed: the quadrant scaling multiplies the *terminal* reward
by 4 x p(quadrant) while the engagement cost is charged separately on earlier timesteps, so an
attempted correct trial in stages 7-10 nets 4p - 0.55, which is negative for a rare quadrant
(p < 0.1375). That is inherent to combining the rats' water-volume scaling with an engagement cost;
it makes the expected value of engaging vary by quadrant.

Arms (PAGAN_STAGES):
    none     the full task from the first trial after the nose-poke phases (= pagan_hist_tscost)
    short    stages 1-6 then stage 10
    full     stages 1-10 with the quadrant anti-bias from stage 7
Analysis: PAGAN_STAGE=10 forces the full task and PAGAN_ANTIBIAS=0 switches the quadrant rule off,
so every arm is measured on the same distribution; PAGAN_STIM0 is left unset in rollouts, so the
window is the full 1,300 ms. Everything else - the cost, the decline output, the four actions,
R_ABORTED = 0, PAGAN_PRETRAIN, PAGAN_PHASE - is inherited and must not be redefined here.
"""
from __future__ import division, print_function

import os
import sys

import numpy as np

from pyrl import tasktools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagan_hist_tscost import *          # noqa: F401,F403  (the per-timestep cost, on the curriculum)
import pagan_hist_tscost as _eff
import pagan_hist_curriculum as _curr
import pagan as _base

ARM = os.environ.get('PAGAN_STAGES', 'full')          # none | short | full
QAB_ON = os.environ.get('PAGAN_ANTIBIAS', '1') != '0'
QAB_REWARD = os.environ.get('PAGAN_QAB_REWARD', '1') != '0'
QAB_BETA, QAB_TAU = 5.0, 50.0
FORCE_STAGE = os.environ.get('PAGAN_STAGE')            # e.g. '10' for analysis rollouts

G = {4: 0.25, 2.5: 0.35, 1: 0.45, 0: 0.5}               # gamma -> strength
# stage -> (tasks, relevant strengths, irrelevant strengths or None (absent), promotion threshold, switch threshold, antibias)
STAGES = {
    1:  (['loc'],        [G[4]],             None,                   0.75, None, False),
    2:  (['loc'],        [G[4]],             None,                   0.75, None, False),
    3:  (['loc', 'frq'], [G[4]],             None,                   0.70, 0.8,  False),
    4:  (['loc', 'frq'], [G[4]],             None,                   0.70, 0.8,  False),
    5:  (['loc', 'frq'], [G[4]],             [G[0]],                 0.70, 0.8,  False),
    6:  (['loc', 'frq'], [G[4]],             [G[0]],                 0.70, 0.8,  False),
    7:  (['loc', 'frq'], [G[4]],             [G[1], G[4]],           0.68, 0.7,  True),
    8:  (['loc', 'frq'], [G[4]],             [G[1], G[2.5], G[4]],   0.68, 0.7,  True),
    9:  (['loc', 'frq'], [G[2.5], G[4]],     [G[1], G[2.5], G[4]],   0.68, 0.7,  True),
    10: (['loc', 'frq'], [G[1], G[2.5], G[4]], [G[1], G[2.5], G[4]], None, 0.7,  True),
}
ORDER = {'none': [10], 'short': [1, 2, 3, 4, 5, 6, 10], 'full': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}[ARM]
MIN_TRIALS, WIN, MIN_BLOCK = 150, 30, 20
# the rats' median block length per stage (pooled tables, stage_statistics.csv, block_len_median);
# a block is ended at twice that even if the switch threshold was never reached
RAT_BLOCK = {1: 876, 2: 700, 3: 91, 4: 51, 5: 42, 6: 44, 7: 70, 8: 88, 9: 86, 10: 149}
MAX_BLOCK = {k: 2 * v for k, v in RAT_BLOCK.items()}
SAB_BETA, SAB_TAU = 5.0, 50.0                           # the side anti-bias (stages 1-6)
_SAB_DECAY = np.exp(-1.0 / SAB_TAU)

_st = {'pos': 0, 'n_attempted': 0, 'recent': {'loc': [], 'frq': []}, 'task': 'loc', 'block_len': 0,
       'validating': False,
       # quadrant anti-bias sums per task: [sum_hit, sum_kernel] x 4 quadrants, decayed per trial
       'q': {'loc': np.zeros((2, 4)), 'frq': np.zeros((2, 4))},
       # side anti-bias sums per task: [sum_hit, sum_kernel] x [L, R], decayed per trial
       'side': {'loc': np.zeros((2, 2)), 'frq': np.zeros((2, 2))}}
_DECAY = np.exp(-1.0 / QAB_TAU)


def stage():
    if FORCE_STAGE is not None:
        return int(FORCE_STAGE)
    return ORDER[min(_st['pos'], len(ORDER) - 1)]


def set_validation(flag):
    _st['validating'] = bool(flag)
    _eff.set_validation(flag) if hasattr(_eff, 'set_validation') else None


def _qab_probs(task):
    sh, sk = _st['q'][task]
    if np.any(sk <= 0):
        return np.full(4, 0.25)
    fr = sh / sk
    fr = fr / fr.sum() if fr.sum() > 0 else np.full(4, 0.25)
    fr = np.round(fr, 2)
    p = np.exp(-fr * QAB_BETA * (1.5 if task == 'loc' else 1.0))
    return p / p.sum()


def _side_p_right(task):
    """P(right is correct) under the lab's side anti-bias for this task: hit rate per correct side
    (kernel-weighted), normalised over the two sides and rounded to 2 decimals, softmax(-frac x
    beta); 0.5 until both sides have been seen (pagan_hist_antibias.antibias_p_right)."""
    sh, sk = _st['side'][task]
    if sk[0] <= 0 or sk[1] <= 0:
        return 0.5
    hit = sh / sk
    if hit.sum() <= 0:
        return 0.5
    fr = np.round(hit / hit.sum(), 2)
    e = np.exp(-fr * SAB_BETA)
    return float(e[1] / e.sum())


def get_condition(rng, dt, context={}):
    """Nose-poke phases first (inherited: the curriculum module sets the side and zeroes the
    pulses); in the task phase the stage decides the task, the strengths, the presence of the
    irrelevant feature and, with the quadrant anti-bias, the sides."""
    if _curr.phase() != 'task' and FORCE_STAGE is None:
        return _curr.get_condition(rng, dt, context)
    s = stage()
    tasks, rel_s, irr_s, _, switch_thr, qab = STAGES[s]
    if _st['validating'] or FORCE_STAGE is not None:
        task = tasktools.choice(rng, tasks)
    else:
        task = _st['task'] if _st['task'] in tasks else tasks[0]
    ctx = dict(context)
    ctx['context'] = task
    rel_key, irr_key = ('strength_loc', 'strength_freq') if task == 'loc' else ('strength_freq', 'strength_loc')
    ctx[rel_key] = float(tasktools.choice(rng, rel_s))
    ctx[irr_key] = float(tasktools.choice(rng, irr_s)) if irr_s is not None else 0.5
    # quadrant anti-bias: choose the sides of both features
    if qab and QAB_ON and not _st['validating'] and FORCE_STAGE is None:
        p = _qab_probs(task)
        q = int(rng.choice(4, p=p))
        rel_right, irr_right = [(True, False), (False, False), (False, True), (True, True)][q]
        # left_right: +1 puts the majority on the right/low side of pagan.py's convention (-1 = left/high)
        lr_rel, lr_irr = (1 if rel_right else -1), (1 if irr_right else -1)
        if task == 'loc':
            ctx['left_right_loc'], ctx['left_right_freq'] = lr_rel, lr_irr
        else:
            ctx['left_right_freq'], ctx['left_right_loc'] = lr_rel, lr_irr
        trial = _curr.get_condition(rng, dt, ctx)
        trial['quadrant'] = q + 1; trial['qab_p'] = float(p[q])
    elif QAB_ON and not _st['validating'] and FORCE_STAGE is None:
        # stages 1-6: the side anti-bias draws the relevant feature's side; the irrelevant side
        # (when present) stays random
        p_right = _side_p_right(task)
        lr_rel = 1 if rng.uniform() < p_right else -1
        ctx[rel_key.replace('strength_', 'left_right_')] = lr_rel
        trial = _curr.get_condition(rng, dt, ctx)
        trial['quadrant'] = 0; trial['qab_p'] = 0.25; trial['sab_p_right'] = p_right
    else:
        trial = _curr.get_condition(rng, dt, ctx)
        trial['quadrant'] = 0; trial['qab_p'] = 0.25
    if irr_s is None:
        # the irrelevant feature's pulses are absent in stages 1-4 (stimulus_mixing 0)
        if task == 'loc':
            trial['hml'] = np.zeros_like(trial['hml']); trial['integral_freq'] = 0
        else:
            trial['lmr'] = np.zeros_like(trial['lmr']); trial['integral_loc'] = 0
    trial['stage'] = s
    return trial


def get_step(rng, dt, trial, t, a):
    u, reward, status = _eff.get_step(rng, dt, trial, t, a)
    # quadrant reward scaling (water = 4 x p): applies to the terminal reward of a correct choice.
    # The engagement cost is charged on earlier timesteps and is not scaled, so an attempted
    # correct trial nets 4p - COST; see the docstring.
    if QAB_REWARD and trial.get('quadrant', 0) and reward > 0 and status.get('choice') is not None:
        reward = reward * 4.0 * trial['qab_p']
    return u, reward, status


def record_outcome(trial, status, reward):
    """Running accuracy per task (for blocks and promotion), the side and quadrant hit rates.

    _eff.record_outcome runs FIRST and unconditionally: it summarises the validation batch and
    early-returns when not validating, where this function early-returns when validating, so the
    two are complementary and both must see every trial. See the docstring."""
    _eff.record_outcome(trial, status, reward)

    if _st['validating'] or _curr.phase() != 'task' or trial.get('stage') is None:
        return
    if status.get('choice') is None:
        return                                   # declined or never decided: not an attempted trial
    task = trial['context']; hit = 1.0 if status.get('correct') else 0.0
    rec = _st['recent'][task]; rec.append(hit)
    if len(rec) > MIN_TRIALS: del rec[0]
    _st['n_attempted'] += 1; _st['block_len'] += 1
    s = stage(); tasks, _, _, _, switch_thr, qab = STAGES[s]
    # performance-gated task blocks, with the cap at twice the rats' median block length
    if len(tasks) > 1 and switch_thr is not None and _st['block_len'] >= MIN_BLOCK and (
            (len(rec) >= WIN and np.mean(rec[-WIN:]) >= switch_thr) or _st['block_len'] >= MAX_BLOCK.get(s, 10 ** 9)):
        _st['task'] = 'frq' if task == 'loc' else 'loc'; _st['block_len'] = 0
    # side anti-bias bookkeeping (stages without the quadrant rule)
    if not qab:
        ss = _st['side'][task]; ss *= _SAB_DECAY
        si = 1 if trial['correct_side'] == 'R' else 0; ss[0, si] += hit; ss[1, si] += 1.0
    # quadrant anti-bias bookkeeping
    if qab and trial.get('quadrant', 0):
        sh_sk = _st['q'][task]; sh_sk *= _DECAY
        qi = trial['quadrant'] - 1; sh_sk[0, qi] += hit; sh_sk[1, qi] += 1.0


def update(perf):
    # _eff.update advances the nose-poke phase clock itself (it calls _curr.update) and owns the
    # duration schedule, so it is called once and neither _curr.update nor _base.update is called
    # again here: a second call would make the pre-training phases pass twice as fast, or give the
    # schedule a second, independent clock. See the docstring.
    _eff.update(perf)
    if _curr.phase() != 'task' or FORCE_STAGE is not None:
        return
    s = stage(); tasks, _, _, thr, _, _ = STAGES[s]
    accs = {t: (np.mean(_st['recent'][t]) if len(_st['recent'][t]) >= 50 else np.nan) for t in tasks}
    print("[ pagan_hist_stages4 ] arm {} stage {}: {} attempted, acc {}, side P(right) {}".format(
        ARM, s, _st['n_attempted'], {t: (round(a, 3) if np.isfinite(a) else None) for t, a in accs.items()},
        {t: round(_side_p_right(t), 2) for t in tasks}))
    if thr is None:
        return
    if _st['n_attempted'] >= MIN_TRIALS and all(np.isfinite(a) and a >= thr for a in accs.values()):
        # the duration schedule lives in Stage 1 and runs once in all of training: no stage may be
        # left while the presentation window is still growing.
        if _base.stimulus < _base.stimulus_max:
            print("[ pagan_hist_stages4 ] stage {} at criterion but held: presentation {} ms, "
                  "duration schedule still running".format(s, _base.stimulus))
            return
        _st['pos'] += 1; _st['n_attempted'] = 0; _st['block_len'] = 0
        for t in _st['recent']: _st['recent'][t] = []
        print("[ pagan_hist_stages4 ] promoted to stage {}".format(stage()))


def terminate(perf):
    # only the full task can end the run, and _eff.terminate keeps tscost's guard against stopping
    # at criterion while the duration schedule is still running
    return stage() == 10 and _eff.terminate(perf)
