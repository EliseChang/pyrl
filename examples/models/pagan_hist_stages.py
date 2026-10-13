"""
The rats' TaskSwitch stage curriculum, on top of the effort network (pagan_hist_effort2) and the
nose-poke pre-training (pagan_hist_curriculum).

Why (part 6 of the Gatsby talk). Effort networks need the early training stages to be motivated,
and the rest of the curriculum to stay motivated as the task gets harder. The networks so far got the nose-poke pre-training and then the full task.
Here they get what the Princeton rats get after the nose-poke stage, as read from the protocol
(learning_rats_RNNs/Protocol/@TaskSwitch4new/TrainingSection.m, StimulusSection.m) and from
the pooled trial tables of 208 rats (/mnt/big/rat_behavior/derived/stage_statistics.csv):

  stage  what is on                                                   promotion (rats)
  1-2    LOC task only, easiest evidence (gamma 4), no FRQ pulses      >150 trials, >0.75 correct
  3-4    LOC and FRQ tasks in performance-gated blocks, each pure      both tasks >0.7
         (the other feature's pulses absent), gamma 4
  5-6    both features present, the irrelevant one uninformative       both >0.7
         (gamma 0 = 50/50 pulses)
  7      irrelevant becomes informative: gamma in {1, 4}; quadrant     >0.68
         anti-bias on
  8      irrelevant in {1, 2.5, 4}                                      >0.68
  9      relevant harder too: {2.5, 4}; irrelevant {1, 2.5, 4}          >0.68
  10     both in {1, 2.5, 4}: the full task (Elise's strengths)         (end)

Gamma to pulse strength. In Elise's task a pulse is left (or high) with probability `strength`
or 1 - strength, strengths = [0.25, 0.35, 0.45] (0.25 easiest). The JAX code of the 2025 paper
calls the same quantity gamma with 0.75 the easiest and 0.5 uninformative; the rats' gamma 4 is
their easiest and gamma 1 their hardest. Mapping used here (said so on the explorer):
    gamma 4 -> 0.25, 2.5 -> 0.35, 1 -> 0.45, 0 (uninformative) -> 0.5.

Blocks. From stage 3 the task is set by performance-gated blocks as in the protocol
(task_switch_min_perf): the context switches when the running accuracy over the last 30
attempted trials reaches the stage's switch threshold (0.8 for stages 3-6, 0.7 from 7) and
the block is at least 20 trials long.

Promotion. Checked at every checkpoint (update(perf)) on the running accuracy of training
trials per task (last 150 attempted trials of the stage): a stage is left when at least 150
trials were attempted in it and every task on in that stage is above the threshold. There is
no demotion. Stage lengths are therefore the network's, not the rats' medians.

Quadrant anti-bias (stages 7+, StimulusSection.m). Per task, a kernel-weighted hit rate per
quadrant of the stimulus matrix (relevant side x irrelevant side: 1 = incongruent, relevant
says right; 2 = congruent left; 3 = incongruent, relevant says left; 4 = congruent right),
kernel exp(-lag/tau) with tau 50, normalised over the four and rounded to 2 decimals; the next
quadrant is drawn from softmax(-fracs x beta x 1.5) for the LOC task and softmax(-fracs x beta)
for the FRQ task, beta 5. The protocol also scales the reward by 4 x p(quadrant) (water) and
the delay; here the reward scaling is a switch (PAGAN_QAB_REWARD=1, default on) and the delay
is not modelled.

Arms (PAGAN_STAGES):
    none     the full task from the first trial after the nose-poke phases (= pagan_hist_effort2)
    short    stages 1-6 then stage 10 (accumulation-only first, as on the Cosyne poster); no quadrant anti-bias
    full     stages 1-10 with the quadrant anti-bias from stage 7
Analysis: PAGAN_STAGE=10 forces the full task and PAGAN_ANTIBIAS=0 switches the quadrant rule
off, so every arm is measured on the same distribution. The outcome hook and the validation
flag come from the working copy of pyrl (11 Oct 2026).
"""
from __future__ import division, print_function

import os
import sys

import numpy as np

from pyrl import tasktools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagan_hist_effort2 import *          # noqa: F401,F403  (effort2 on top of the curriculum)
import pagan_hist_effort2 as _eff
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
if ARM != 'full':
    # the quadrant anti-bias belongs to the full ladder only (it is what the full arm adds over the
    # short one); the short arm's stage 10 is the plain full task
    STAGES[10] = STAGES[10][:5] + (False,)
MIN_TRIALS, WIN, MIN_BLOCK = 150, 30, 20

_st = {'pos': 0, 'n_attempted': 0, 'recent': {'loc': [], 'frq': []}, 'task': 'loc', 'block_len': 0,
       'validating': False, 'started': False,
       # quadrant anti-bias sums per task: [sum_hit, sum_kernel] x 4 quadrants, decayed per trial
       'q': {'loc': np.zeros((2, 4)), 'frq': np.zeros((2, 4))}, 'n_flipped': 0}
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

def _quadrant(rel_right, irr_right):
    # 1: incongruent, relevant right; 2: congruent left; 3: incongruent, relevant left; 4: congruent right
    if rel_right and not irr_right: return 0
    if (not rel_right) and (not irr_right): return 1
    if (not rel_right) and irr_right: return 2
    return 3

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
    # quadrant reward scaling (water = 4 x p): applies to the terminal reward of a correct choice
    if QAB_REWARD and trial.get('quadrant', 0) and reward > 0 and status.get('choice') is not None:
        reward = reward * 4.0 * trial['qab_p']
    return u, reward, status

def record_outcome(trial, status, reward):
    """Running accuracy per task (for blocks and promotion) and the quadrant hit rates."""
    if _st['validating'] or _curr.phase() != 'task' or trial.get('stage') is None:
        return
    if status.get('choice') is None:
        return                                   # declined or never decided: not an attempted trial
    task = trial['context']; hit = 1.0 if status.get('correct') else 0.0
    rec = _st['recent'][task]; rec.append(hit)
    if len(rec) > MIN_TRIALS: del rec[0]
    _st['n_attempted'] += 1; _st['block_len'] += 1
    s = stage(); tasks, _, _, _, switch_thr, qab = STAGES[s]
    # performance-gated task blocks
    if len(tasks) > 1 and switch_thr is not None and _st['block_len'] >= MIN_BLOCK and len(rec) >= WIN and np.mean(rec[-WIN:]) >= switch_thr:
        _st['task'] = 'frq' if task == 'loc' else 'loc'; _st['block_len'] = 0
    # quadrant anti-bias bookkeeping
    if qab and trial.get('quadrant', 0):
        sh_sk = _st['q'][task]; sh_sk *= _DECAY
        qi = trial['quadrant'] - 1; sh_sk[0, qi] += hit; sh_sk[1, qi] += 1.0

def update(perf):
    # effort2.update advances the curriculum clock itself (it calls _curr.update), so it is
    # called once and _curr.update is not called again here: a second call would make the
    # nose-poke phases pass twice as fast as in every other arm
    _eff.update(perf)
    if _curr.phase() != 'task' or FORCE_STAGE is not None:
        return
    s = stage(); tasks, _, _, thr, _, _ = STAGES[s]
    accs = {t: (np.mean(_st['recent'][t]) if len(_st['recent'][t]) >= 50 else np.nan) for t in tasks}
    print("[ pagan_hist_stages ] arm {} stage {}: {} attempted, acc {}".format(
        ARM, s, _st['n_attempted'], {t: (round(a, 3) if np.isfinite(a) else None) for t, a in accs.items()}))
    if thr is None:
        return
    if _st['n_attempted'] >= MIN_TRIALS and all(np.isfinite(a) and a >= thr for a in accs.values()):
        _st['pos'] += 1; _st['n_attempted'] = 0; _st['block_len'] = 0
        for t in _st['recent']: _st['recent'][t] = []
        print("[ pagan_hist_stages ] promoted to stage {}".format(stage()))

def terminate(perf):
    # only the full task can end the run (criterion as in the base task)
    return stage() == 10 and _curr.terminate(perf)
