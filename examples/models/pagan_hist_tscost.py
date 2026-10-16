"""
Engagement cost, version 4 (16 Oct 2026): the cost accrues per timestep, and declining has its
own output.

Why. Version 2 (pagan_hist_effort2.py) gave the network a dedicated DECLINE action at the first
step and charged the whole cost of a trial at once. It trains, but engagement is sticky: a network
at chance keeps attempting about three quarters of trials for tens of thousands of trials, because
the decision to engage is taken once, before any evidence, and never revisited. Version 3
(pagan_hist_fixcost.py) made the cost per-step - a rat that does not want a trial simply does not
hold its nose in the port, and it can leave at any moment - but paid for it twice over: it removed
the DECLINE output and signalled withdrawal with CHOOSE-LEFT/RIGHT, and it made a stimulus-epoch
CHOOSE-LEFT/RIGHT an aborting violation, i.e. it restored the fixation requirement that was
troubleshot out of this project early on (inherited from the original pyrl). It failed for the
first reason: CHOOSE-LEFT/RIGHT served double duty as the choice action and the withdrawal action,
so every reward success that pushed "left" up pushed it up at step 0 too, where it means withdraw,
and the policy collapsed into declining everything (P(choice) 0.656 -> 0.160 *during* training).

This version keeps version 3's per-step cost and drops its two mistakes:

  - there is NO fixation requirement. A CHOOSE-LEFT/RIGHT before the decision epoch is a null
    action, masked from the reward function and non-aborting, exactly as in the base task
    (pagan.get_step). The network is never asked to choose FIXATE specifically.
  - withdrawal has its own output: DECLINE, the fourth action, as in version 2. The choice policy
    and the engagement policy are therefore learned on separate readout weights.

Rules:
  1. Four actions: FIXATE, CHOOSE-LEFT, CHOOSE-RIGHT, DECLINE. pyrl draws Wout last from the weight
     seed, so Win, bin and Wrec are bit-identical to the three-action arms and to effort2 at the
     same seed; only the readout differs. The output bias is NOT touched (version 3 biased it
     toward FIXATE with a hand-set `bout` vector): `bout` stays at the pyrl default 0 and Wout is
     the usual N(0, 1/N), so the four actions start near-uniform, averaging about 1/4 each, and
     the initial engagement rate is whatever that implies rather than a handcrafted number.
  2. A cost of PAGAN_TSCOST / 130 is charged on every timestep, from the first timestep of the
     trial, on which the action is not DECLINE, up to and not including the first timestep at which
     a choice becomes potent (`dec_t0`, the end of the stimulus-presentation period). 130 is the
     number of such timesteps at the full 1,300 ms stimulus, so holding to the decision epoch at
     full length costs exactly PAGAN_TSCOST (default 0.55, version 2's lump cost).
  3. DECLINE ends the trial at once, at any timestep, with reward 0 at that step and no choice (so,
     through pagan_hist, no history on the next trial). The timestep on which DECLINE is chosen is
     not itself charged, but every charge already made stands - there is no refund. DECLINE on the
     very first timestep therefore earns exactly 0; DECLINE at step k of the window has cost
     k x PAGAN_TSCOST / 130; DECLINE after the window has paid the whole of it.
  4. From `dec_t0` on, the base task exactly: no cost, CHOOSE-LEFT/RIGHT ends the trial at +1 for
     correct and 0 for incorrect, FIXATE lets the trial run on. A trial whose buffer ends with no
     choice pays no timeout penalty (PAGAN_TIMEOUT, default 0): it has already paid the whole
     window, and that is its penalty. Version 3's first launch died on the inherited -1 here.
  5. A late DECLINE (after the window) is worth -PAGAN_TSCOST, exactly what a wrong choice is
     worth, so it is never profitable and adds no exploitable escape. It does, however, let a
     network that has learned to withdraw when uncertain inflate P(correct|choice), which is why
     the checkpoint statistics below separate the three places a decline can happen.

The stimulus-presentation period and the duration schedule. During the nose-poke pre-training
(pagan_hist_curriculum: no pulses, the correct side set by the phase, the LOC channel as the go
cue) the presentation is ONE step: PAGAN_PRESTIM = 20 ms. That is 20 and not 10 because
pagan.get_condition pins the pulses to the indices actually shown (`stim_t0 = stimulus[0] + 1`),
so at 10 ms there are zero input bins and no input is ever delivered, not even the go cue - the
defect version 3 trained with. At 20 ms there is exactly one input bin, delivered on the second
timestep of the trial, `dec_t0` is 2, so choice actions are ignored on the first and second
timesteps and potent from the third, and the go-cue offset still coincides with `dec_t0` as it does
in the full task.

At the hand-off to the task phase the presentation length either jumps straight to the full
1,300 ms or starts at PAGAN_STIM0 and grows by pagan.STIMULUS_STEP (50 ms, i.e. five timesteps, one
checkpoint being five gradient batches) at one opportunity per checkpoint, taken according to
PAGAN_RAMP. Training starts the schedule at PAGAN_STIM0 = 50 ms (four input bins): the correct side
is the sign of the realized pulse count, which is zero on 69% of one-bin trials against 33% of
four-bin ones, so the accuracy ceiling for a perfect integrator is 0.833 at 50 ms against 0.653 at
20 ms - the performance-gated arm needs headroom over its 0.55 gate at the very first task
checkpoint, and at one bin it has almost none.

    off      no schedule: 1,300 ms from the hand-off on. The control.
    fixed    grow at every opportunity, regardless of P(choice) or P(correct).
    choice   grow iff P(choice) >= PAGAN_RAMP_PCHOICE (0.70). What pagan.update() does.
    perf     grow iff P(choice) >= PAGAN_RAMP_PCHOICE and P(correct|choice) > PAGAN_RAMP_PCORRECT
             (0.55).

Why the perf gate is 0.55 and not 0.50. The gate reads a single validation batch, and with
n_decision about 718 the standard error of P(correct) at chance is 0.019, so a threshold of 0.50
would be cleared by a network that is *exactly* at chance on 48.5% of checkpoints (exact
binomial) - the schedule would walk to the full length on validation noise in about 52
checkpoints, at roughly half the speed of `fixed` and for no performance-related reason. 0.55 is
cleared on 0.4% of chance checkpoints, so it is a real gate: it opens only once the network is
reliably better than chance. The consequence is accepted deliberately - this arm may hold a short
window for a long time, and a stalled perf arm is an interpretable result (a performance gate is
strictly more conservative than a choice gate), not a failure.

All three schedules share the same opportunity schedule and the same step, so the arms differ in
the criterion and in nothing else. The cost per timestep does not change with the length, so a
short trial costs little and the full one costs PAGAN_TSCOST. pagan.update() is deliberately NOT
called: this module replaces that single-criterion schedule rather than layering on it (version 3's
stages3 chained both and ended up with two independent length clocks).

PAGAN_STIM0 defaults to the full length, so an analysis rollout - which sets PAGAN_PHASE=task and
leaves PAGAN_STIM0 unset - measures every checkpoint on the full 1,300 ms task. Training passes
PAGAN_STIM0=50 explicitly; the module default is deliberately NOT 50, because a rollout that
silently measured a 50 ms window would corrupt every record. Early stopping is suppressed while the schedule is still running, so a
run cannot stop at criterion on a 70 ms stimulus.

Note that PAGAN_PRESTIM stays at 20 ms: the pre-training window is one presentation step by
design, and only the task-phase schedule starts at 50, so the hand-off steps 20 -> 50 ms.

Switches: PAGAN_TSCOST (0.55), PAGAN_RAMP (choice), PAGAN_STIM0 (1300), PAGAN_PRESTIM (20),
PAGAN_RAMP_PCHOICE (0.70), PAGAN_RAMP_PCORRECT (0.55), PAGAN_TIMEOUT (0), plus PAGAN_PRETRAIN=0 and
PAGAN_PHASE=task inherited from pagan_hist_curriculum.

Statistics printed at every checkpoint, over that checkpoint's validation batch only (so they
describe exactly the batch the schedule's gate reads): the five mutually exclusive outcomes - chose,
declined on the first timestep, declined inside the window, declined after the window, timed out -
the median withdrawal step of the in-window declines, the mean cost paid, the current presentation
length, and the gate's decision.
"""
from __future__ import division, print_function

import os
import sys

import numpy as np

from pyrl import tasktools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagan_hist_curriculum import *          # noqa: F401,F403
import pagan_hist_curriculum as _curr
import pagan as _base

#-----------------------------------------------------------------------------------------
# The action set. Redefined here, so the model harvests Nout = 4 from this namespace.
# `bout` is deliberately left alone (pyrl default 0, so bout = zeros(4)): initial action
# probabilities are left to Wout ~ N(0, 1/N), not handcrafted.
#-----------------------------------------------------------------------------------------

actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT', 'DECLINE')
DECLINE = actions['DECLINE']

#-----------------------------------------------------------------------------------------
# The cost
#-----------------------------------------------------------------------------------------

COST = float(os.environ.get('PAGAN_TSCOST', '0.55'))

# Charged timesteps at the full stimulus length: the stimulus epoch of a 1,300 ms trial spans
# indices 0..129 and dec_t0 is 130, so 130 timesteps are chargeable and the maximum cost of a
# trial is exactly COST.
FULL_STEPS = int(_base.stimulus_max / 10)       # 130 at dt = 10 ms
STEP_COST = COST / float(FULL_STEPS)

# No timeout penalty. The harness pays R_ABORTED (as R_TERMINAL) when the trial buffer ends with
# no choice; a timed-out trial here has already paid the whole window, which is its penalty. The
# inherited -1 is the trap that killed version 3's first launch. Note that R_ABORTED also supplies
# the critic's initial output bias when `baseline_bout` is unset (policygradient.py), and 0 is the
# better value for it anyway, the reward range being [-COST, +1].
R_ABORTED = float(os.environ.get('PAGAN_TIMEOUT', '0'))

#-----------------------------------------------------------------------------------------
# The stimulus-presentation length and its schedule
#-----------------------------------------------------------------------------------------

RAMP = os.environ.get('PAGAN_RAMP', 'choice')
if RAMP not in ('off', 'fixed', 'choice', 'perf'):
    raise ValueError("PAGAN_RAMP must be one of off|fixed|choice|perf, got {!r}".format(RAMP))

# One-step presentation during the pre-training phases; see the module docstring for why 20 and
# not 10 ms.
PRESTIM = int(os.environ.get('PAGAN_PRESTIM', '20'))

# Where the schedule starts in the task phase. Defaults to the full length so that analysis
# rollouts, which do not set it, measure the full task.
STIM0 = int(os.environ.get('PAGAN_STIM0', str(_base.stimulus_max)))

P_CHOICE  = float(os.environ.get('PAGAN_RAMP_PCHOICE',  '0.70'))
# 0.55, not 0.50: see the module docstring. At 0.50 a network at chance clears the gate on 48.5%
# of checkpoints and the schedule completes on validation noise alone; 0.55 is cleared on 0.4%.
P_CORRECT = float(os.environ.get('PAGAN_RAMP_PCORRECT', '0.55'))


def _task_length():
    """Presentation length in force for the first task-phase checkpoint."""
    return _base.stimulus_max if RAMP == 'off' else STIM0


# Set the length for the trials drawn before the first update() call. `task_started` stays False
# even when the phase is already 'task' (PAGAN_PRETRAIN=0, or PAGAN_PHASE=task in a rollout), so
# the first update() call takes the hand-off branch and grants no growth opportunity.
_base.stimulus = PRESTIM if _curr.phase() != 'task' else _task_length()

_ts = {'task_started': False, 'validating': False}

#-----------------------------------------------------------------------------------------
# Per-checkpoint statistics, over the validation batch only
#-----------------------------------------------------------------------------------------

_OUTCOMES = ('chose', 'declined_first', 'declined_window', 'declined_late', 'timed_out')


def _reset_stats():
    _ts['n'] = 0
    _ts['cost'] = 0.0
    _ts['window_steps'] = []
    for k in _OUTCOMES:
        _ts[k] = 0


_reset_stats()


def set_validation(flag):
    """pyrl tells the task when it is running a validation batch (policygradient.py)."""
    _ts['validating'] = bool(flag)


def record_outcome(trial, status, reward):
    """Called once per trial with its final status. Classify the validation trials into the five
    mutually exclusive outcomes and accumulate the cost paid. The cost is derived from the outcome
    rather than tallied step by step, which is exact given the rules above."""
    if not _ts['validating']:
        return

    dec_t0 = trial['dec_t0']
    _ts['n'] += 1

    if status.get('choice') is not None:
        _ts['chose'] += 1
        _ts['cost'] += dec_t0 * STEP_COST
    elif status.get('declined'):
        where = status.get('declined_where')
        step = status.get('declined_step', 0)
        if where == 'first':
            _ts['declined_first'] += 1
        elif where == 'window':
            _ts['declined_window'] += 1
            _ts['window_steps'].append(step)
            _ts['cost'] += step * STEP_COST
        else:
            _ts['declined_late'] += 1
            _ts['cost'] += dec_t0 * STEP_COST
    else:
        # the buffer ended with no choice and no decline: the whole window was paid for
        _ts['timed_out'] += 1
        _ts['cost'] += dec_t0 * STEP_COST

#-----------------------------------------------------------------------------------------
# The trial
#-----------------------------------------------------------------------------------------


def get_step(rng, dt, trial, t, a):
    """The action `a` was taken at time index t-1 (pagan.get_step's convention)."""
    step = t - 1
    in_window = step < trial['dec_t0']

    if a == DECLINE:
        # Withdrawal, available at every timestep. The trial ends now, this timestep is not
        # charged, and whatever was charged on earlier timesteps stands.
        if step == 0:
            where = 'first'
        elif in_window:
            where = 'window'
        else:
            where = 'decision'
        status = {'continue': False, 'declined': True, 'declined_step': step,
                  'declined_where': where,
                  'stim_choice': trial.get('stim_choice', False)}
        return np.zeros(len(inputs)), 0.0, status          # noqa: F405

    if in_window:
        # FIXATE and a null (non-aborting, reward-masked) CHOOSE-LEFT/RIGHT are charged alike:
        # the only way not to pay for a timestep is to decline it.
        u, reward, status = _curr.get_step(rng, dt, trial, t, a)
        return u, reward - STEP_COST, status

    return _curr.get_step(rng, dt, trial, t, a)            # decision epoch: the base rule, no cost

#-----------------------------------------------------------------------------------------
# Early stopping and the curriculum hook
#-----------------------------------------------------------------------------------------


def terminate(perf):
    if _base.stimulus < _base.stimulus_max:
        return False          # never stop at criterion while the duration schedule is still running
    return _curr.terminate(perf)


def _gate(perf):
    """(grow?, a one-line account of the decision)."""
    p_decision, p_correct = tasktools.correct_2AFC(perf)
    if RAMP == 'fixed':
        return True, 'fixed schedule'
    if RAMP == 'choice':
        ok = p_decision >= P_CHOICE
        return ok, 'P(choice) {:.3f} {} {:.2f}'.format(
            p_decision, '>=' if ok else '<', P_CHOICE)
    ok_c = p_decision >= P_CHOICE
    ok_p = p_correct > P_CORRECT
    return (ok_c and ok_p), 'P(choice) {:.3f} {} {:.2f}, P(correct|choice) {:.3f} {} {:.2f}'.format(
        p_decision, '>=' if ok_c else '<', P_CHOICE,
        p_correct, '>' if ok_p else '<=', P_CORRECT)


def update(perf):
    """Called once per checkpoint, with that checkpoint's own validation Performance."""
    _curr.update(perf)                      # advance the pre-training phase clock

    note = ''
    if _curr.phase() != 'task':
        _base.stimulus = PRESTIM
        note = 'pre-training phase {}, presentation held at {} ms'.format(
            _curr.phase(), _base.stimulus)
    elif not _ts['task_started']:
        _ts['task_started'] = True
        _base.stimulus = _task_length()
        note = 'task phase begins, presentation set to {} ms ({})'.format(
            _base.stimulus, 'no schedule' if RAMP == 'off' else 'schedule starts here')
    elif RAMP == 'off':
        note = 'no schedule'
    elif _base.stimulus >= _base.stimulus_max:
        note = 'schedule complete'
    else:
        grow, why = _gate(perf)
        if grow:
            was = _base.stimulus
            _base.stimulus = min(_base.stimulus + _base.STIMULUS_STEP, _base.stimulus_max)
            note = '{} -> presentation {} -> {} ms'.format(why, was, _base.stimulus)
        else:
            note = '{} -> presentation held at {} ms'.format(why, _base.stimulus)

    _report(note)
    _reset_stats()


def _report(note):
    n = _ts['n']
    steps = int(_base.stimulus / 10)
    head = ('[ pagan_hist_tscost ] COST {} ({:.6f}/step) ramp={} presentation {} ms '
            '({} charged steps)'.format(COST, STEP_COST, RAMP, _base.stimulus, steps))
    if not n:
        print(head)
        print('   no validation trials recorded; {}'.format(note))
        return
    ws = _ts['window_steps']
    print(head)
    print('   of {} val trials: chose {:.3f}, declined-first {:.3f}, declined-in-window {:.3f} '
          '(median step {}), declined-late {:.3f}, timed out {:.3f}'
          .format(n, _ts['chose'] / n, _ts['declined_first'] / n, _ts['declined_window'] / n,
                  int(np.median(ws)) if ws else '-', _ts['declined_late'] / n,
                  _ts['timed_out'] / n))
    print('   mean cost paid {:.4f}; {}'.format(_ts['cost'] / n, note))
