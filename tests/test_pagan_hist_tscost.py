# -*- coding: utf-8 -*-
"""
Offline checks for examples/models/pagan_hist_tscost.py (the per-timestep engagement cost,
version 4 -- see section 9 of README_for_elise.md and the model's own docstring).

Run from the repository root under the Python 2.7 environment:

  python tests/test_pagan_hist_tscost.py            # reward structure, this file
  python tests/test_pagan_hist_tscost_ramp.py off   # duration schedule, one mode per process

Neither imports Theano: they walk get_step and drive update() directly, so both run in
seconds. PAGAN_RAMP is read at import, which is why the schedule test takes the mode as an
argument and is a separate file.
"""
from __future__ import division, print_function

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('PYRL_REPO',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'examples', 'models'))

os.environ['PAGAN_PRETRAIN'] = '1'
os.environ['PAGAN_RAMP'] = 'choice'
os.environ['PAGAN_STIM0'] = '20'

import pagan as _base                                      # noqa: E402
import pagan_hist_tscost as M                              # noqa: E402

DT = 10
FAIL = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        FAIL.append(msg)


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def trial_at(stimulus, phase='task', seed=1):
    """A condition generated at the given presentation length."""
    _base.stimulus = stimulus
    M._curr.set_phase(phase)
    rng = np.random.RandomState(seed)
    return M.get_condition(rng, DT)


def walk(trial, action_at, tmax=331):
    """Run the trial with action_at(step) -> action index. Returns (rewards, statuses)."""
    rng = np.random.RandomState(0)
    rewards, statuses, us = [], [], []
    for step in range(tmax):
        a = action_at(step)
        u, r, status = M.get_step(rng, DT, trial, step + 1, a)
        rewards.append(r)
        statuses.append(status)
        us.append(u)
        if not status['continue']:
            break
    return rewards, statuses, us


FIX = M.actions['FIXATE']
LEFT = M.actions['CHOOSE-LEFT']
RIGHT = M.actions['CHOOSE-RIGHT']
DEC = M.actions['DECLINE']

print('STEP_COST = %.9f  (COST %.3f / %d)' % (M.STEP_COST, M.COST, M.FULL_STEPS))
print()

# ---------------------------------------------------------------- action set and output bias
print('action set and output bias')
check(len(M.actions) == 4, 'four actions, Nout = 4')
check(list(M.actions) == ['FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT', 'DECLINE'],
      'action order FIXATE, CHOOSE-LEFT, CHOOSE-RIGHT, DECLINE (Wout drawn last, seeds aligned)')
check('bout' not in vars(M), 'bout not defined: initial action probabilities not handcrafted')
check(close(M.R_ABORTED, 0.0), 'R_ABORTED = 0, no timeout penalty')
check(M.FULL_STEPS == 130 and close(M.STEP_COST, 0.55 / 130.0),
      'STEP_COST = 0.55/130, max cost exactly 0.55')
print()

# ---------------------------------------------------------------- epoch geometry
print('epoch geometry')
t20 = trial_at(20)
check(t20['dec_t0'] == 2, 'at 20 ms: dec_t0 == 2 (choices ignored on steps 0 and 1)')
check(len(t20['lmr']) == 1, 'at 20 ms: exactly one input bin')

t1300 = trial_at(1300)
check(t1300['dec_t0'] == 130, 'at 1300 ms: dec_t0 == 130')
check(len(t1300['lmr']) == 129, 'at 1300 ms: 129 input bins')

# the go cue: u produced at get_step t is consumed by the policy at rollout step t
_, _, us = walk(trial_at(20, phase='fixed'), lambda s: FIX)
LOC = M.inputs['LOC']
check(us[0][LOC] == 1, 'at 20 ms: LOC cue in the bin written at get_step t=1 (seen on step 1)')
check(us[1][LOC] == 0, 'at 20 ms: LOC cue off in the bin seen at step dec_t0 = 2')

_, _, us = walk(trial_at(1300, phase='fixed'), lambda s: FIX)
check(us[128][LOC] == 1, 'at 1300 ms: LOC cue on in the bin seen at step 129')
check(us[129][LOC] == 0, 'at 1300 ms: LOC cue off in the bin seen at step dec_t0 = 130')
print()

# ---------------------------------------------------------------- DECLINE on the first step
print('DECLINE on the first timestep')
tr = trial_at(1300)
rewards, statuses, _ = walk(tr, lambda s: DEC)
check(len(rewards) == 1 and not statuses[0]['continue'], 'trial ends at once')
check(close(sum(rewards), 0.0), 'total trial reward exactly 0')
check(statuses[0]['declined_where'] == 'first', "declined_where == 'first'")
check('choice' not in statuses[0], 'no choice in status: no history for the next trial')
print()

# ---------------------------------------------------------------- DECLINE inside the window
print('DECLINE inside the window')
for k in (1, 5, 63, 129):
    tr = trial_at(1300)
    rewards, statuses, _ = walk(tr, lambda s, k=k: DEC if s == k else FIX)
    check(close(sum(rewards), -k * M.STEP_COST),
          'declined at step %3d -> total %.9f == -%d x STEP_COST' % (k, sum(rewards), k))
    check(statuses[-1]['declined_where'] == 'window' and statuses[-1]['declined_step'] == k,
          "  declined_where == 'window', declined_step == %d" % k)
print()

# ---------------------------------------------------------------- holding to a choice
print('holding the whole window, then choosing')
tr = trial_at(1300)
side = LEFT if tr['correct_side'] == 'L' else RIGHT
rewards, statuses, _ = walk(tr, lambda s: side if s == tr['dec_t0'] else FIX)
check(close(sum(rewards), 1.0 - 0.55), 'correct choice at dec_t0 -> 1 - 0.55 = %.9f (130 charges)'
      % sum(rewards))
check(statuses[-1].get('correct') is True, 'status reports correct')

wrong = RIGHT if tr['correct_side'] == 'L' else LEFT
rewards, _, _ = walk(trial_at(1300), lambda s: wrong if s == tr['dec_t0'] else FIX)
check(close(sum(rewards), -0.55), 'wrong choice at dec_t0 -> -0.55 = %.9f' % sum(rewards))

# a choice taken later in the decision epoch is still free
rewards, _, _ = walk(tr, lambda s: side if s == tr['dec_t0'] + 40 else FIX)
check(close(sum(rewards), 1.0 - 0.55), 'correct choice 40 steps into the decision epoch -> same')
print()

# ---------------------------------------------------------------- null choices in the window
print('CHOOSE-LEFT/RIGHT inside the window is null, non-aborting, and charged')
tr = trial_at(1300)
rewards, statuses, _ = walk(tr, lambda s: LEFT if s == 5 else (side if s == tr['dec_t0'] else FIX))
check(statuses[5]['continue'] is True, 'a stimulus-epoch CHOOSE-LEFT does not end the trial')
check(close(rewards[5], -M.STEP_COST), 'it is charged STEP_COST, like FIXATE')
check(statuses[-1]['stim_choice'] is True, 'status reports stim_choice for the diagnostic')
check(close(sum(rewards), 1.0 - 0.55), 'total unchanged: still exactly 130 charges')
print()

# ---------------------------------------------------------------- DECLINE after the window
print('DECLINE after the window')
tr = trial_at(1300)
rewards, statuses, _ = walk(tr, lambda s: DEC if s == tr['dec_t0'] + 3 else FIX)
check(not statuses[-1]['continue'], 'trial ends')
check('choice' not in statuses[-1], 'no choice recorded')
check(statuses[-1]['declined_where'] == 'decision', "declined_where == 'decision'")
check(close(sum(rewards), -0.55), 'total -0.55: the window was paid for and is not refunded')
print()

# ---------------------------------------------------------------- timeout
print('timeout (buffer ends with no choice and no decline)')
tr = trial_at(1300)
rewards, statuses, _ = walk(tr, lambda s: FIX)
check(len(rewards) == 331, 'the trial runs to the end of the buffer')
check(close(sum(rewards), -0.55), 'total -0.55: the window, no extra penalty from get_step')
print()

# ---------------------------------------------------------------- the pre-training window
print('the one-step pre-training window (20 ms)')
tr = trial_at(20, phase='fixed')
check(close(sum(walk(tr, lambda s: DEC)[0]), 0.0), 'DECLINE on step 0 -> 0')
check(close(sum(walk(tr, lambda s: DEC if s == 1 else FIX)[0]), -M.STEP_COST),
      'DECLINE on step 1 -> -1 x STEP_COST')
rewards, statuses, _ = walk(tr, lambda s: (LEFT if tr['correct_side'] == 'L' else RIGHT)
                            if s == 2 else FIX)
check(close(sum(rewards), 1.0 - 2 * M.STEP_COST),
      'correct choice on step 2 -> 1 - 2 x STEP_COST = %.9f' % sum(rewards))
check(statuses[-1].get('correct') is True, 'the choice on the third timestep is potent')
rewards, statuses, _ = walk(tr, lambda s: (LEFT if s == 1 else FIX))
check(statuses[1]['continue'] is True, 'a choice on the second timestep is ignored')
check(np.all(tr['lmr'] == 0) and np.all(tr['hml'] == 0), 'no pulses in pre-training')
print()

# ---------------------------------------------------------------- outcome bookkeeping
print('record_outcome bookkeeping over a validation batch')
M.set_validation(True)
M._reset_stats()
tr13 = trial_at(1300)
cases = [
    ({'choice': 'L', 'correct': True}, 130 * M.STEP_COST),
    ({'declined': True, 'declined_where': 'first', 'declined_step': 0}, 0.0),
    ({'declined': True, 'declined_where': 'window', 'declined_step': 40}, 40 * M.STEP_COST),
    ({'declined': True, 'declined_where': 'decision', 'declined_step': 133},
     130 * M.STEP_COST),
    ({'reward': 0.0}, 130 * M.STEP_COST),
]
for status, _c in cases:
    M.record_outcome(tr13, status, 0.0)
check(M._ts['n'] == 5, 'five trials counted')
check([M._ts[k] for k in M._OUTCOMES] == [1, 1, 1, 1, 1],
      'one of each outcome: chose, declined_first, declined_window, declined_late, timed_out')
check(close(M._ts['cost'], sum(c for _, c in cases)), 'cost accumulated exactly')
check(M._ts['window_steps'] == [40], 'in-window withdrawal step recorded')
M.set_validation(False)
M._reset_stats()
M.record_outcome(tr13, {'choice': 'L'}, 1.0)
check(M._ts['n'] == 0, 'training trials are not counted')
print()

print('=' * 70)
if FAIL:
    print('%d FAILURES' % len(FAIL))
    for f in FAIL:
        print('  - ' + f)
    sys.exit(1)
print('all reward-structure checks passed')
