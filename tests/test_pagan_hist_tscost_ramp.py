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

REPO = os.environ.get('PYRL_REPO',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'examples', 'models'))

MODE = sys.argv[1]
PRETRAIN0 = len(sys.argv) > 2 and sys.argv[2] == 'pretrain0'

# The schedule start, 50 ms in training (four input bins: see the model docstring on the accuracy
# ceiling). Overridable so the test can be run against another start.
STIM0 = int(os.environ.get('PAGAN_STIM0', '50'))

os.environ['PAGAN_RAMP'] = MODE
os.environ['PAGAN_STIM0'] = str(STIM0)
os.environ['PAGAN_PRETRAIN'] = '0' if PRETRAIN0 else '1'

import pagan as _base                                      # noqa: E402
import pagan_hist_tscost as M                              # noqa: E402

FAIL = []


def check(ok, msg):
    if not ok:
        FAIL.append(msg)
        print('  FAIL ' + msg)
    return ok


class Perf(object):
    """Minimal stand-in for Performance2AFC: tasktools.correct_2AFC reads these three.

    `n_correct` may be given as an exact count, because a rounded p_correct cannot land exactly
    on a gate threshold when n_decision is odd (and Python 2.7 rounds half away from zero)."""

    def __init__(self, p_decision, p_correct, n=720, n_correct=None):
        self.n_trials = n
        self.n_decision = int(round(p_decision * n))
        self.n_correct = int(round(p_correct * self.n_decision)) if n_correct is None else n_correct


# Silence the per-checkpoint reports; we are testing the state machine, not the print.
M._report = lambda note: None

N_CALLS = 45
PRE_CALLS = 0 if PRETRAIN0 else 6          # update() calls still in a pre-training phase
HANDOFF = PRE_CALLS + 1                    # the call at which the task phase begins

# A policy that always satisfies both gates, so `fixed`/`choice`/`perf` all grow at every
# opportunity -- the three then differ only where the gates are made to fail, below.
log = []
for i in range(1, N_CALLS + 1):
    M.update(Perf(0.99, 0.99))
    log.append((i, M._curr.phase(), _base.stimulus))

print('%s%s: phase/length by update call' % (MODE, ' (pretrain0)' if PRETRAIN0 else ''))
print('  ' + ', '.join('%d:%s@%d' % (i, p[:4], s) for i, p, s in log[:min(12, N_CALLS)]) + ' ...')

# The pre-training window stays one presentation step (PAGAN_PRESTIM = 20 ms) whatever the
# schedule start is, so the hand-off steps 20 -> STIM0.
pre = [s for i, p, s in log if i <= PRE_CALLS]
check(all(s == M.PRESTIM for s in pre),
      'presentation pinned at PRESTIM = %d ms for all %d pre-training checkpoints (got %s)'
      % (M.PRESTIM, PRE_CALLS, sorted(set(pre))))
check(all(p != 'task' for i, p, s in log if i <= PRE_CALLS),
      'phase is still pre-training for those calls')
check(log[HANDOFF - 1][1] == 'task', 'task phase begins at update call %d' % HANDOFF)

handoff_len = log[HANDOFF - 1][2]
if MODE == 'off':
    check(handoff_len == 1300, 'off: jumps to 1300 ms at the hand-off (got %d)' % handoff_len)
    check(all(s == 1300 for i, p, s in log[HANDOFF - 1:]), 'off: never moves again')
else:
    import math
    growths = int(math.ceil((1300 - STIM0) / 50.0))
    check(handoff_len == STIM0, '%s: starts the schedule at PAGAN_STIM0 = %d ms (got %d)'
          % (MODE, STIM0, handoff_len))
    check(log[HANDOFF][2] == STIM0 + 50, '%s: first growth is +50 ms, to %d (got %d)'
          % (MODE, STIM0 + 50, log[HANDOFF][2]))
    full = [i for i, p, s in log if s == 1300]
    check(bool(full) and full[0] == HANDOFF + growths,
          '%s: reaches 1300 ms after %d growths, at call %d (got %s)'
          % (MODE, growths, HANDOFF + growths, full[0] if full else None))
    check(all(s == 1300 for i, p, s in log if i >= HANDOFF + growths),
          '%s: stays at 1300 ms once there' % MODE)

# ---------------------------------------------------------------- the gates
if MODE in ('choice', 'perf'):
    M._base.stimulus = 320
    M.update(Perf(0.69, 0.99))
    check(_base.stimulus == 320, '%s: P(choice) 0.69 < 0.70 -> held' % MODE)
    M.update(Perf(0.70, 0.99))
    check(_base.stimulus == 370, '%s: P(choice) 0.70 >= 0.70 -> grew' % MODE)

if MODE == 'perf':
    check(abs(M.P_CORRECT - 0.55) < 1e-9,
          'perf: the default gate is 0.55, not 0.50 (got %.3f) -- at 0.50 a chance network '
          'clears it on 48.5%% of checkpoints' % M.P_CORRECT)
    # exactly at the threshold: held, because the test is strict (>)
    at = int(round(M.P_CORRECT * 720))
    M._base.stimulus = 320
    M.update(Perf(1.0, None, n=720, n_correct=at))
    check(_base.stimulus == 320, 'perf: P(correct) exactly %.3f is not > %.3f -> held'
          % (at / 720.0, M.P_CORRECT))
    M.update(Perf(1.0, None, n=720, n_correct=at + 1))
    check(_base.stimulus == 370, 'perf: P(correct) %.4f > %.3f -> grew'
          % ((at + 1) / 720.0, M.P_CORRECT))
    M._base.stimulus = 320
    M.update(Perf(0.60, 0.99))
    check(_base.stimulus == 320, 'perf: good accuracy but P(choice) 0.60 -> held')

if MODE == 'choice':
    M._base.stimulus = 320
    M.update(Perf(0.99, 0.20))
    check(_base.stimulus == 370, 'choice: ignores P(correct) (0.20) and grows on P(choice)')

if MODE == 'fixed':
    M._base.stimulus = 320
    M.update(Perf(0.0, 0.0))
    check(_base.stimulus == 370, 'fixed: grows even at P(choice) = 0')

# ---------------------------------------------------------------- early stopping guard
M._base.stimulus = 1290
check(M.terminate(Perf(1.0, 1.0)) is False,
      'terminate() is False below the full length even at perfect performance')
M._base.stimulus = 1300
M._curr.set_phase('task')
check(M.terminate(Perf(1.0, 1.0)) is True, 'terminate() fires at criterion at the full length')
check(M.terminate(Perf(0.5, 1.0)) is False, 'terminate() is False below 0.99 decided')

print('  %s' % ('%d FAILURES' % len(FAIL) if FAIL else 'all schedule checks passed'))
sys.exit(1 if FAIL else 0)
