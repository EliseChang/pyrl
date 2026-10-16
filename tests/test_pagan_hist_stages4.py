# -*- coding: utf-8 -*-
"""
Offline checks for examples/models/pagan_hist_stages4.py (the TaskSwitch stage ladder on the
per-timestep engagement cost -- see section 11 of README_for_elise.md and the model's docstring).

Run from the repository root under the Python 2.7 environment:

  python tests/test_pagan_hist_stages4.py              # arm full (default)
  PAGAN_STAGES=short python tests/test_pagan_hist_stages4.py

No Theano: the ladder is driven directly through update()/record_outcome()/get_condition(). The
emphasis is on hook chaining, because that is the bug class that killed pagan_hist_stages3 -- a
three-line copy of the ladder that silently shadowed its parent's hooks.
"""
from __future__ import division, print_function

import os
import sys

import numpy as np

REPO = os.environ.get('PYRL_REPO',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'examples', 'models'))

os.environ.setdefault('PAGAN_STAGES', 'full')
os.environ.setdefault('PAGAN_RAMP', 'choice')
os.environ['PAGAN_STIM0'] = '50'

import pagan as _base                                      # noqa: E402
import pagan_hist_curriculum as _curr                      # noqa: E402
import pagan_hist_tscost as _eff                           # noqa: E402
import pagan_hist_stages4 as M                             # noqa: E402

DT = 10
FAIL = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        FAIL.append(msg)


class Perf(object):
    """Minimal stand-in for Performance2AFC; tasktools.correct_2AFC reads these three."""

    def __init__(self, p_decision=1.0, p_correct=1.0, n=360, n_correct=None):
        self.n_trials = n
        self.n_decision = int(round(p_decision * n))
        self.n_correct = int(round(p_correct * self.n_decision)) if n_correct is None else n_correct


def reset_ladder(pos=0, task='loc'):
    M._st['pos'] = pos
    M._st['n_attempted'] = 0
    M._st['block_len'] = 0
    M._st['task'] = task
    for t in M._st['recent']:
        M._st['recent'][t] = []
    M._st['validating'] = False
    _eff.set_validation(False)


def a_trial(stage=1, context='loc', correct_side='L', dec_t0=130, quadrant=0):
    return {'stage': stage, 'context': context, 'correct_side': correct_side,
            'dec_t0': dec_t0, 'quadrant': quadrant, 'qab_p': 0.25}


print('arm %s, ORDER %s' % (M.ARM, M.ORDER))
print()

# ---------------------------------------------------------------- inheritance survived
print('the cost model is inherited intact')
check(len(M.actions) == 4, 'four actions (Nout = 4): %s' % list(M.actions))
check(M.actions is _eff.actions, 'the action map is tscost\'s own object, not a redefinition')
check(abs(M.R_ABORTED - 0.0) < 1e-12, 'R_ABORTED = 0, no timeout penalty')
check('bout' not in vars(M), 'bout not defined: initial action probabilities not handcrafted')
check(abs(M.STEP_COST - 0.55 / 130.0) < 1e-12, 'STEP_COST = 0.55/130 carried through')
check(M.n_validation == 720, 'n_validation default inherited (720; the runs override by --config)')
print()

# ---------------------------------------------------------------- hook chaining
print('record_outcome chains to tscost (the stages3 bug)')
reset_ladder()
_eff._reset_stats()
M.set_validation(True)
check(_eff._ts['validating'] is True, 'set_validation propagates to tscost')
check(M._st['validating'] is True, 'and sets the ladder\'s own flag')
M.record_outcome(a_trial(), {'choice': 'L', 'correct': True}, 0.45)
M.record_outcome(a_trial(), {'declined': True, 'declined_where': 'window', 'declined_step': 40}, 0.0)
check(_eff._ts['n'] == 2, 'tscost saw both validation trials (n = %d, must not be 0)' % _eff._ts['n'])
check(_eff._ts['chose'] == 1 and _eff._ts['declined_window'] == 1,
      'tscost classified them: chose %d, declined_window %d'
      % (_eff._ts['chose'], _eff._ts['declined_window']))
check(M._st['n_attempted'] == 0,
      'the ladder ignored them (validation trials do not count toward promotion)')

M.set_validation(False)
_eff._reset_stats()
_curr.set_phase('task')
M.record_outcome(a_trial(), {'choice': 'L', 'correct': True}, 0.45)
check(_eff._ts['n'] == 0, 'tscost ignores training trials (n = %d)' % _eff._ts['n'])
check(M._st['n_attempted'] == 1, 'the ladder counted the training trial')
check(M._st['recent']['loc'] == [1.0], 'and recorded the hit from status[correct]')
print()

print('record_outcome: only choices count as attempted')
reset_ladder()
_curr.set_phase('task')
for status in ({'declined': True, 'declined_where': 'first', 'declined_step': 0},
               {'declined': True, 'declined_where': 'window', 'declined_step': 7},
               {'declined': True, 'declined_where': 'decision', 'declined_step': 133},
               {'reward': 0.0}):
    M.record_outcome(a_trial(), status, 0.0)
check(M._st['n_attempted'] == 0,
      'all four non-choice outcomes excluded (declined first/window/late, timed out)')
check(M._st['recent']['loc'] == [], 'and none entered the accuracy window')
# hit comes from status['correct'], not from reward, so the cost and the quadrant scaling cannot
# corrupt it
M.record_outcome(a_trial(), {'choice': 'R', 'correct': True}, -0.1)
check(M._st['recent']['loc'] == [1.0], 'a correct choice with negative net reward still scores 1')
M.record_outcome(a_trial(), {'choice': 'R', 'correct': False}, 0.9)
check(M._st['recent']['loc'] == [1.0, 0.0], 'a wrong choice with positive reward still scores 0')
print()

print('a block the network declines its way through still ends (the stage-3 deadlock)')
# v2 incremented block_len only on ATTEMPTED trials, so under the per-timestep cost a block the
# network refuses never reached the cap and never ended: the first stages4 smoke sat in stage 3
# for 15,000 trials with the attempt counter frozen at 60. The block clock now counts presented
# trials.
_curr.set_phase('task')
reset_ladder()
S3 = 3 if 3 in M.ORDER else M.ORDER[0]
saved = M.stage; M.stage = lambda: S3
try:
    tasks = M.STAGES[S3][0]
    if len(tasks) > 1:
        M._st['task'] = 'loc'
        cap = M.MAX_BLOCK[S3]
        # every trial declined: nothing is ever attempted
        for _ in range(cap + 5):
            M.record_outcome(a_trial(stage=S3, context=M._st['task']),
                             {'declined': True, 'declined_where': 'window', 'declined_step': 11}, 0.0)
            if M._st['task'] != 'loc':
                break
        check(M._st['task'] == 'frq',
              'the block switched to frq after the cap (%d presented trials), with zero attempted'
              % cap)
        check(M._st['n_attempted'] == 0, 'and nothing was counted as attempted')
        check(M._st['block_len'] == 0, 'and the block clock reset')
    else:
        check(True, 'arm %s has no two-task stage; block cap not applicable' % M.ARM)
finally:
    M.stage = saved
print()

print('the performance-gated switch still needs attempted trials and a good window')
_curr.set_phase('task')
reset_ladder()
saved = M.stage; M.stage = lambda: S3
try:
    tasks, _, _, _, switch_thr, _ = M.STAGES[S3]
    if len(tasks) > 1:
        M._st['task'] = 'loc'
        for _ in range(M.WIN):             # WIN correct, attempted trials: above the threshold
            M.record_outcome(a_trial(stage=S3, context='loc'), {'choice': 'L', 'correct': True}, 0.45)
        check(M._st['task'] == 'frq',
              'switched on %d correct attempted trials (threshold %.1f)' % (M.WIN, switch_thr))
        reset_ladder(); M._st['task'] = 'loc'
        for _ in range(M.WIN):             # all wrong: below the threshold, no switch
            M.record_outcome(a_trial(stage=S3, context='loc'), {'choice': 'L', 'correct': False}, 0.0)
        check(M._st['task'] == 'loc', 'no switch while accuracy is below the threshold')
    else:
        check(True, 'arm %s has no two-task stage' % M.ARM)
finally:
    M.stage = saved
print()

print('update calls the phase clock exactly once')
_curr.set_phase('task')
calls = {'n': 0}
real_curr_update, real_base_update = _curr.update, _base.update
_curr.update = lambda perf: calls.__setitem__('n', calls['n'] + 1)
_base.update = lambda perf: calls.__setitem__('base', calls.get('base', 0) + 1)
try:
    _base.stimulus = _base.stimulus_max
    M.update(Perf(1.0, 0.2))
    check(calls['n'] == 1, '_curr.update called once per update() (got %d)' % calls['n'])
    check('base' not in calls, '_base.update never called (no second duration clock)')
finally:
    _curr.update, _base.update = real_curr_update, real_base_update
print()

print('terminate keeps tscost\'s duration-schedule guard')
_curr.set_phase('task')
M._st['pos'] = len(M.ORDER) - 1
check(M.stage() == 10, 'at the end of the ladder, stage() == 10')
_base.stimulus = 1290
check(M.terminate(Perf(1.0, 1.0)) is False,
      'False at criterion while the window is still growing (1290 ms)')
_base.stimulus = _base.stimulus_max
check(M.terminate(Perf(1.0, 1.0)) is True, 'True at criterion once the window is full')
check(M.terminate(Perf(0.5, 1.0)) is False, 'False below 0.99 decided')
M._st['pos'] = 0
if M.ORDER[0] != 10:
    _base.stimulus = _base.stimulus_max
    check(M.terminate(Perf(1.0, 1.0)) is False, 'False before stage 10, whatever the performance')
print()

# ---------------------------------------------------------------- the stage-1 hard block
# Arm `none` is ORDER = [10], a single terminal stage with no promotion threshold, so the block
# and promotion tests below do not apply to it.
LADDERED = M.ORDER[0] != 10
if not LADDERED:
    print('arm %s has a single terminal stage; skipping the promotion and hard-block checks'
          % M.ARM)

if LADDERED:
  print('the duration schedule confines the ladder to stage 1 until it completes')
  _curr.set_phase('task')
  reset_ladder()
  M._st['n_attempted'] = 200
  for t in M._st['recent']:
      M._st['recent'][t] = [1.0] * 60                 # well past every promotion threshold
  _base.stimulus = 100
  M.update(Perf(1.0, 1.0))
  check(M._st['pos'] == 0 and M.stage() == M.ORDER[0],
        'at criterion but held at stage %d while the window is 100 ms' % M.stage())
  _base.stimulus = 600
  M.update(Perf(1.0, 1.0))
  check(M._st['pos'] == 0, 'still held at 600 ms')
  _base.stimulus = _base.stimulus_max
  M.update(Perf(1.0, 1.0))
  check(M._st['pos'] == 1, 'promoted once the window reached %d ms' % _base.stimulus_max)
  check(M._st['n_attempted'] == 0 and M._st['recent']['loc'] == [],
        'promotion cleared the attempt counter and the accuracy windows')
  print()

  print('promotion needs both the trial count and the accuracy')
  _curr.set_phase('task')
  reset_ladder()
  _base.stimulus = _base.stimulus_max
  M._st['n_attempted'] = M.MIN_TRIALS - 1
  for t in M._st['recent']:
      M._st['recent'][t] = [1.0] * 60
  M.update(Perf(1.0, 1.0))
  check(M._st['pos'] == 0, 'held below MIN_TRIALS = %d attempted' % M.MIN_TRIALS)
  M._st['n_attempted'] = M.MIN_TRIALS
  thr = M.STAGES[M.ORDER[0]][3]
  for t in M._st['recent']:
      M._st['recent'][t] = [1.0] * 30 + [0.0] * 30    # 0.5, below every threshold
  M.update(Perf(1.0, 1.0))
  check(M._st['pos'] == 0, 'held at accuracy 0.5, below the stage threshold %.2f' % thr)
  for t in M._st['recent']:
      M._st['recent'][t] = [1.0] * 60
  M.update(Perf(1.0, 1.0))
  check(M._st['pos'] == 1, 'promoted with both satisfied')
  print()

print('the ladder never promotes out of stage 10, and there is no demotion')
reset_ladder(pos=len(M.ORDER) - 1)
_base.stimulus = _base.stimulus_max
M._st['n_attempted'] = 1000
for t in M._st['recent']:
    M._st['recent'][t] = [1.0] * 60
before = M._st['pos']
M.update(Perf(1.0, 1.0))
check(M._st['pos'] == before, 'stage 10 is terminal (promotion threshold is None)')
print()

# ---------------------------------------------------------------- the ladder itself
print('ORDER per arm')
check(M.ORDER == {'none': [10], 'short': [1, 2, 3, 4, 5, 6, 10],
                  'full': list(range(1, 11))}[M.ARM],
      'arm %s -> %s' % (M.ARM, M.ORDER))
print()

print('what each stage puts on the inputs')
_curr.set_phase('task')
rng = np.random.RandomState(0)
for s in sorted(M.STAGES):
    reset_ladder()
    M._st['pos'] = M.ORDER.index(s) if s in M.ORDER else 0
    M._st['validating'] = True                      # uniform task draw, no anti-bias redraw
    tasks, rel_s, irr_s, thr, switch, qab = M.STAGES[s]
    # force the stage directly, bypassing ORDER for the stages this arm skips
    saved = M.stage
    M.stage = lambda s=s: s
    try:
        got_rel, got_irr, zeroed = set(), set(), []
        for _ in range(40):
            tr = M.get_condition(rng, DT, {})
            check_task = tr['context']
            rel_k = 'strength_loc' if check_task == 'loc' else 'strength_freq'
            irr_k = 'strength_freq' if check_task == 'loc' else 'strength_loc'
            got_rel.add(round(tr[rel_k], 2)); got_irr.add(round(tr[irr_k], 2))
            other = tr['hml'] if check_task == 'loc' else tr['lmr']
            zeroed.append(bool(np.all(other == 0)))
            if tr['stage'] != s:
                break
        ok_rel = got_rel <= set(round(x, 2) for x in rel_s)
        exp_irr = set(round(x, 2) for x in irr_s) if irr_s is not None else {0.5}
        ok_irr = got_irr <= exp_irr
        if irr_s is None:
            ok_zero = all(zeroed)
            zmsg = 'irrelevant pulses absent'
        elif irr_s == [M.G[0]]:
            ok_zero = True
            zmsg = 'irrelevant present but uninformative (0.5)'
        else:
            ok_zero = True
            zmsg = 'irrelevant informative'
        check(ok_rel and ok_irr and ok_zero,
              'stage %2d: tasks %-14s relevant %-18s %s'
              % (s, tasks, sorted(got_rel), zmsg))
    finally:
        M.stage = saved
print()

print('the irrelevant feature really is zeroed in stages 1-4 and present in 5-6')
_curr.set_phase('task')
for s, want_zero in ((1, True), (3, True), (5, False), (10, False)):
    reset_ladder()
    M._st['validating'] = True
    saved = M.stage; M.stage = lambda s=s: s
    try:
        any_nonzero = False
        for _ in range(200):
            tr = M.get_condition(rng, DT, {})
            other = tr['hml'] if tr['context'] == 'loc' else tr['lmr']
            if np.any(other != 0):
                any_nonzero = True; break
        check(any_nonzero == (not want_zero),
              'stage %2d: irrelevant stream %s' % (s, 'all zero' if want_zero else 'has pulses'))
    finally:
        M.stage = saved
print()

print('pre-training is untouched by the ladder')
for ph in ('fixed', 'blocks', 'random'):
    _curr.set_phase(ph)
    reset_ladder()
    tr = M.get_condition(rng, DT, {})
    ok = (np.all(tr['lmr'] == 0) and np.all(tr['hml'] == 0)
          and tr['context'] == 'loc' and 'stage' not in tr)
    check(ok, 'phase %-7s: no pulses, LOC go cue, no stage tag' % ph)
_curr.set_phase('task')
print()

print('get_step: the quadrant scaling touches only a correct terminal choice')
tr = a_trial(dec_t0=130, quadrant=1)
tr['qab_p'] = 0.5
_base.stimulus = _base.stimulus_max
cond = M.get_condition(rng, DT, {})
cond['quadrant'] = 1; cond['qab_p'] = 0.5
side = M.actions['CHOOSE-LEFT'] if cond['correct_side'] == 'L' else M.actions['CHOOSE-RIGHT']
_, r_in, st_in = M.get_step(rng, DT, cond, 6, M.actions['FIXATE'])
check(abs(r_in + M.STEP_COST) < 1e-12 and st_in['continue'],
      'an in-window step is charged STEP_COST and is not scaled')
_, r_hit, st_hit = M.get_step(rng, DT, cond, cond['dec_t0'] + 1, side)
check(abs(r_hit - 4.0 * 0.5) < 1e-9,
      'a correct choice is scaled by 4 x qab_p: %.3f' % r_hit)
cond2 = M.get_condition(rng, DT, {})
cond2['quadrant'] = 0; cond2['qab_p'] = 0.25
side2 = M.actions['CHOOSE-LEFT'] if cond2['correct_side'] == 'L' else M.actions['CHOOSE-RIGHT']
_, r_unscaled, _ = M.get_step(rng, DT, cond2, cond2['dec_t0'] + 1, side2)
check(abs(r_unscaled - 1.0) < 1e-12, 'with quadrant 0 (stages 1-6) the reward is unscaled')
print()

print('=' * 70)
if FAIL:
    print('%d FAILURES' % len(FAIL))
    for f in FAIL:
        print('  - ' + f)
    sys.exit(1)
print('all stages4 checks passed')
