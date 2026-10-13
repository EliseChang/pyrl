"""
pagan_hist_curriculum.py plus an effort cost, second version: a DECLINE action of its own and
one lump cost per accepted trial.

Why. The first effort model (pagan_hist_effort.py) never
learned the task in 201,600 trials, at any cost level. Two design faults, found on the
explorer pages: (1) declining was signalled with CHOOSE-LEFT/RIGHT at the first step, so the
opt-out shared its output weights with the choice itself and every push toward declining
was a push toward one side; (2) the cost was charged per stimulus step, so every FIXATE action
during the stimulus carried a negative return and REINFORCE read it as "stop fixating". Here
declining has its own output unit and the whole cost is charged once when the trial is
accepted, so the choice policy and the effort policy are learned on separate weights and the
within-trial actions are rewarded as in the base task.

Rules:
  1. Four actions: FIXATE, CHOOSE-LEFT, CHOOSE-RIGHT, DECLINE. pyrl draws Wout last from the
     weight seed, so Win, bin and Wrec are bit-identical to the three-action arms with the
     same seed; only the readout differs.
  2. Opt in at the first step: the action taken at t = 0 (seen at t = 1). DECLINE ends the
     trial at once, reward 0, no choice (and, through pagan_hist, no history on the next
     trial). Anything else accepts the trial and is charged EFFORT right there.
  3. After that, the base task exactly: pulses, contexts, pre-training phases, previous-
     outcome channels, +1 / 0 for a correct / incorrect choice. DECLINE during the trial is a
     null action, like FIXATE; L/R during the stimulus stay null; L/R in the decision epoch
     decide.
At initialisation P(DECLINE) is about 1/4, so three trials in four are attempted from the
start. Expected value of attempting at chance is 0.5 - EFFORT, as before.

Switches: PAGAN_EFFORT (default 0.55), PAGAN_PRETRAIN=0, PAGAN_PHASE=task (inherited).
"""
from __future__ import division, print_function

import os
import sys

import numpy as np

from pyrl import tasktools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagan_hist_curriculum import *          # noqa: F401,F403
import pagan_hist_curriculum as _curr

# Redefine the action set: the model harvests `actions` from this namespace, so Nout = 4.
actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT', 'DECLINE')
DECLINE = actions['DECLINE']

EFFORT = float(os.environ.get('PAGAN_EFFORT', '0.55'))
_stats = {'attempted': 0, 'declined': 0}

def get_step(rng, dt, trial, t, a):
    if t - 1 == 0:
        if a == DECLINE:
            _stats['declined'] += 1
            return np.zeros(len(inputs)), 0.0, {'continue': False, 'declined': True, 'stim_choice': False}   # noqa: F405
        _stats['attempted'] += 1
        u, reward, status = _curr.get_step(rng, dt, trial, t, a)
        return u, reward - EFFORT, status            # the one lump cost of accepting the trial
    if a == DECLINE:
        a = actions['FIXATE']                        # inside a trial DECLINE is a null action
    return _curr.get_step(rng, dt, trial, t, a)

def update(perf):
    _curr.update(perf)
    n = _stats['attempted'] + _stats['declined']
    if n:
        print("[ pagan_hist_effort2 ] EFFORT {}: attempted {} of {} trials so far ({:.2f})"
              .format(EFFORT, _stats['attempted'], n, _stats['attempted'] / n))
