"""
pagan_hist_curriculum.py plus an effort cost: attending a trial costs, declining is free.

Why (model 3). The rats' learning data say that what separates
learners from non-learners is participation, not perception: every non-learner sits at the
same stimulus-weight plateau the learners pass through, and what differs is completion rate
(rat_strategy_discovery, story.md section 4). The reward-trained networks so far cannot
disengage: pyrl runs every trial to its decision epoch. This module gives the network the
choice. Nothing about engagement is installed; the network decides whether a trial is worth
its effort, and disengagement, if it appears, is a learned policy (the normative version of
the discovered engagement state; cf. Njaradi, Carrasco-Davis, Latham, Saxe 2026, effort under
the agent's control).

The mechanism, three rules on top of the base task:
  1. Opt in at the first step. The action taken at t = 0 (evaluated at t = 1) is the
     decision to work: FIXATE attempts the trial; CHOOSE-LEFT or CHOOSE-RIGHT declines it.
     A declined trial ends at once with reward 0, no choice, and (through pagan_hist) no
     history on the next trial, like a violation. At initialisation P(FIXATE) is about 0.25,
     so an untrained network attempts a quarter of its trials and learns from there.
  2. Attending costs. Every step of the stimulus epoch that the network sits through costs
     c = EFFORT / n_stimulus_steps, so a full attempt costs EFFORT in total (0.4, 0.55 or
     0.7 in the first sweep). The decision epoch costs nothing; a trial that never decides
     still gets R_TERMINAL as before.
  3. Everything else is inherited: pulses, contexts, the pre-training phases, the previous-
     outcome channels, rewards +1 / 0 for a correct / incorrect choice.

Expected value of attempting at chance is 0.5 - EFFORT. With EFFORT below 0.5 attempting
always beats declining and nothing should disengage; above 0.5 a network that is not
beating chance is better off declining, and if it declines it stops learning: the trap the
rats' non-learners are in. The pre-training phases pay about 0.9 per attempt, so the
prediction is that they raise the attempt rate and carry the network through the stretch
where the task itself pays less than the effort.

Switches (environment variables):
    PAGAN_EFFORT=0.55     total cost of attending one trial (per-step cost is this divided
                          by the number of stimulus steps, 130 at dt 10 ms)
    PAGAN_PRETRAIN=0, PAGAN_PHASE=task   inherited
"""
from __future__ import division, print_function

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagan_hist_curriculum import *          # noqa: F401,F403
import pagan_hist_curriculum as _curr
import pagan as _base

EFFORT = float(os.environ.get('PAGAN_EFFORT', '0.55'))
_LEFT, _RIGHT = actions['CHOOSE-LEFT'], actions['CHOOSE-RIGHT']   # noqa: F405 (star import)
_stats = {'attempted': 0, 'declined': 0}

def get_step(rng, dt, trial, t, a):
    # The opt-in step: the action taken at t-1 = 0 decides whether this trial is attempted.
    if t - 1 == 0 and a in (_LEFT, _RIGHT):
        _stats['declined'] += 1
        u = np.zeros(len(inputs))                                   # noqa: F405
        return u, 0.0, {'continue': False, 'declined': True, 'stim_choice': False}
    if t - 1 == 0:
        _stats['attempted'] += 1
    u, reward, status = _curr.get_step(rng, dt, trial, t, a)
    # Effort: every stimulus step attended costs its share of EFFORT.
    n_stim = len(trial['lmr'])
    if status['continue'] and trial['stim_t0'] <= t - 1 < trial['dec_t0'] and n_stim > 0:
        reward -= EFFORT / n_stim
    return u, reward, status

def update(perf):
    _curr.update(perf)
    n = _stats['attempted'] + _stats['declined']
    if n:
        print("[ pagan_hist_effort ] EFFORT {}: attempted {} of {} trials so far ({:.2f})"
              .format(EFFORT, _stats['attempted'], n, _stats['attempted'] / n))
