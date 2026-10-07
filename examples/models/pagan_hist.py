"""
pagan.py plus meta-RL-style trial-history inputs.

The joint outcome of the *previous* trial - one-hot over {L-win, L-lose, R-win, R-lose} -
is held on four extra input channels for the whole of the current trial. All four are zero
on the first trial of a batch and after any trial that ended without a decision. Win/lose
is read off the previous trial's terminal reward alone (win iff reward > 0), so an
incorrect choice (R_INCORRECT = 0) and a trial that never decided (R_TERMINAL = -1) are
distinguishable: the former sets a `-LOSE` channel, the latter sets nothing.

The history channels are on from the second timestep of the trial onwards, which is every
timestep the network actually receives input: U[t] produced at rollout step t is consumed
by the policy at step t+1, and at t = 0 the network runs from x0 with no input at all.

`prev_outcome` is put on the trial dict by PolicyGradient.run_trials, which walks the batch
sequentially; gradient updates still happen once per batch, so the history dependence lives
entirely within a batch.

Everything else - stimulus generation, trial conditions, rewards, Performance, curriculum,
early stopping - is inherited unchanged from pagan.py, so the two tasks differ only in Nin
(4 vs 8). Note that a pagan savefile/checkpoint cannot be loaded by this task, or vice
versa: Nin is baked into both.
"""
from __future__ import division

import os
import sys

import numpy as np

from pyrl import tasktools

# pagan.py lives alongside this file; every key of pyrl.configs.default is harvested from
# this module's namespace, so the star import is what supplies tmax, n_gradient,
# n_validation, actions, Performance, get_condition, terminate, update, the reward
# constants, and the rest of the spec.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagan import *          # noqa: F401,F403
import pagan as _base

# The four base channels must stay first and in this order: get_step() below writes
# _base.get_step()'s vector into the leading slots.
inputs = tasktools.to_map('LMR', 'HML', 'LOC', 'FREQ',
                          'PREV-L-WIN', 'PREV-L-LOSE', 'PREV-R-WIN', 'PREV-R-LOSE')

# (previous choice, previous trial was rewarded) -> channel
PREV_CHANNEL = {('L', True):  'PREV-L-WIN',
                ('L', False): 'PREV-L-LOSE',
                ('R', True):  'PREV-R-WIN',
                ('R', False): 'PREV-R-LOSE'}

def terminate(perf):
    # Overrides pagan.py's permanently-disabled terminate() (see that module for why it's
    # off there). Early stopping is fine here since this model isn't used by the
    # checkpoint-aligned 100-seed sweep.
    p_decision, p_correct = tasktools.correct_2AFC(perf)
    return p_decision >= 0.99 and p_correct >= 0.95

def get_step(rng, dt, trial, t, a):
    u_base, reward, status = _base.get_step(rng, dt, trial, t, a)

    u = np.zeros(len(inputs))
    u[:len(u_base)] = u_base

    prev = trial.get('prev_outcome')
    if prev is not None:
        choice, prev_reward = prev
        u[inputs[PREV_CHANNEL[(choice, prev_reward > 0)]]] = 1

    return u, reward, status
