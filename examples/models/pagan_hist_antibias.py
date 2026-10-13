"""
pagan_hist_curriculum.py plus the lab's anti-bias schedule in the task phase.

Why (part six of the Gatsby talk). The rats' trainer does not draw the
correct side 50/50: BControl's anti-bias pushes the next correct side toward the side the rat
has been getting wrong. The talk plan says the anti-bias works by discouraging win-stay/
lose-switch; the discovery project measured on the rats' trials that the schedule as
implemented pays win-stay/lose-switch 3-6% above chance early in training
(rat_strategy_discovery/learning/03_results/story.md; explorer/07_antibias.html). This module
puts the same rule in front of a reward-trained network so the direction and size of its
effect on the habit can be measured on networks that have the habit (the pre-trained ones)
and on networks that do not.

The rule (psytrack_analyses/code/preprocessing/recomputeAntibias.m, pipeline values tau 50,
beta 5), per trial t:
    hit rate per side = sum over past trials of hit * exp(-lag/tau), per correct side,
                        divided by the sum of the kernel for that side
    fracs = [hitL, hitR] / (hitL + hitR), each rounded to 2 decimals
    P(right correct) = exp(-beta * fracR) / (exp(-beta * fracL) + exp(-beta * fracR))
    (0.5 until both sides have been seen)
Keyed on the CORRECT side of past trials, as the MATLAB variable `sides` is.

Validation batches are drawn with the schedule OFF (set_validation hook), so P(correct) and the
early-stopping criterion mean the same thing in every arm.

How it is applied. The stimulus is drawn exactly as in pagan.py; if the rule asks for the
other side, the relevant feature is mirrored (left<->right pulses in the LOC context,
high<->low in the FRQ context), which keeps every pulse statistic and the irrelevant feature
unchanged and only moves the correct side. The rule only acts in the task phase; the
pre-training phases set the side themselves.

The outcome of each trial reaches the task through `record_outcome(trial, status, reward)`,
called by PolicyGradient.run_trials after every trial (a hook added to our working copy of
pyrl on 11 Oct 2026; pyrl otherwise never tells the task what happened). Only training trials
are counted; validation batches are excluded (see set_validation).

Switches (environment variables):
    PAGAN_PRETRAIN=0      no pre-training (inherited)
    PAGAN_PHASE=task      force the task phase, e.g. on resume (inherited)
    PAGAN_ANTIBIAS=0      schedule off: analysis rollouts must measure every arm on the same
                          50/50 trial distribution
    PAGAN_AB_BETA=5       gain; negative inverts the schedule (pushes toward the side the
                          network gets right)
    PAGAN_AB_TAU=50       kernel time constant in trials
"""
from __future__ import division, print_function

import os
import sys

import numpy as np

from pyrl import tasktools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagan_hist_curriculum import *          # noqa: F401,F403
import pagan_hist_curriculum as _curr

AB_ON   = os.environ.get('PAGAN_ANTIBIAS', '1') != '0'
AB_BETA = float(os.environ.get('PAGAN_AB_BETA', '5'))
AB_TAU  = float(os.environ.get('PAGAN_AB_TAU', '50'))

# Exponentially weighted per-side hit sums and kernel sums, updated recursively (one decay
# per trial, as the MATLAB loop's kernel exp(-lag/tau) implies).
_ab = {'sumL': 0.0, 'wL': 0.0, 'sumR': 0.0, 'wR': 0.0, 'n': 0, 'n_flipped': 0, 'validating': False}

def set_validation(flag):
    """Called by PolicyGradient.train around every validation batch (hook added 11 Oct 2026).
    Validation trials are drawn 50/50 and not counted, so P(correct) and the early-stopping
    criterion measure the same thing in every arm. Without this the inverted schedule
    (beta < 0) hands a one-sided network a validation batch it gets 98% right and the run
    stops at update 35 having learned nothing, which is what happened on the first launch."""
    _ab['validating'] = bool(flag)
_DECAY = np.exp(-1.0 / AB_TAU)

def antibias_p_right():
    """P(right is correct) under the rule, from the history so far; 0.5 until both sides seen."""
    if _ab['wL'] <= 0 or _ab['wR'] <= 0:
        return 0.5
    hitL, hitR = _ab['sumL'] / _ab['wL'], _ab['sumR'] / _ab['wR']
    s = hitL + hitR
    if s <= 0:
        fracL = fracR = 0.5
    else:
        fracL, fracR = round(hitL / s, 2), round(hitR / s, 2)
    eL, eR = np.exp(-fracL * AB_BETA), np.exp(-fracR * AB_BETA)
    return float(eR / (eL + eR))

def record_outcome(trial, status, reward):
    """Hook called by run_trials after every trial. Count it only in the task phase and only
    if a choice was made (a trial without a decision is not a hit or a miss on a side)."""
    if not AB_ON or _ab['validating'] or _curr.phase() != 'task' or status.get('choice') is None:
        return
    hit = 1.0 if reward > 0 else 0.0
    _ab['sumL'] *= _DECAY; _ab['wL'] *= _DECAY; _ab['sumR'] *= _DECAY; _ab['wR'] *= _DECAY
    if trial['correct_side'] == 'R':
        _ab['sumR'] += hit; _ab['wR'] += 1.0
    else:
        _ab['sumL'] += hit; _ab['wL'] += 1.0
    _ab['n'] += 1

def _mirror(trial):
    """Swap the correct side by mirroring the relevant feature's pulses."""
    if trial['context'] == 'loc':
        trial['lmr'] = (-trial['lmr']).astype(trial['lmr'].dtype)
        trial['integral_loc'] = -trial['integral_loc']
    else:
        trial['hml'] = (-trial['hml']).astype(trial['hml'].dtype)
        trial['integral_freq'] = -trial['integral_freq']
    trial['integral'] = -trial['integral']
    trial['correct_side'] = 'R' if trial['correct_side'] == 'L' else 'L'
    return trial

def get_condition(rng, dt, context={}):
    trial = _curr.get_condition(rng, dt, context)
    if not AB_ON or _ab['validating'] or trial.get('phase') != 'task':
        return trial
    p_right = antibias_p_right()
    want = 'R' if rng.uniform() < p_right else 'L'
    trial['ab_p_right'] = p_right
    if trial['integral'] == 0:
        # zero evidence: the base drew the side at random; just take the rule's side
        trial['correct_side'] = want
        return trial
    if trial['correct_side'] != want:
        _mirror(trial)
        _ab['n_flipped'] += 1
    return trial

def update(perf):
    _curr.update(perf)
    if AB_ON and _curr.phase() == 'task' and _ab['n'] > 0:
        print("[ pagan_hist_antibias ] beta {} tau {}: {} trials counted, {} flipped, P(right) now {:.2f}"
              .format(AB_BETA, AB_TAU, _ab['n'], _ab['n_flipped'], antibias_p_right()))
