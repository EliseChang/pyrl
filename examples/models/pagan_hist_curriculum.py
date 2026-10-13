"""
pagan_hist.py plus the rats' nose-poke pre-training, as a curriculum.

Why. No reward-trained network, in any stack, starts the pulse task
with the win-stay habit the rats bring to it. The rats are not naive: before TaskSwitch
stage 1 every Princeton rat went through the ProAnti3Marino pro-only protocol, which the
extracted sessions (/mnt/big/rat_behavior/derived/pretraining, 213 rats) show to be

    one side only (left in 209/213 rats)       ~480 trials, hit 0.90
    alternating side blocks of ~20 trials      ~900 trials, ~43 switches, hit 0.73
    random side, MaxSame 4                     ~570 trials, hit 0.47 (chance)

and that they leave it with P(stay|win) 0.72-0.84 and P(stay|lose) ~0.6, i.e. a
"repeat the rewarded side" habit and no use of the cue. The hypothesis is that this
sequence, not anything in the task, installs the habit. This module gives the network the
same sequence and then the task.

How. Phases are advanced by update(), which pyrl calls once per checkpoint (iteration 0,
then every `checkfreq` iterations), so phase lengths are in checkpoints. With
checkfreq = 5 and n_gradient = 72, one checkpoint is 360 training trials:

    fixed   1 checkpoint   = 360 trials     (rats: ~480)
    blocks  3 checkpoints  = 1080 trials    (rats: ~900)
    random  2 checkpoints  = 720 trials     (rats: ~570)
    task    the rest

During the pre-training phases there are no pulses (lmr = hml = 0) and the context
channel LOC is on for the stimulus epoch exactly as in the base task, so it still serves
as the go cue; the correct side is set by the phase, not by evidence. Reward structure,
epochs, actions, history inputs and Nin = 8 are inherited from pagan_hist, so weight
seeds are aligned with the histgauss100 / iti10 libraries. There is deliberately NO light
cue in this version: the rats demonstrably do not use it (hit 0.47 in the random phase),
and adding a channel would break seed alignment. A light variant can be a separate file.

Validation batches drawn during a pre-training phase come from that phase, so the
checkpoint's P(correct) measures performance on the phase in force. Early stopping only
applies in the task phase.

Switches (environment variables, because do.py --config cannot set module globals):
    PAGAN_PRETRAIN=0      skip the pre-training: start in the task phase. This is the
                          matched control (identical seeds, identical everything else).
    PAGAN_PHASE=task      force a phase at import, for analysis rollouts: do.py 'run'
                          imports the module in a fresh process, and rollouts for the
                          history GLM must be drawn from the task distribution whatever
                          checkpoint is loaded.

Usage (from the repo root, PYTHONPATH set):
    python examples/do.py --suffix _curr_w001 --config weight_seed=1 ... \
        --config checkfreq=5 examples/models/pagan_hist_curriculum.py train
    PAGAN_PHASE=task python examples/do.py --suffix _curr_w001 \
        examples/models/pagan_hist_curriculum.py run examples/analysis/pagan_hist.py \
        trials-b 25 iters=all
"""
from __future__ import division, print_function

import os
import sys

import numpy as np

from pyrl import tasktools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagan_hist import *          # noqa: F401,F403
import pagan_hist as _hist
import pagan as _base

# A continuous-chain validation batch of 3600 trials every 5 iterations would dominate the
# run time; 720 is what pagan_iti uses and is enough for P(correct) to 2 decimals.
n_validation = 720

#-----------------------------------------------------------------------------------------
# The curriculum
#-----------------------------------------------------------------------------------------

# (name, length in checkpoints); the last phase runs to max_iter.
PHASES = [('fixed', 1), ('blocks', 3), ('random', 2), ('task', None)]

FIXED_SIDE = 'L'     # 209 of 213 rats were started on the left
BLOCK_LEN  = 20      # trials per side block in the 'blocks' phase (P127: switches every ~20)
MAX_SAME   = 4       # 'random' phase: no more than 4 consecutive trials on one side

PRETRAIN = os.environ.get('PAGAN_PRETRAIN', '1') != '0'

# Mutable phase state. `_checkpoints` counts update() calls; `_block_pos` and `_recent`
# are per-trial counters advanced by get_condition in the blocks and random phases.
# `checkpoints` starts at -1 because pyrl calls update() once at iteration 0, before any
# training, and that call must not count as a completed checkpoint.
_state = {'phase_idx': 0 if PRETRAIN else len(PHASES) - 1,
          'checkpoints': -1, 'block_pos': 0, 'recent': []}

_forced = os.environ.get('PAGAN_PHASE')
if _forced is not None:
    _state['phase_idx'] = [p[0] for p in PHASES].index(_forced)

def phase():
    return PHASES[_state['phase_idx']][0]

def set_phase(name):
    _state['phase_idx'] = [p[0] for p in PHASES].index(name)
    _state['block_pos'] = 0
    _state['recent']    = []

def _pretrain_side(rng):
    """Correct side for the current pre-training phase; mirrors SidesSection.m."""
    ph = phase()
    if ph == 'fixed':
        return FIXED_SIDE
    if ph == 'blocks':
        # Side flips every BLOCK_LEN calls. Both training and validation draws advance
        # the counter, which only shifts block boundaries, as the trainer's hand did.
        side = 'L' if (_state['block_pos'] // BLOCK_LEN) % 2 == 0 else 'R'
        _state['block_pos'] += 1
        return side
    if ph == 'random':
        recent = _state['recent']
        if len(recent) >= MAX_SAME and all(s == recent[-1] for s in recent[-MAX_SAME:]):
            side = 'R' if recent[-1] == 'L' else 'L'
        else:
            side = tasktools.choice(rng, ['L', 'R'])
        recent.append(side)
        if len(recent) > MAX_SAME:
            del recent[0]
        return side
    raise ValueError(ph)

def get_condition(rng, dt, context={}):
    """In the task phase, exactly pagan.get_condition. In a pre-training phase, the same
    trial structure with zero pulses and the correct side set by the phase. The pulse RNG
    draws are still consumed so that the stimulus stream stays aligned with the task-phase
    draws of a control run with the same seeds."""
    trial = _base.get_condition(rng, dt, context)
    ph = phase()
    if ph == 'task':
        trial['phase'] = ph
        return trial
    trial['lmr'] = np.zeros_like(trial['lmr'])
    trial['hml'] = np.zeros_like(trial['hml'])
    trial['integral_loc'] = trial['integral_freq'] = trial['integral'] = 0
    trial['context'] = 'loc'              # the LOC channel is the go cue in every phase
    trial['correct_side'] = _pretrain_side(rng)
    trial['phase'] = ph
    return trial

def terminate(perf):
    if phase() != 'task':
        return False
    return _hist.terminate(perf)

def update(perf):
    """Called once per checkpoint. Advance the phase clock; no performance gating, because
    the phase lengths are set to the rats' median trial counts (TrainingSection-style
    gating can be added once the un-gated version has been seen)."""
    _state['checkpoints'] += 1
    idx = _state['phase_idx']
    name, length = PHASES[idx]
    if length is None:
        return
    done = sum(l for _, l in PHASES[:idx + 1] if l is not None)
    if _state['checkpoints'] >= done:
        set_phase(PHASES[idx + 1][0])
        p_decision, p_correct = tasktools.correct_2AFC(perf)
        print("[ pagan_hist_curriculum ] phase {} -> {} after {} checkpoints "
              "(last validation: P(choice) {:.3f}, P(correct|choice) {:.3f})"
              .format(name, phase(), _state['checkpoints'], p_decision, p_correct))

# get_step is inherited from pagan_hist: it writes the four base channels (zero pulses here,
# LOC cue on) and the previous-outcome one-hot, which is the only thing the network can use
# in the pre-training phases.
