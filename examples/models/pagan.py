"""
Pulse-based context-dependent decision-making task, based on

  Individual variability of neural computations underlying flexible decisions.
  M Pagan, V Tang, MS Aoi, JW Pillow, V Mante, D Sussillo, & CD Brody, Nature 2025.

  http://dx.doi.org/10.1038/s41586-024-08433-6

The structure follows examples/models/mante.py as closely as possible, so that the two
context-dependent integration tasks can be compared under identical network
hyperparameters. The stimulus is a numpy port of `generate_pulses` in
rnn_sussillo/train_model.py: four independent Poisson sub-streams (left-high, left-low,
right-high, right-low) with total rate `pulse_rate`, presented as left-minus-right and
high-minus-low pulse counts. Binning a Poisson process at dt is exactly a per-bin Poisson
draw, so the counts are drawn directly rather than by cumulating inter-pulse intervals.

Unlike the Mante task, the correct response is set by the *realized* pulse counts rather
than by the nominal pulse ratio, so downstream analyses must bin trials by the realized
integral rather than by the nominal stimulus strength.

Note that every key of pyrl.configs.default is harvested from this module's namespace, so
module-level names must avoid `N`, `dt`, `lr`, `tau`, `p0`, `rho`, `Win`, `bout`, etc.
"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Inputs
inputs = tasktools.to_map('LMR', 'HML', 'LOC', 'FREQ')

# Actions
actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

# Trial conditions
contexts     = ['loc', 'freq']
left_rights  = [-1, 1]
strengths    = [0.25, 0.35, 0.45]
n_conditions = len(contexts) * (len(left_rights)*len(strengths))**2

# Training
n_gradient   = n_conditions
n_validation = 50*n_conditions
max_iter     = 5000

# Validate (and write a parameter checkpoint) this often. The default of 50 is too coarse here:
# the task reaches criterion in ~100 updates, which would leave only three points on the learning
# curve and three checkpoints to analyse.
checkfreq    = 10

# Total pulse rate (Hz) across the four sub-streams
pulse_rate = 40

# Rewards
R_ABORTED   = -1
R_CORRECT   = +1
R_INCORRECT = -1

# Epoch durations. There is no separate fixation epoch: fixation is the correct action
# throughout the stimulus, and the offset of the context cue serves as the go cue.
stimulus = 1300
decision = 2000
tmax     = stimulus + decision

def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    durations = {
        'stimulus': (0, stimulus),
        'decision': (stimulus, tmax),
        'tmax':     tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    # The policy receives no input on the first step of a trial, so the first stimulus
    # index is never delivered. Pin the pulses to the indices that are actually shown, so
    # that the realized evidence cannot drift out of sync with what the network saw.
    stim_t0 = epochs['stimulus'][0] + 1
    nbins   = epochs['stimulus'][-1] - stim_t0 + 1
    dec_t0  = epochs['decision'][0]

    #-------------------------------------------------------------------------------------
    # Trial
    #-------------------------------------------------------------------------------------

    context_ = context.get('context')
    if context_ is None:
        context_ = rng.choice(contexts)

    left_right_loc = context.get('left_right_loc')
    if left_right_loc is None:
        left_right_loc = rng.choice(left_rights)

    left_right_freq = context.get('left_right_freq')
    if left_right_freq is None:
        left_right_freq = rng.choice(left_rights)

    strength_loc = context.get('strength_loc')
    if strength_loc is None:
        strength_loc = rng.choice(strengths)

    strength_freq = context.get('strength_freq')
    if strength_freq is None:
        strength_freq = rng.choice(strengths)

    # lr_ratio is P(pulse is left), hl_ratio is P(pulse is high). left_right = -1 puts the
    # majority on left/high, which is the response 'L' - i.e. high maps to the same choice
    # as left, as in the reference implementation where both contexts use the sign of the
    # accumulated evidence.
    lr_ratio = (1 - strength_loc)  if left_right_loc  < 0 else strength_loc
    hl_ratio = (1 - strength_freq) if left_right_freq < 0 else strength_freq

    #-------------------------------------------------------------------------------------
    # Pulses
    #-------------------------------------------------------------------------------------

    lam  = pulse_rate*dt/1000
    n_lh = rng.poisson(lam*lr_ratio*hl_ratio,             nbins)
    n_ll = rng.poisson(lam*lr_ratio*(1 - hl_ratio),       nbins)
    n_rh = rng.poisson(lam*(1 - lr_ratio)*hl_ratio,       nbins)
    n_rl = rng.poisson(lam*(1 - lr_ratio)*(1 - hl_ratio), nbins)

    lmr = (n_lh + n_ll - n_rh - n_rl).astype(np.int16)
    hml = (n_lh - n_ll + n_rh - n_rl).astype(np.int16)

    #-------------------------------------------------------------------------------------
    # Correct response, from the realized pulse counts
    #-------------------------------------------------------------------------------------

    integral_loc  = int(lmr.sum())
    integral_freq = int(hml.sum())

    integral = integral_loc if context_ == 'loc' else integral_freq
    if integral > 0:
        correct_side = 'L'
    elif integral < 0:
        correct_side = 'R'
    else:
        correct_side = tasktools.choice(rng, ['L', 'R'])

    return {
        'durations':       durations,
        'time':            time,
        'epochs':          epochs,
        'stim_t0':         stim_t0,
        'dec_t0':          dec_t0,
        'context':         context_,
        'left_right_loc':  left_right_loc,
        'left_right_freq': left_right_freq,
        'strength_loc':    strength_loc,
        'strength_freq':   strength_freq,
        'lr_ratio':        lr_ratio,
        'hl_ratio':        hl_ratio,
        'lmr':             lmr,
        'hml':             hml,
        'integral_loc':    integral_loc,
        'integral_freq':   integral_freq,
        'integral':        integral,
        'correct_side':    correct_side
        }

def get_step(rng, dt, trial, t, a):
    #-------------------------------------------------------------------------------------
    # Reward
    #
    # The action `a` was taken at time index t-1. Comparing indices rather than testing
    # membership of epochs['decision'] is equivalent here, and avoids scanning a list of
    # several hundred indices on every step of every trial.
    #-------------------------------------------------------------------------------------

    status = {'continue': True}
    reward = 0
    if t-1 < trial['dec_t0']:
        if a != actions['FIXATE']:
            status['continue'] = False
            reward = R_ABORTED
    elif a == actions['CHOOSE-LEFT'] or a == actions['CHOOSE-RIGHT']:
        if a == actions['CHOOSE-LEFT']:
            choice = 'L'
        else:
            choice = 'R'
        status['continue'] = False
        status['choice']   = choice
        status['t_choice'] = t-1
        status['correct']  = (choice == trial['correct_side'])
        if status['correct']:
            reward = R_CORRECT
        else:
            reward = R_INCORRECT

    #-------------------------------------------------------------------------------------
    # Inputs
    #
    # This observation is consumed by the policy at time index t. The evidence channels and
    # the context cue are on for exactly the same steps, so the offset of the context cue
    # marks the start of the decision epoch (the evidence channels are frequently zero
    # within the stimulus, and so cannot serve as a go cue by themselves).
    #-------------------------------------------------------------------------------------

    u = np.zeros(len(inputs))
    i = t - trial['stim_t0']
    if 0 <= i < len(trial['lmr']):
        u[inputs['LMR']] = trial['lmr'][i]
        u[inputs['HML']] = trial['hml'][i]
        if trial['context'] == 'loc':
            u[inputs['LOC']] = 1
        else:
            u[inputs['FREQ']] = 1

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.85
