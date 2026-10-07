"""
Analysis for the pulse-based context-dependent task in examples/models/pagan.py.

Ported from examples/analysis/mante.py. Two things differ from the Mante task and drive
every task-specific change below:

  1. The correct response is set by the *realized* pulse integral, not by a nominal
     coherence, so trials are binned by `integral_loc`/`integral_freq` rather than by
     `left_right_* x coh_*`.

  2. Sign convention. In mante, `left_right > 0` means 'R', so its evidence regressor is
     positive-for-R. In pagan a *positive* integral means 'L'. Every pagan evidence term
     is therefore negated, so that positive still means "evidence for R" and the loadings
     and trajectories come out with the same orientation as mante's.

Single-unit panels (mante's `sort`) are deliberately not ported.
"""
from __future__ import absolute_import, division

import os
import re
from   collections import OrderedDict

import numpy as np

from pyrl          import fittools, runtools, tasktools, utils
from pyrl.figtools import apply_alpha, Figure

#/////////////////////////////////////////////////////////////////////////////////////////
# Binning the realized evidence
#/////////////////////////////////////////////////////////////////////////////////////////

# |integral| boundaries separating magnitudes 1, 2 and 3. Fixed rather than quantile-based,
# so that a bin means the same thing at every checkpoint of a training run. Three magnitudes
# x two signs = six levels, matching mante's three coherences x two signs, which lets
# `plot_taskaxes` and the figure legends be reused unchanged.
INTEGRAL_EDGES = (10, 20)

def integral_bin(I):
    """
    Signed evidence level in {-3,-2,-1,+1,+2,+3}, or None when the integral is zero.

    A zero integral has no side: the correct response was drawn at random, so such trials
    are dropped from binned analyses. They are kept in the regression, which uses the
    continuous integral.

    """
    if I == 0:
        return None

    return int(np.sign(I))*(1 + sum(abs(I) >= edge for edge in INTEGRAL_EDGES))

def rightward(I):
    """
    Evidence for 'R', i.e. the integral in mante's sign convention.

    """
    return -I

#/////////////////////////////////////////////////////////////////////////////////////////
# Selectivity helpers (TDR infrastructure)
#/////////////////////////////////////////////////////////////////////////////////////////

def stimulus_idx(trial):
    """
    Indices of the stimulus epoch within `trial['time']`.

    `trial['epochs']` indexes the full-resolution time grid, so it goes stale as soon as
    `runtools.run` subsamples `trial['time']` for a `--dt-save` run. The epoch boundaries in
    `trial['durations']` are in ms and so survive subsampling; recompute from those, using
    the same half-open convention as `tasktools.get_idx`.

    """
    start, end = trial['durations']['stimulus']
    t          = trial['time']

    return np.where((start <= t) & (t < end))[0]

def compute_dprime(trials, perf, r):
    """
    Compute d' for choice.

    """
    N  = r.shape[-1]
    L  = np.zeros(N)
    L2 = np.zeros(N)
    R  = np.zeros(N)
    R2 = np.zeros(N)
    nL = 0
    nR = 0

    for n, trial in enumerate(trials):
        if perf.choices[n] is None:
            continue

        r_n = r[stimulus_idx(trial),n]

        if perf.choices[n] == 'L':
            L  += np.sum(r_n,    axis=0)
            L2 += np.sum(r_n**2, axis=0)
            nL += r_n.shape[0]
        else:
            R  += np.sum(r_n,    axis=0)
            R2 += np.sum(r_n**2, axis=0)
            nR += r_n.shape[0]

    mean_L = L/nL
    var_L  = L2/nL - mean_L**2

    mean_R = R/nR
    var_R  = R2/nR - mean_R**2

    return -utils.div(mean_L - mean_R, np.sqrt((var_L + var_R)/2))

def get_preferred_targets(trials, perf, r):
    """
    Determine preferred targets.

    """
    return 2*(compute_dprime(trials, perf, r) > 0) - 1

# A unit counts as active if it varies appreciably relative to the most variable unit in
# the population. mante uses an absolute cutoff (0.2), which is calibrated for unbounded
# relu rates and would discard almost every unit of a tanh network, whose rates are
# confined to [-1, 1]. Expressing the cutoff as a fraction of the population maximum gives
# one code path that behaves the same way for either activation. 0.13 is chosen to
# reproduce the absolute cutoff on the trained relu model, where max sd is 1.53: it selects
# 86 of 100 units where the absolute cutoff selected 85, and drops none of them.
ACTIVE_FRAC = 0.13

def get_active_units(r, M):
    N   = r.shape[-1]
    M_  = (np.tile(M.T, (N, 1, 1))).T
    r_  = r*M_
    n   = np.sum(M)
    var = (r_**2).sum(axis=0).sum(axis=0)/n - (r_.sum(axis=0).sum(axis=0)/n)**2
    sd  = np.sqrt(var)

    return np.where(sd > ACTIVE_FRAC*sd.max())[0]

def sort_func(s, preferred_targets, target, trial):
    """
    Per-unit condition labels for one trial, or None if the trial has no evidence level.

    """
    choices = preferred_targets*target

    if s == 'loc-choice':
        level = integral_bin(rightward(trial['integral_loc']))
    elif s == 'freq-choice':
        level = integral_bin(rightward(trial['integral_freq']))
    else:
        raise ValueError(s)

    if level is None:
        return None

    levels = preferred_targets*level

    return [(choice, lev, trial['context']) for choice, lev in zip(choices, levels)]

def sort_statespace(trialsfile, network='p'):
    """
    Condition-averaged, per-unit-normalized trajectories for the state-space analysis.

    Returns `(time_a, sorted_trials)`, where `time_a` is time aligned to stimulus onset and
    `sorted_trials[mode][(choice, level, context)]` has shape `(len(time_a), N)`.

    """
    trials, U, Z, Z_b, A, P, M, perf, r_p, r_v = utils.load(trialsfile)

    r = r_p if network == 'p' else r_v
    N = r.shape[-1]

    # Same for every trial: pagan has no random delay
    time  = trials[0]['time']
    Ntime = len(time)

    # Aligned time
    time_a  = np.concatenate((-time[1:][::-1], time))
    Ntime_a = len(time_a)

    preferred_targets = get_preferred_targets(trials, perf, r)

    sortby = ['loc-choice', 'freq-choice']

    #-------------------------------------------------------------------------------------
    # Sort
    #-------------------------------------------------------------------------------------

    sorted_trials = {s: {} for s in sortby}
    X  = 0
    X2 = 0
    NX = 0
    for n, trial in enumerate(trials):
        target = +1 if perf.choices[n] == 'R' else -1

        if perf.corrects[n]:
            for s in sortby:
                sorted_trial = sort_func(s, preferred_targets, target, trial)
                if sorted_trial is None:
                    continue
                for u, cond in enumerate(sorted_trial):
                    sorted_trials[s].setdefault(cond, []).append((n, u))

        # For normalizing, per unit
        Mn  = np.tile(M[:,n], (N,1)).T
        Rn  = r[:,n]*Mn
        X  += np.sum(Rn,    axis=0)
        X2 += np.sum(Rn**2, axis=0)
        NX += np.sum(Mn,    axis=0)
    mean = X/NX
    sd   = np.sqrt(X2/NX - mean**2)

    #-------------------------------------------------------------------------------------
    # Average within conditions
    #-------------------------------------------------------------------------------------

    for s in sorted_trials:
        trials_by_cond = {}
        for cond, n_u in sorted_trials[s].items():
            trials_by_cond.setdefault(cond, {'r': np.zeros((Ntime_a, N)),
                                             'n': np.zeros((Ntime_a, N))})
            for n, u in n_u:
                Mn  = M[:,n]
                Rnu = r[:,n,u]*Mn

                # Normalize, then re-apply the mask: subtracting the mean makes the
                # padded timesteps non-zero, so without this they contribute -mean/sd to
                # 'r' at bins where 'n' stays 0, and utils.div turns that into -inf.
                Rnu = (Rnu - mean[u])/sd[u]*Mn

                # Align point. Unlike mante there is no pre-stimulus epoch, so the
                # stimulus starts at index 0; mante's "one bin before onset" convention
                # would give t0 = -1 and push the whole trial into the pre-onset half.
                t0 = stimulus_idx(trials[n])[0]

                # Before
                n_b = Rnu[:t0].shape[0]
                trials_by_cond[cond]['r'][Ntime-1-n_b:Ntime-1,u] += Rnu[:t0]
                trials_by_cond[cond]['n'][Ntime-1-n_b:Ntime-1,u] += Mn[:t0]

                # After
                n_a = Rnu[t0:].shape[0]
                trials_by_cond[cond]['r'][Ntime-1:Ntime-1+n_a,u] += Rnu[t0:]
                trials_by_cond[cond]['n'][Ntime-1:Ntime-1+n_a,u] += Mn[t0:]

        for cond in trials_by_cond:
            trials_by_cond[cond] = utils.div(trials_by_cond[cond]['r'],
                                             trials_by_cond[cond]['n'])

        sorted_trials[s] = trials_by_cond

    return time_a, sorted_trials

#/////////////////////////////////////////////////////////////////////////////////////////
# State space (targeted dimensionality reduction)
#/////////////////////////////////////////////////////////////////////////////////////////

# Regression coefficients
CHOICE       = 0
LOC          = 1
FREQ         = 2
CONTEXT      = 3
CONSTANT     = 4
CHOICE_LOC   = 5
CHOICE_FREQ  = 6
CHOICE_CTX   = 7
LOC_FREQ     = 8
LOC_CTX      = 9
FREQ_CTX     = 10
nreg         = 11

def plot_taskaxes(plot, yax, p_vc, basecolor):
    abslevels = sorted(set(abs(level) for choice, level, context in p_vc))

    #-------------------------------------------------------------------------------------
    # Subtract mean
    #-------------------------------------------------------------------------------------

    p = p_vc.values()[0]
    Xchoice = np.zeros_like(p[CHOICE])
    Xloc    = np.zeros_like(p[LOC])
    Xfreq   = np.zeros_like(p[FREQ])

    for p in p_vc.values():
        Xchoice += p[CHOICE]
        Xloc    += p[LOC]
        Xfreq   += p[FREQ]
    mean_choice = Xchoice/len(p_vc)
    mean_loc    = Xloc/len(p_vc)
    mean_freq   = Xfreq/len(p_vc)

    for cond, p in p_vc.items():
        p[CHOICE] -= mean_choice
        p[LOC]    -= mean_loc
        p[FREQ]   -= mean_freq

    #-------------------------------------------------------------------------------------

    xall = []
    yall = []
    for cond, p in p_vc.items():
        idx = abslevels.index(abs(cond[1]))
        if idx == 0:
            color = apply_alpha(basecolor, 0.4)
        elif idx == 1:
            color = apply_alpha(basecolor, 0.7)
        else:
            color = apply_alpha(basecolor, 1)

        if cond[1] > 0:
            prop = dict(mfc=color, mec=color, ms=2.5, mew=0.5)
        else:
            prop = dict(mfc='w', mec=color, ms=2.5, mew=0.5)

        plot.plot(p[CHOICE],      p[yax],      '-', color=color, lw=0.75)
        plot.plot(p[CHOICE][::2], p[yax][::2], 'o', color=color, **prop)

        xall.append(p[CHOICE])
        yall.append(p[yax])

    if yax == LOC:
        plot.ylabel('Location')
    elif yax == FREQ:
        plot.ylabel('Frequency')

    return np.concatenate(xall), np.concatenate(yall)

def plot_statespace(units, t, sorted_trials, Q, plots):
    # Task axes
    M = Q.T

    # Epoch to plot
    w, = np.where((0 <= t) & (t <= 1300))

    # Down-sample
    dt   = t[1] - t[0]
    step = int(50/dt)
    w    = w[::step]

    # Colors
    color_l = 'k'
    color_f = Figure.colors('darkblue')

    xall = []
    yall = []

    plots['f1'].xlabel('Choice')

    def project(mode, context):
        return {cond: M.dot(r.T[units,:][:,w])
                for cond, r in sorted_trials[mode].items() if cond[2] == context}

    panels = [
        # panel, sort mode,      context, y axis, colour
        ('l1',   'loc-choice',   'loc',   LOC,    color_l),
        ('l2',   'loc-choice',   'loc',   FREQ,   color_l),
        ('l3',   'freq-choice',  'loc',   FREQ,   color_f),
        ('f1',   'loc-choice',   'freq',  LOC,    color_l),
        ('f2',   'freq-choice',  'freq',  LOC,    color_f),
        ('f3',   'freq-choice',  'freq',  FREQ,   color_f),
        ]

    for panel, mode, context, yax, color in panels:
        x, y = plot_taskaxes(plots[panel], yax, project(mode, context), color)
        xall.append(x)
        yall.append(y)

    plots['l1'].ylabel('Location')

    #-------------------------------------------------------------------------------------
    # Shared axes
    #-------------------------------------------------------------------------------------

    xall = np.concatenate(xall)
    yall = np.concatenate(yall)

    for plot in plots.values():
        plot.lim('x', xall)
        plot.lim('y', yall)

def statespace(trialsfile, plots=None, dt_reg=50, **kwargs):
    """
    State-space analysis.

    """
    trials_, U, Z, Z_b, A, P, M, perf, r_p, r_v = utils.load(trialsfile)

    # Use policy network for this analysis
    r = r_p

    # Time step. Same for every trial: pagan has no random delay.
    time = trials_[0]['time']
    dt   = time[1] - time[0]
    step = int(dt_reg/dt)

    #=====================================================================================
    # Setup
    #=====================================================================================

    units = get_active_units(r, M)
    print("[ pagan.statespace ] Performing regression on {} active units."
          .format(len(units)))

    preferred_targets = get_preferred_targets(trials_, perf, r)[units]

    # Stimulus period
    stimulus = stimulus_idx(trials_[0])[::step]

    trials  = []
    infos   = []
    ints_l  = []
    ints_f  = []
    for n, trial_ in enumerate(trials_):
        if perf.choices[n] is None:
            continue

        ints_l.append(abs(trial_['integral_loc']))
        ints_f.append(abs(trial_['integral_freq']))

        trial           = {}
        trial['target'] = +1 if perf.choices[n] == 'R' else -1
        trial['t']      = time[stimulus]
        trial['r']      = r[stimulus,n,:][:,units].T
        trials.append(trial)
        infos.append(trial_)
    maxint_l = max(ints_l)
    maxint_f = max(ints_f)

    #-------------------------------------------------------------------------------------
    # Normalize
    #-------------------------------------------------------------------------------------

    X  = 0
    X2 = 0
    n  = 0
    for trial in trials:
        r   = trial['r']
        X  += np.sum(r,    axis=1)
        X2 += np.sum(r**2, axis=1)
        n  += r.shape[1]
    mean = X/n
    sd   = np.sqrt(X2/n - mean**2)

    mean = np.tile(mean, (r.shape[1], 1)).T
    sd   = np.tile(sd,   (r.shape[1], 1)).T
    for trial in trials:
        trial['r'] = (trial['r'] - mean)/sd

    #-------------------------------------------------------------------------------------
    # Regress
    #-------------------------------------------------------------------------------------

    nunits, ntime = trials[0]['r'].shape
    ntrials = len(trials)

    # Coefficient matrix. The evidence terms use the continuous integral (no binning), and
    # are negated so that positive means evidence for 'R', as in mante.
    r = np.zeros((nunits, ntime, ntrials))
    F = np.zeros((nunits, nreg, ntrials))
    for i, trial in enumerate(trials):
        info = infos[i]

        # First-order terms
        r[:,:,i]       = trial['r']
        F[:,CHOICE,i]  = preferred_targets*trial['target']
        F[:,LOC,i]     = preferred_targets*rightward(info['integral_loc'])/maxint_l
        F[:,FREQ,i]    = preferred_targets*rightward(info['integral_freq'])/maxint_f
        F[:,CONTEXT,i] = +1 if info['context'] == 'loc' else -1

        # Interaction terms
        F[:,CHOICE_LOC, i] = F[:,CHOICE,i]*F[:,LOC,i]
        F[:,CHOICE_FREQ,i] = F[:,CHOICE,i]*F[:,FREQ,i]
        F[:,CHOICE_CTX, i] = F[:,CHOICE,i]*F[:,CONTEXT,i]
        F[:,LOC_FREQ,   i] = F[:,LOC,i]*F[:,FREQ,i]
        F[:,LOC_CTX,    i] = F[:,LOC,i]*F[:,CONTEXT,i]
        F[:,FREQ_CTX,   i] = F[:,FREQ,i]*F[:,CONTEXT,i]
    F[:,CONSTANT,:] = 1

    # Regression coefficients
    beta = np.zeros((nunits, ntime, nreg))
    for i in xrange(nunits):
        Ainv = np.linalg.inv(F[i].dot(F[i].T)).dot(F[i])
        for k in xrange(ntime):
            beta[i,k] = Ainv.dot(r[i,k])
            if np.any(np.isnan(beta[i,k])):
                raise RuntimeError("[ pagan.statespace ] Regression failed.")

    #-------------------------------------------------------------------------------------
    # Sort trials
    #-------------------------------------------------------------------------------------

    utils.println("[ pagan.statespace ] Sorting trials ...")
    time_a, sorted_trials = sort_statespace(trialsfile)
    print(" done!")

    #-------------------------------------------------------------------------------------
    # Task axes
    #-------------------------------------------------------------------------------------

    # Rearrange from (units, time, reg) to (reg, time, units)
    beta = np.swapaxes(beta, 0, 2)

    # Time-independent regression vectors
    beta_max = np.zeros((nreg, nunits))
    for v in xrange(nreg):
        imax        = np.argmax(np.linalg.norm(beta[v], axis=1))
        beta_max[v] = beta[v,imax]

    Bmax = beta_max[:4].T
    Q, R = np.linalg.qr(Bmax)
    Q    = Q*np.sign(np.diag(R))

    #=====================================================================================
    # Plot regression coefficients
    #=====================================================================================

    if isinstance(plots, tuple):
        figspath, name = plots

        w = 4.5
        ratio = 1.4
        fig = Figure(w=w, r=ratio)

        x0 = 0.15
        y0 = 0.1
        w  = 0.3
        h  = w/ratio
        DX = 0.17
        DY = 0.1

        fig.add('context-loc',    [x0, y0, w, h])
        fig.add('context-freq',   [fig[-1].right+DX, y0, w, h])

        fig.add('context-choice', [x0, fig[-1].top+DY, w, h])
        fig.add('freq-loc',       [fig[-1].right+DX, fig[-1].y, w, h])

        fig.add('loc-choice',     [x0, fig[-1].top+DY, w, h])
        fig.add('freq-choice',    [fig[-1].right+DX, fig[-1].y, w, h])

        #---------------------------------------------------------------------------------

        regaxes = {'choice': CHOICE, 'loc': LOC, 'freq': FREQ, 'context': CONTEXT}
        xall = []
        yall = []
        for k, plot in fig.plots.items():
            Y, X = k.split('-')

            plot.equal()
            plot.xlabel(X.capitalize())
            plot.ylabel(Y.capitalize())

            x = Q[:,regaxes[X]]
            y = Q[:,regaxes[Y]]
            plot.plot(x, y, 'o', mfc='0.2', mec='w', ms=2.5, mew=0.3, zorder=10)
            xall += [x, -x]
            yall += [y, -y]

        for k, plot in fig.plots.items():
            plot.lim('x', xall)
            plot.lim('y', yall)

            plot.hline(0, lw=0.5, color='k', zorder=1)
            plot.vline(0, lw=0.5, color='k', zorder=1)

        fig.save(path=figspath, name='regress_coeffs')
        fig.close()

    #=====================================================================================

    if isinstance(plots, dict):
        plot_statespace(units, time_a, sorted_trials, Q, plots)
        return

    figspath, name = plots

    w = utils.mm_to_inch(174)
    ratio = 0.7
    fig = Figure(w=w, r=ratio, thickness=0.8, axislabelsize=9, ticklabelsize=7.5,
                 labelpadx=4.5, labelpady=4.5)

    w  = 0.24
    h  = 0.35
    x0 = 0.1
    y0 = 0.1
    dx = 0.07
    dy = 0.15

    fig.add('f1', [x0, y0, w, h])
    fig.add('f2', [fig[-1].right+dx, fig[-1].y, w, h])
    fig.add('f3', [fig[-1].right+dx, fig[-1].y, w, h])
    fig.add('l1', [fig['f1'].x, fig['f1'].top+dy, w, h])
    fig.add('l2', [fig[-1].right+dx, fig[-1].y, w, h])
    fig.add('l3', [fig[-1].right+dx, fig[-1].y, w, h])

    #-------------------------------------------------------------------------------------

    plot_statespace(units, time_a, sorted_trials, Q, fig.plots)

    #-------------------------------------------------------------------------------------
    # Legends
    #-------------------------------------------------------------------------------------

    fig['l2'].text_upper_center('Location context', dy=0.03, fontsize=8.5, color='k')
    fig['f2'].text_upper_center('Frequency context', dy=0.03, fontsize=8.5,
                                color=Figure.colors('darkblue'))

    ms_filled = 3
    ms_empty  = 3

    mew_filled = 0.5
    mew_empty  = 0.5

    y  = 1.13
    dx = 0.08
    dy = 0.06

    fontsize = 6

    for context, plot, basecolor in zip(['Location', 'Frequency'],
                                        [fig['f1'], fig['f3']],
                                        ['k', Figure.colors('darkblue')]):
        transform = plot.ax.transAxes
        colors    = [apply_alpha(basecolor, alpha) for alpha in [0.4, 0.7, 1]]
        for i in xrange(3):
            plot.plot(0.5+(i+0.5)*dx, y, 'o', mfc=colors[i], mec=colors[i],
                      ms=ms_filled, mew=mew_filled, transform=transform)
            plot.plot(0.5-(i+0.5)*dx, y, 'o', mfc='none', mec=colors[i],
                      ms=ms_empty, mew=mew_empty, transform=transform)

        # Strength label
        plot.text(0.5, y+dy, 'Weak', ha='center', va='bottom', fontsize=fontsize,
                  color=colors[0], transform=transform)
        plot.text(0.5+2.5*dx, y+dy, 'Strong', ha='center', va='bottom',
                  fontsize=fontsize, color=colors[-1], transform=transform)
        plot.text(0.5-2.5*dx, y+dy, 'Strong', ha='center', va='bottom',
                  fontsize=fontsize, color=colors[-1], transform=transform)

        if context == 'Location':
            plot.text(0.5-5*dx, y, context, ha='right', va='center',
                      fontsize=1.2*fontsize, color=colors[-1], transform=transform)
        else:
            plot.text(0.5+5*dx, y, context, ha='left', va='center',
                      fontsize=1.2*fontsize, color=colors[-1], transform=transform)

        # Choice label
        plot.text(0.5+2.5*dx, y-dy, 'To choice 1', ha='center', va='top',
                  fontsize=fontsize, color='k', transform=transform)
        plot.text(0.5-2.5*dx, y-dy, 'To choice 2', ha='center', va='top',
                  fontsize=fontsize, color='k', transform=transform)

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////
# Behaviour
#/////////////////////////////////////////////////////////////////////////////////////////

def _psychometric_results(trials, perf):
    """
    Core of psychometric(), operating on in-memory (trials, perf) rather than a trialsfile.

    Trials are binned by `integral_bin`, and each bin's x is the mean rightward evidence it
    actually contains, which gives `fittools.fit_psychometric` a meaningful x-axis. Binning
    by the nominal `strength` would be wrong: a sizeable fraction of strong-nominal trials
    realize an integral on the opposite side.

    """
    # results[k][level] = ([rightward evidence], [choices]), where k is
    # <cued context><evidence stream>.
    results = {cond: {} for cond in ['ll', 'lf', 'fl', 'ff']}
    for n, trial in enumerate(trials):
        if not perf.decisions[n]:
            continue

        choice = 1 if perf.choices[n] == 'R' else 0
        prefix = 'l' if trial['context'] == 'loc' else 'f'

        for stream, integral in [('l', rightward(trial['integral_loc'])),
                                 ('f', rightward(trial['integral_freq']))]:
            level = integral_bin(integral)
            if level is None:
                continue
            evidence, choices = results[prefix+stream].setdefault(level, ([], []))
            evidence.append(integral)
            choices.append(choice)

    # Convert to (mean evidence, P(right))
    for k, by_level in results.items():
        levels = sorted(by_level.keys())
        x  = np.array([np.mean(by_level[level][0]) for level in levels])
        p1 = np.array([np.mean(by_level[level][1]) for level in levels])
        results[k] = (x, p1)

    return results

def psychometric(trialsfile, plots=None, **kwargs):
    """
    Psychometric functions against the realized evidence.

    """
    trials, A, R, M, perf = utils.load(trialsfile)
    results = _psychometric_results(trials, perf)

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if plots is None:
        return results

    lw      = kwargs.get('lw', 1.25)
    ms      = kwargs.get('ms', 5)
    color_l = 'k'
    color_f = Figure.colors('darkblue')

    x_all = {'l': [], 'f': []}
    for k, (x, p1) in results.items():
        # Cued context
        if k[0] == 'l':
            color = color_l
            label = 'Location context'
        else:
            color = color_f
            label = 'Frequency context'

        plot = plots[k[1]]

        plot.plot(x, 100*p1, 'o', ms=ms, mew=0, mfc=color, zorder=10)
        props = dict(color=color, lw=lw, zorder=5, label=label)
        try:
            popt, func = fittools.fit_psychometric(x, p1)

            fit_x  = np.linspace(min(x), max(x), 201)
            fit_p1 = func(fit_x, **popt)
            plot.plot(fit_x, 100*fit_p1, **props)
        except RuntimeError:
            print("Unable to fit, drawing a line through the points.")
            plot.plot(x, 100*p1, **props)
        x_all[k[1]].append(x)

    for s in ['l', 'f']:
        plots[s].lim('x', x_all[s])
        plots[s].ylim(0, 100)
        plots[s].yticks([0, 50, 100])

    return results

#/////////////////////////////////////////////////////////////////////////////////////////
# Per-checkpoint sweep (in-memory rollouts, no trials_behavior.pkl written)
#/////////////////////////////////////////////////////////////////////////////////////////

LEVELS = [-3, -2, -1, 1, 2, 3]   # ascending: index 0 = strongly-left/low, index 5 = strongly-right/high

def trial_evidence(trials):
    """
    Per-trial realized evidence bin indices, split by context. Trials/evidence realizations
    are checkpoint- and (given fixed stim_seed) seed-invariant, so this is computed once per
    seed (on the first checkpoint's trial build) and cached/reused for every subsequent
    checkpoint, and stored once (not per-checkpoint) in the results file.

    Returns {'loc': {...}, 'freq': {...}}, each a dict of arrays:
      'trial_idx' -- index into `trials`/`perf` arrays
      'x_loc', 'x_freq' -- realized (signed, rightward-convention) evidence value for that
        trial's loc/freq stream
      'loc_idx', 'freq_idx' -- 0-5 bin index (via LEVELS) for that trial's loc/freq evidence

    Trials where either stream's integral_bin is None (zero evidence) are dropped, matching
    _psychometric_results' existing behavior.

    """
    evidence = {'loc':  {'trial_idx': [], 'x_loc': [], 'x_freq': [], 'loc_idx': [], 'freq_idx': []},
                'freq': {'trial_idx': [], 'x_loc': [], 'x_freq': [], 'loc_idx': [], 'freq_idx': []}}
    for n, trial in enumerate(trials):
        x_loc, x_freq = rightward(trial['integral_loc']), rightward(trial['integral_freq'])
        loc_level, freq_level = integral_bin(x_loc), integral_bin(x_freq)
        if loc_level is None or freq_level is None:
            continue

        ctx = trial['context']   # 'loc' or 'freq'
        evidence[ctx]['trial_idx'].append(n)
        evidence[ctx]['x_loc'].append(x_loc)
        evidence[ctx]['x_freq'].append(x_freq)
        evidence[ctx]['loc_idx'].append(LEVELS.index(loc_level))
        evidence[ctx]['freq_idx'].append(LEVELS.index(freq_level))

    for ctx in evidence:
        evidence[ctx]['trial_idx'] = np.array(evidence[ctx]['trial_idx'], dtype=int)
        evidence[ctx]['x_loc']     = np.array(evidence[ctx]['x_loc'], dtype=float)
        evidence[ctx]['x_freq']    = np.array(evidence[ctx]['x_freq'], dtype=float)
        evidence[ctx]['loc_idx']   = np.array(evidence[ctx]['loc_idx'], dtype=int)
        evidence[ctx]['freq_idx']  = np.array(evidence[ctx]['freq_idx'], dtype=int)
    return evidence

def context_choice_matrix(perf, evidence_ctx):
    """
    Build the 6x6 joint (p_right, n) matrix for one context from cached evidence indices
    (from trial_evidence) plus the current checkpoint's perf.choices/perf.decisions. Rows =
    loc_idx (0=strongly-left .. 5=strongly-right), columns = freq_idx (0=strongly-low ..
    5=strongly-high). Only decided trials count (matches _psychometric_results); cells with
    n==0 have p_right = nan.

    """
    n         = np.zeros((6, 6), dtype=int)
    right_sum = np.zeros((6, 6))
    for trial_n, i, j in zip(evidence_ctx['trial_idx'], evidence_ctx['loc_idx'],
                              evidence_ctx['freq_idx']):
        if not perf.decisions[trial_n]:
            continue
        n[i, j] += 1
        if perf.choices[trial_n] == 'R':
            right_sum[i, j] += 1

    p_right = np.full((6, 6), np.nan)
    mask = n > 0
    p_right[mask] = right_sum[mask]/n[mask]
    return {'p_right': p_right, 'n': n}

def checkpoint_psychometric_metrics(trials, perf):
    """Returns (fits, derived) for one in-memory rollout, via marginal per-stream binning."""
    points = _psychometric_results(trials, perf)
    fits = {}
    for k, (x, p1) in points.items():
        if len(x) < 3:
            continue
        try:
            popt, _ = fittools.fit_psychometric(x, p1)
        except (RuntimeError, TypeError, ValueError):
            continue
        fits[k] = popt

    nan = {'slope_rel': np.nan, 'slope_irr': np.nan, 'selectivity': np.nan,
           'bias_rel': np.nan}
    if not all(k in fits for k in ['ll', 'lf', 'fl', 'ff']):
        return fits, nan

    slope = lambda k: 1/abs(fits[k]['sigma'])
    rel = np.mean([slope('ll'), slope('ff')])
    irr = np.mean([slope('lf'), slope('fl')])
    derived = {'slope_rel': rel, 'slope_irr': irr,
               'selectivity': (rel - irr)/(rel + irr) if rel + irr > 0 else np.nan,
               'bias_rel': np.mean([fits['ll']['mu'], fits['ff']['mu']])}
    return fits, derived

def checkpoint_record(iter_, trials, perf, R, M, evidence, history_record):
    """Assemble one checkpoint's lightweight metrics record."""
    fits, derived = checkpoint_psychometric_metrics(trials, perf)
    loc_context  = context_choice_matrix(perf, evidence['loc'])
    freq_context = context_choice_matrix(perf, evidence['freq'])

    n = len(trials)
    p_decision = perf.n_decision/perf.n_trials if perf.n_trials > 0 else np.nan
    p_correct  = perf.n_correct/perf.n_decision if perf.n_decision > 0 else np.nan
    mean_reward_rollout = float(np.sum(R*M)/n)

    if history_record is not None:
        h = history_record['perf']
        history = {'n_correct': h.n_correct, 'n_decision': h.n_decision,
                   'n_trials': h.n_trials,   # validation-batch size; NOT
                                              # history_record['n_trials'] (= trials_tot)
                   'mean_reward': history_record['mean_reward']}
    else:
        print("[ checkpoint_record ] WARNING: no training_history record for iter {}"
              .format(iter_))
        history = {'n_correct': np.nan, 'n_decision': np.nan, 'n_trials': np.nan,
                   'mean_reward': np.nan}

    return {'iter': iter_, 'history': history,
            'rollout': {'p_correct': p_correct, 'p_decision': p_decision,
                       'mean_reward': mean_reward_rollout},
            'loc_context': loc_context, 'freq_context': freq_context,
            'psychometric_fits': {k: dict(v) for k, v in fits.items()},
            'derived': derived}

def checkpointsweep_metrics(pg, spec, trials_per_condition, checkpoints, history_by_iter,
                            resultsfile=None):
    """
    Roll out build_trials(...) in memory at every checkpoint, restoring pg.rng/
    pg.rng_action to the same state before each (so trial/action draws are identical
    across checkpoints). Per-trial evidence bin indices (trial_evidence) are
    checkpoint-invariant, so computed once (from the first checkpoint processed) and cached
    under saved['evidence']; only saved['checkpoints'] grows thereafter. Writes to
    resultsfile after every checkpoint (small file, cheap to rewrite in full) so an
    interrupted run is resumable: skips iters already present.

    """
    saved = {'evidence': None, 'checkpoints': []}
    if resultsfile is not None and os.path.isfile(resultsfile):
        saved = utils.load(resultsfile)
        print("[ checkpointsweep ] {} checkpoints already done in {}"
              .format(len(saved['checkpoints']), resultsfile))
    done_iters = set(r['iter'] for r in saved['checkpoints'])

    rng_state        = pg.rng.get_state()
    rng_action_state = pg.rng_action.get_state()
    for iter_ in [it for it in checkpoints if it not in done_iters]:
        pg.rng.set_state(rng_state)
        pg.rng_action.set_state(rng_action_state)
        pg.load_checkpoint(iter_)

        trials = build_trials(pg, spec, trials_per_condition)
        (U, Q, Q_b, Z, Z_b, A, R, M, init, init_b, x0, x0_b,
         perf) = pg.run_trials(trials, progress_bar=False)

        if saved['evidence'] is None:
            saved['evidence'] = trial_evidence(trials)

        record = checkpoint_record(iter_, trials, perf, R, M, saved['evidence'],
                                    history_by_iter.get(iter_))
        saved['checkpoints'].append(record)
        saved['checkpoints'].sort(key=lambda r: r['iter'])
        if resultsfile is not None:
            utils.save(resultsfile, saved)

        print("[ checkpointsweep ] iter {:6d}  p_correct={:.3f}  p_decision={:.3f}  "
              "selectivity={:.3f}".format(iter_, record['rollout']['p_correct'],
                                          record['rollout']['p_decision'],
                                          record['derived']['selectivity']))
    return saved

def performance(savefile, plot, **kwargs):
    """
    Learning curve from the training history.

    """
    training_history = utils.load(savefile)['training_history']

    all_trials    = []
    all_corrects  = []
    best_trials   = []
    best_corrects = []
    for record in training_history:
        ntrials = record['n_trials']
        perf    = record['perf']
        if perf.n_decision > 0:
            p_correct = perf.n_correct/perf.n_decision
            all_trials.append(ntrials)
            all_corrects.append(p_correct)
            if record['new_best']:
                best_trials.append(ntrials)
                best_corrects.append(p_correct)

    all_trials    = np.asarray(all_trials)
    all_corrects  = np.asarray(all_corrects)
    best_trials   = np.asarray(best_trials)
    best_corrects = np.asarray(best_corrects)

    M = 100

    plot.plot(all_trials/M, 100*all_corrects, color='0.8', lw=1.5)
    plot.plot(all_trials/M, 100*all_corrects, 'o', mfc='0.8', mew=0)

    plot.plot(best_trials/M, 100*best_corrects, color='k', lw=1.5)
    plot.plot(best_trials/M, 100*best_corrects, 'o', mfc='k', mew=0)

    plot.xlim(0, max(all_trials/M))
    plot.ylim(0, 100)

    plot.xlabel(r'Number of trials ($\times$ 10$^2$)')
    plot.ylabel('Percent correct')

#/////////////////////////////////////////////////////////////////////////////////////////
# Single trial
#/////////////////////////////////////////////////////////////////////////////////////////

def plot_trial(pg, m, rng, figspath, name, trial=None):
    """
    Observables, policy, actions and rewards for a single trial.

    """
    if trial is None:
        trial = pg.task.get_condition(rng, pg.dt)

    (U, Q, Q_b, Z, Z_b, A, R, M, init, init_b, x0, x0_b,
     perf) = pg.run_trials([trial])

    U = U[:,0,:]
    Z = Z[:,0,:]
    A = A[:,0,:]
    R = R[:,0]
    M = M[:,0]
    b = Z_b[:,0]
    t = int(np.sum(M))

    w  = 0.65
    h  = 0.18
    x  = 0.17
    dy = h + 0.05
    y0 = 0.08
    y1 = y0 + dy
    y2 = y1 + dy
    y3 = y2 + dy

    fig   = Figure(h=6)
    plots = {'observables': fig.add('observables', [x, y3, w, h]),
             'policy':      fig.add('policy',      [x, y2, w, h]),
             'actions':     fig.add('actions',     [x, y1, w, h]),
             'rewards':     fig.add('rewards',     [x, y0, w, h])}

    time        = trial['time']
    dt          = time[1] - time[0]
    act_time    = time[:t]
    obs_time    = time[:t] + dt
    reward_time = act_time + dt
    xlim        = (0, max(time))

    #-------------------------------------------------------------------------------------
    # Observables
    #-------------------------------------------------------------------------------------

    plot = plots['observables']
    channels = [('LMR',  'orange'),
                ('HML',  'purple'),
                ('LOC',  'blue'),
                ('FREQ', 'green')]
    for label, color in channels:
        u = U[:t,m.inputs[label]]
        plot.plot(obs_time, u, 'o', ms=5, mew=0, mfc=Figure.colors(color))
        plot.plot(obs_time, u, lw=1.25, color=Figure.colors(color), label=label)

    plot.xlim(*xlim)
    # The pulse channels are signed integer counts, so autoscale symmetrically rather than
    # clamping to [0, 1] as for the binary observables of the other tasks.
    umax = max(1, np.max(np.abs(U[:t])))
    plot.ylim(-umax, umax)
    plot.ylabel('Observables')

    color = Figure.colors('orange' if trial['correct_side'] == 'L' else 'purple')
    plot.text_upper_right('{} | I_loc={:+d} I_freq={:+d} $\\to$ {}'
                          .format(trial['context'], trial['integral_loc'],
                                  trial['integral_freq'], trial['correct_side']),
                          color=color)

    props = {'prop': {'size': 7}, 'handlelength': 1.2,
             'handletextpad': 1.2, 'labelspacing': 0.8}
    plot.legend(bbox_to_anchor=(1.2, 0.8), **props)

    # The offset of the context cue is the go cue, so highlighting the stimulus epoch marks
    # the period during which the network must hold fixation.
    plot.highlight(0, m.stimulus)

    #-------------------------------------------------------------------------------------
    # Policy
    #-------------------------------------------------------------------------------------

    plot = plots['policy']
    for label, color in [('FIXATE', 'blue'), ('CHOOSE-LEFT', 'orange'),
                         ('CHOOSE-RIGHT', 'purple')]:
        z = Z[:t,m.actions[label]]
        plot.plot(act_time, z, 'o', ms=5, mew=0, mfc=Figure.colors(color))
        plot.plot(act_time, z, lw=1.25, color=Figure.colors(color), label=label)

    plot.xlim(*xlim)
    plot.ylim(0, 1)
    plot.ylabel('Action probabilities')
    plot.legend(bbox_to_anchor=(1.27, 0.8), **props)

    plot.highlight(0, m.stimulus)

    #-------------------------------------------------------------------------------------
    # Actions
    #-------------------------------------------------------------------------------------

    plot = plots['actions']
    actions = [np.argmax(a) for a in A[:t]]
    plot.plot(act_time, actions, 'o', ms=5, mew=0, mfc=Figure.colors('red'))
    plot.plot(act_time, actions, lw=1.25, color=Figure.colors('red'))
    plot.xlim(*xlim)
    yticklabels = ['Fixate', 'Choose LEFT', 'Choose RIGHT']
    plot.yticklabels(yticklabels)
    plot.ylim(0, len(yticklabels)-1)
    plot.yticks(range(len(yticklabels)))
    plot.ylabel('Action')

    plot.highlight(0, m.stimulus)

    #-------------------------------------------------------------------------------------
    # Rewards
    #-------------------------------------------------------------------------------------

    plot = plots['rewards']
    plot.plot(reward_time, R[:t], 'o', ms=5, mew=0, mfc=Figure.colors('red'))
    plot.plot(reward_time, R[:t], lw=1.25, color=Figure.colors('red'))

    # Value-network prediction
    plot.plot(reward_time, b[:t], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
    plot.plot(reward_time, b[:t], lw=1.25, color=Figure.colors('orange'))

    plot.xlim(*xlim)
    plot.ylim(m.R_ABORTED, m.R_CORRECT)
    plot.xlabel('Time (ms)')
    plot.ylabel('Reward')

    plot.highlight(0, m.stimulus)

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////

def build_trials(pg, spec, trials_per_condition):
    """
    One trial per (context, left_right_loc, left_right_freq, strength_loc, strength_freq),
    repeated `trials_per_condition` times.

    """
    contexts     = spec.contexts
    left_rights  = spec.left_rights
    strengths    = spec.strengths
    n_trials     = spec.n_conditions * trials_per_condition

    task   = pg.task
    trials = []
    for n in xrange(n_trials):
        k = tasktools.unravel_index(n, (len(contexts),
                                        len(left_rights), len(left_rights),
                                        len(strengths), len(strengths)))
        condition = {
            'context':         contexts[k.pop(0)],
            'left_right_loc':  left_rights[k.pop(0)],
            'left_right_freq': left_rights[k.pop(0)],
            'strength_loc':    strengths[k.pop(0)],
            'strength_freq':   strengths[k.pop(0)]
            }
        trials.append(task.get_condition(pg.rng, pg.dt, condition))

    return trials

def checkpoint_targets(config, args):
    """
    Resolve an optional `iters=` argument into `(iter_, trialspath, figspath)` triples.

    `iters` is 'all' or a comma-separated list of iterations; omitting it targets the best
    parameters and the top-level directories, as mante does. Per-checkpoint figures go to
    `<figspath>/checkpoints/iter_NNNNNN/`, mirroring the layout of the rollouts.

    """
    kwargs    = dict(kv.split('=', 1) for kv in args if '=' in kv)
    iters_arg = kwargs.get('iters')
    if iters_arg is None:
        return [(None, config['trialspath'], config['figspath'])]

    ckptpath = os.path.join(config['trialspath'], 'checkpoints')
    if iters_arg == 'all':
        iters = sorted(int(d[5:]) for d in os.listdir(ckptpath) if d.startswith('iter_'))
    else:
        iters = [int(i) for i in iters_arg.split(',')]

    targets = []
    for iter_ in iters:
        subdir   = 'iter_{:06d}'.format(iter_)
        figspath = os.path.join(config['figspath'], 'checkpoints', subdir)
        utils.mkdir_p(figspath)
        targets.append((iter_, os.path.join(ckptpath, subdir), figspath))

    return targets

#/////////////////////////////////////////////////////////////////////////////////////////
# Architecture sweep
#/////////////////////////////////////////////////////////////////////////////////////////

# Run directories are named pagan_<f_hidden>_rho<NN>_<noise|nonoise>_s<seed> by
# examples/sweep_pagan.py. Parsing the name rather than importing the driver keeps the
# analysis usable on any subset of runs that happens to be on disk.
SWEEP_RE = re.compile(r'^pagan_(?P<f_hidden>[a-z]+)_rho(?P<rho>\d+)'
                      r'_(?P<noise>noise|nonoise)_s(?P<seed>\d+)$')

def sweep_runs(datapath):
    """
    Discover completed sweep runs under `datapath`, newest-style names only.

    Returns a list of (f_hidden, rho, noise, seed, savefile), sorted.

    """
    runs = []
    for name in sorted(os.listdir(datapath)):
        m = SWEEP_RE.match(name)
        if m is None:
            continue

        savefile = os.path.join(datapath, name, name + '.pkl')
        if not os.path.isfile(savefile):
            continue

        runs.append((m.group('f_hidden'), int(m.group('rho'))/10,
                     m.group('noise'), int(m.group('seed')), savefile))

    return runs

def sweep_curve(savefile):
    """
    (iters, p_correct, converged, final_p_decision) for one run.

    """
    save    = utils.load(savefile)
    history = save['training_history']

    iters     = []
    p_correct = []
    p_dec     = []
    for record in history:
        perf = record['perf']
        if perf.n_decision > 0:
            iters.append(record['iter'])
            p_correct.append(perf.n_correct/perf.n_decision)
            p_dec.append(perf.n_decision/perf.n_trials)

    # `terminate` in the task module is what stopped training early; a run that used up
    # its whole budget is a failure to reach criterion.
    converged = save['config']['max_iter'] not in iters

    return (np.asarray(iters), np.asarray(p_correct), converged,
            p_dec[-1] if p_dec else 0.)

def sweep(datapath, figspath):
    """
    Compare learning across the architecture grid.

    """
    runs = sweep_runs(datapath)
    if not runs:
        raise ValueError('no sweep runs found in {}'.format(datapath))

    rhos   = sorted(set(r[1] for r in runs))
    colors = {rho: c for rho, c in zip(rhos, ['0.6', 'k', '#d62728', '#1f77b4'])}

    activations = sorted(set(r[0] for r in runs))
    noises      = [n for n in ['noise', 'nonoise'] if any(r[2] == n for r in runs)]

    #=====================================================================================
    # Learning curves, one panel per (activation, noise condition)
    #=====================================================================================

    w, h, dx, dy = 0.36, 0.33, 0.12, 0.14
    x0, y0       = 0.1, 0.58

    fig = Figure(w=6.5, h=5)
    for i, f_hidden in enumerate(activations):
        for j, noise in enumerate(noises):
            name = '{}-{}'.format(f_hidden, noise)
            fig.add(name, [x0 + j*(w + dx), y0 - i*(h + dy), w, h])

    summary = []
    for f_hidden, rho, noise, seed, savefile in runs:
        name = '{}-{}'.format(f_hidden, noise)
        if name not in fig.plots:
            continue

        iters, p_correct, converged, p_dec = sweep_curve(savefile)
        summary.append((f_hidden, rho, noise, seed, iters, p_correct, converged, p_dec))

        if len(iters) == 0:
            continue

        fig[name].plot(iters, 100*p_correct, color=colors[rho], lw=1.25)
        fig[name].plot(iters, 100*p_correct, 'o', mfc=colors[rho], mew=0, ms=3)

    for i, f_hidden in enumerate(activations):
        for j, noise in enumerate(noises):
            plot = fig['{}-{}'.format(f_hidden, noise)]
            plot.hline(85, color='0.8', lw=0.75, zorder=1)
            plot.ylim(0, 100)
            plot.yticks([0, 50, 100])
            plot.text_upper_left('{}, {}'.format(
                f_hidden, 'noise' if noise == 'noise' else 'no noise'))
            if i == len(activations) - 1:
                plot.xlabel('Policy gradient updates')
            if j == 0:
                plot.ylabel('Percent correct')

    # One legend is enough for the whole grid.
    first = fig['{}-{}'.format(activations[0], noises[0])]
    for k, rho in enumerate(rhos):
        first.text_upper_right(r'$\rho$ = {}'.format(rho), dy=-0.09*k,
                               color=colors[rho])

    fig.save(path=figspath, name='learning')
    fig.close()

    #=====================================================================================
    # Text summary, averaged over seeds
    #=====================================================================================

    cells = OrderedDict()
    for f_hidden, rho, noise, seed, iters, p_correct, converged, p_dec in summary:
        cells.setdefault((f_hidden, rho, noise), []).append(
            (seed, iters, p_correct, converged, p_dec))

    lines = ['{:6s} {:5s} {:8s} {:>5s} {:>9s} {:>12s} {:>12s}'.format(
        'act', 'rho', 'noise', 'seeds', 'converged', 'iter@crit', 'P(correct)')]
    for (f_hidden, rho, noise), entries in cells.items():
        n_conv    = sum(e[3] for e in entries)
        last_corr = [e[2][-1] for e in entries if len(e[2])]
        crit      = [e[1][-1] for e in entries if e[3]]

        lines.append('{:6s} {:<5.1f} {:8s} {:>5d} {:>9s} {:>12s} {:>12s}'.format(
            f_hidden, rho, noise, len(entries),
            '{}/{}'.format(n_conv, len(entries)),
            '{:.0f}'.format(np.mean(crit)) if crit else '-',
            '{:.3f}+-{:.3f}'.format(np.mean(last_corr), np.std(last_corr))
            if last_corr else '-'))

    text = '\n'.join(lines) + '\n'
    print(text)

    utils.mkdir_p(figspath)
    with open(os.path.join(figspath, 'summary.txt'), 'w') as f:
        f.write(text)

#/////////////////////////////////////////////////////////////////////////////////////////
# Seed variance decomposition
#/////////////////////////////////////////////////////////////////////////////////////////

# Run directories named by examples/sweep_seeds.py: weight seed x training seed.
SEEDVAR_RE = re.compile(r'^pagan_seedvar_w(?P<w>\d+)_t(?P<t>\d+)$')

# Metrics reported for every run. The first four also get figure panels.
SEEDVAR_METRICS = ['p_correct', 'p_decision', 'iter_crit', 'mean_reward',
                   'slope_rel', 'slope_irr', 'selectivity', 'bias_rel']
SEEDVAR_PANELS  = ['p_correct', 'iter_crit', 'slope_rel', 'selectivity']

def seedvar_runs(datapath, trialsroot):
    """
    Discover seed-sweep runs. Returns a list of (w, t, savefile, trialsfile), sorted.

    """
    runs = []
    for name in sorted(os.listdir(datapath)):
        m = SEEDVAR_RE.match(name)
        if m is None:
            continue

        savefile = os.path.join(datapath, name, name + '.pkl')
        if not os.path.isfile(savefile):
            continue

        runs.append((int(m.group('w')), int(m.group('t')), savefile,
                     runtools.behaviorfile(os.path.join(trialsroot, name))))

    return sorted(runs)

def seedvar_savefile_metrics(savefile):
    """
    Scalar outcomes read straight out of one run's savefile.

    """
    save    = utils.load(savefile)
    history = save['training_history']

    records = [r for r in history if r['perf'].n_decision > 0]
    if not records:
        return {'p_correct': np.nan, 'p_decision': np.nan, 'iter_crit': np.nan,
                'mean_reward': np.nan, 'converged': False}

    # `terminate` in the task module is what stops training early, so a run that used up
    # its whole budget never reached criterion. Its time-to-criterion is censored at the
    # cap rather than dropped (non-convergence is part of the variability being measured).
    max_iter  = save['config']['max_iter']
    converged = max_iter not in [r['iter'] for r in history]

    best = max(records, key=lambda r: r['perf'].n_correct/r['perf'].n_decision)

    return {'p_correct':   best['perf'].n_correct/best['perf'].n_decision,
            'p_decision':  best['perf'].n_decision/best['perf'].n_trials,
            'iter_crit':   best['iter'] if converged else max_iter,
            'mean_reward': history[-1]['mean_reward'],
            'converged':   converged}

def seedvar_psychometric_metrics(trialsfile):
    """
    Psychometric shape parameters for one run.

    `psychometric` with `plots=None` returns {'<context><stream>': (evidence, P(right))};
    'll'/'ff' are the cued (relevant) stream, 'lf'/'fl' the ignored one. Sensitivity is
    the slope 1/sigma of the fitted cumulative Gaussian, so a network that gates context
    properly has a large relevant slope and a near-zero irrelevant one.

    """
    nan = {'slope_rel': np.nan, 'slope_irr': np.nan, 'selectivity': np.nan,
           'bias_rel': np.nan}
    if not os.path.isfile(trialsfile):
        return nan

    results = psychometric(trialsfile, plots=None)

    fits = {}
    for k, (x, p1) in results.items():
        if len(x) < 3:
            continue
        try:
            popt, _ = fittools.fit_psychometric(x, p1)
        except (RuntimeError, TypeError, ValueError):
            continue
        fits[k] = popt

    if not all(k in fits for k in ['ll', 'lf', 'fl', 'ff']):
        return nan

    slope = lambda k: 1/abs(fits[k]['sigma'])
    rel   = np.mean([slope('ll'), slope('ff')])
    irr   = np.mean([slope('lf'), slope('fl')])

    return {'slope_rel':   rel,
            'slope_irr':   irr,
            'selectivity': (rel - irr)/(rel + irr) if rel + irr > 0 else np.nan,
            'bias_rel':    np.mean([fits['ll']['mu'], fits['ff']['mu']])}

def variance_components(M):
    """
    Balanced two-factor random-effects decomposition of `M[i,j]`, rows = factor A
    (weight seed), columns = factor B (training seed), with one observation per cell.

    With n = 1 the interaction and the residual are confounded, so MS_AB estimates
    sigma^2_(interaction+error) and is also the error term for both main effects.

    Returns an OrderedDict of component -> (variance, F, p); F/p are None for the
    confounded term.

    """
    from scipy.stats import f as f_dist

    a, b = M.shape
    grand = M.mean()
    row   = M.mean(axis=1)
    col   = M.mean(axis=0)

    ms_a  = b*((row - grand)**2).sum()/(a - 1)
    ms_b  = a*((col - grand)**2).sum()/(b - 1)
    resid = M - row[:,None] - col[None,:] + grand
    ms_ab = (resid**2).sum()/((a - 1)*(b - 1))

    out = OrderedDict()
    if ms_ab > 0:
        out['weight']   = (max(0., (ms_a - ms_ab)/b), ms_a/ms_ab,
                           f_dist.sf(ms_a/ms_ab, a - 1, (a - 1)*(b - 1)))
        out['training'] = (max(0., (ms_b - ms_ab)/a), ms_b/ms_ab,
                           f_dist.sf(ms_b/ms_ab, b - 1, (a - 1)*(b - 1)))
    else:
        out['weight']   = (max(0., ms_a/b), None, None)
        out['training'] = (max(0., ms_b/a), None, None)
    out['interaction+residual'] = (ms_ab, None, None)

    return out

def seedvar(datapath, trialsroot, figspath):
    """
    How much of the run-to-run variability is the network (weight seed) and how much is
    the training experience (stimulus + action sampling)?

    """
    runs = seedvar_runs(datapath, trialsroot)
    if not runs:
        raise ValueError('no seedvar runs found in {}'.format(datapath))

    ws = sorted(set(r[0] for r in runs))
    ts = sorted(set(r[1] for r in runs))

    metrics = {}
    n_conv  = 0
    for w, t, savefile, trialsfile in runs:
        m = seedvar_savefile_metrics(savefile)
        n_conv += m.pop('converged')
        m.update(seedvar_psychometric_metrics(trialsfile))
        metrics[(w, t)] = m

    def grid_of(metric):
        return np.array([[metrics[(w, t)][metric] if (w, t) in metrics else np.nan
                          for t in ts] for w in ws])

    #=====================================================================================
    # Text report
    #=====================================================================================

    lines = ['{} runs, {} weight seeds x {} training seeds, converged {}/{}'
             .format(len(runs), len(ws), len(ts), n_conv, len(runs)), '']

    for metric in SEEDVAR_METRICS:
        M = grid_of(metric)

        lines.append(metric)
        lines.append('  {:>4s} '.format('w\\t') +
                     ' '.join('{:>10d}'.format(t) for t in ts))
        for i, w in enumerate(ws):
            lines.append('  {:>4d} '.format(w) +
                         ' '.join('{:>10.4g}'.format(v) for v in M[i]))

        if np.isnan(M).any() or M.shape[0] < 2 or M.shape[1] < 2:
            lines.append('  (incomplete grid, no decomposition)')
            lines.append('')
            continue

        comps = variance_components(M)
        total = sum(v for v, _, _ in comps.values())
        lines.append('  total variance {:.4g}'.format(total))
        for name, (var, F, p) in comps.items():
            frac = 100*var/total if total > 0 else 0.
            fp   = '' if F is None else '   F = {:.2f}, p = {:.3g}'.format(F, p)
            lines.append('  {:<22s} {:>10.4g}  {:>5.1f}%{}'.format(name, var, frac, fp))
        lines.append('')

    text = '\n'.join(lines) + '\n'
    print(text)

    utils.mkdir_p(figspath)
    with open(os.path.join(figspath, 'seedvar.txt'), 'w') as f:
        f.write(text)

    #=====================================================================================
    # Figure: per-metric spread across the grid, and its decomposition
    #=====================================================================================

    colors = ['k', '#d62728', '#1f77b4', '#2ca02c', '#9467bd']

    wpanel, hpanel, dx, dy = 0.42, 0.16, 0.16, 0.07
    x0, y0 = 0.1, 0.78

    fig = Figure(w=6.5, h=8)
    for i, metric in enumerate(SEEDVAR_PANELS):
        fig.add(metric,          [x0, y0 - i*(hpanel + dy), wpanel, hpanel])
        fig.add(metric + '-var', [x0 + wpanel + dx, y0 - i*(hpanel + dy),
                                  0.22, hpanel])

    for i, metric in enumerate(SEEDVAR_PANELS):
        M    = grid_of(metric)
        plot = fig[metric]
        for j, t in enumerate(ts):
            plot.plot(ws, M[:,j], 'o', ms=3.5, mew=0,
                      mfc=colors[j % len(colors)], label='t = {}'.format(t))
        plot.plot(ws, np.nanmean(M, axis=1), color='0.6', lw=1, zorder=1)
        plot.ylabel(metric)
        if i == len(SEEDVAR_PANELS) - 1:
            plot.xlabel('Weight seed')

        bars = fig[metric + '-var']
        if np.isnan(M).any():
            continue

        comps = variance_components(M)
        total = sum(v for v, _, _ in comps.values())
        if total <= 0:
            continue

        fracs = [100*v/total for v, _, _ in comps.values()]
        bars.bar(range(len(fracs)), fracs, color=['k', '#d62728', '0.7'], width=0.6)
        bars.set_xticks(range(len(fracs)))
        bars.set_xticklabels(['weight', 'train', 'int+res'], fontsize=6)
        bars.ylim(0, 100)
        bars.ylabel('% of variance')

    fig[SEEDVAR_PANELS[0]].text_upper_left('{} weight x {} training seeds'
                                           .format(len(ws), len(ts)))

    fig.save(path=figspath, name='seedvar')
    fig.close()

def do(action, args, config):
    """
    Manage tasks.

    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    #=====================================================================================

    if action == 'performance':
        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        performance(config['savefile'], plot)

        fig.save(path=config['figspath'], name='performance')
        fig.close()

    #=====================================================================================

    elif action == 'trial':
        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        # Resetting the RNG per checkpoint draws the same trial condition every time, so
        # the figures differ only in what the network did with it.
        rng_state        = pg.rng.get_state()
        rng_action_state = pg.rng_action.get_state()
        for iter_, _, figspath in checkpoint_targets(config, args):
            pg.rng.set_state(rng_state)
            pg.rng_action.set_state(rng_action_state)
            if iter_ is not None:
                pg.load_checkpoint(iter_)
            plot_trial(pg, model.spec, pg.rng, figspath, 'trial')

    #=====================================================================================

    elif 'trials' in action:
        # Optional `iters=` selects which checkpoints to roll out: 'all', or a
        # comma-separated list of iterations. The default rolls out the best parameters
        # only, into the top-level trials directory, as mante does.
        kwargs = dict(kv.split('=', 1) for kv in args if '=' in kv)
        positional = [a for a in args if '=' not in a]
        try:
            trials_per_condition = int(positional[0])
        except (IndexError, ValueError):
            trials_per_condition = 1000

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        iters_arg = kwargs.get('iters')
        if iters_arg is None:
            iters = [None]
        elif iters_arg == 'all':
            iters = pg.list_checkpoints()
        else:
            iters = [int(i) for i in iters_arg.split(',')]

        print("{} trials, checkpoints {}"
              .format(model.spec.n_conditions*trials_per_condition, iters))

        # One trial set, reused across checkpoints: resetting the RNG to the same state
        # regenerates identical stimuli, so any difference between checkpoints is due to
        # the parameters alone. The trials must be rebuilt each time because runtools.run
        # subsamples trial['time'] in place when dt_save is set.
        rng_state        = pg.rng.get_state()
        rng_action_state = pg.rng_action.get_state()
        for iter_ in iters:
            pg.rng.set_state(rng_state)
            pg.rng_action.set_state(rng_action_state)

            if iter_ is None:
                path = config['trialspath']
            else:
                pg.load_checkpoint(iter_)
                path = os.path.join(config['trialspath'], 'checkpoints',
                                    'iter_{:06d}'.format(iter_))
                utils.mkdir_p(path)

            trials = build_trials(pg, model.spec, trials_per_condition)
            runtools.run(action, trials, pg, path, dt_save=config['dt-save'])

    #=====================================================================================

    elif action == 'checkpointsweep':
        kwargs     = dict(kv.split('=', 1) for kv in args if '=' in kv)
        positional = [a for a in args if '=' not in a]
        try:
            trials_per_condition = int(positional[0])
        except (IndexError, ValueError):
            trials_per_condition = 50

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        iters_arg = kwargs.get('iters', 'all')
        checkpoints = pg.list_checkpoints() if iters_arg == 'all' \
                      else [int(i) for i in iters_arg.split(',')]

        print("checkpointsweep: {} checkpoints, {} trials/condition ({} trials/checkpoint)"
              .format(len(checkpoints), trials_per_condition,
                      model.spec.n_conditions*trials_per_condition))

        save            = utils.load(config['savefile'])
        history_by_iter = {r['iter']: r for r in save['training_history']}

        resultsfile = os.path.join(config['datapath'], 'checkpoint_metrics.pkl')
        saved = checkpointsweep_metrics(pg, model.spec, trials_per_condition, checkpoints,
                                        history_by_iter, resultsfile=resultsfile)
        print("checkpointsweep: wrote {} checkpoint records to {}"
              .format(len(saved['checkpoints']), resultsfile))

    #=====================================================================================

    elif action == 'psychometric':
        for iter_, trialspath, figspath in checkpoint_targets(config, args):
            fig = Figure(w=6, h=2.7)

            x0 = 0.12
            y0 = 0.2
            w  = 0.36
            h  = 0.7
            dx = 0.1

            fig.add('l', [x0, y0, w, h])
            fig.add('f', [fig[-1].right+dx, y0, w, h])

            # The earliest checkpoints never leave fixation, so there is no psychometric
            # curve to fit; report and move on rather than aborting the whole sweep.
            try:
                psychometric(runtools.behaviorfile(trialspath), fig.plots)
            except Exception as e:
                print("[ pagan.psychometric ] Skipping iter {}: {}".format(iter_, e))
                fig.close()
                continue

            fig['l'].xlabel('Net rightward location pulses')
            fig['l'].ylabel('Percent right')
            fig['f'].xlabel('Net rightward frequency pulses')

            fig.save(path=figspath, name='psychometric')
            fig.close()

    #=====================================================================================

    elif action == 'statespace':
        for iter_, trialspath, figspath in checkpoint_targets(config, args):
            # Likewise: with no correct trials there is nothing to regress or sort.
            try:
                statespace(runtools.activityfile(trialspath), (figspath, 'statespace'))
            except Exception as e:
                print("[ pagan.statespace ] Skipping iter {}: {}".format(iter_, e))

    #=====================================================================================

    elif action == 'sweep':
        # Sibling run directories of this run's own data directory.
        datapath = os.path.dirname(os.path.dirname(config['savefile']))
        figspath = os.path.join(os.path.dirname(config['figspath']), 'pagan_sweep')
        utils.mkdir_p(figspath)

        sweep(datapath, figspath)

    #=====================================================================================

    elif action == 'seedvar':
        # Sibling run directories of this run's own data and trials directories.
        datapath   = os.path.dirname(os.path.dirname(config['savefile']))
        trialsroot = os.path.dirname(config['trialspath'])
        figspath   = os.path.join(os.path.dirname(config['figspath']), 'pagan_seedvar')
        utils.mkdir_p(figspath)

        seedvar(datapath, trialsroot, figspath)

    #=====================================================================================

    else:
        raise ValueError("Unrecognized action \'{}\'.".format(action))
