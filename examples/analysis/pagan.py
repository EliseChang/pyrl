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

def is_active(r):
    return np.std(r, axis=0) > 0.1

def get_active_units(r, M):
    N   = r.shape[-1]
    M_  = (np.tile(M.T, (N, 1, 1))).T
    r_  = r*M_
    n   = np.sum(M)
    var = (r_**2).sum(axis=0).sum(axis=0)/n - (r_.sum(axis=0).sum(axis=0)/n)**2

    return np.where(np.sqrt(var) > 0.2)[0]

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

def psychometric(trialsfile, plots=None, **kwargs):
    """
    Psychometric functions against the realized evidence.

    Trials are binned by `integral_bin`, and each bin is plotted at the mean rightward
    evidence it actually contains, which gives `fittools.fit_psychometric` a meaningful
    x-axis. Binning by the nominal `strength` would be wrong: a sizeable fraction of
    strong-nominal trials realize an integral on the opposite side.

    """
    trials, A, R, M, perf = utils.load(trialsfile)

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
        rng_state = pg.rng.get_state()
        for iter_, _, figspath in checkpoint_targets(config, args):
            pg.rng.set_state(rng_state)
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
        rng_state = pg.rng.get_state()
        for iter_ in iters:
            pg.rng.set_state(rng_state)

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

    else:
        raise ValueError("Unrecognized action \'{}\'.".format(action))
