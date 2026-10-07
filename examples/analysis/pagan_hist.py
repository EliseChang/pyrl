"""
Analysis for examples/models/pagan_hist.py: pagan plus meta-RL trial-history inputs.

Everything that does not touch the input channels or the previous trial's outcome is
inherited from examples/analysis/pagan.py, exactly as the task module inherits from
examples/models/pagan.py. What differs:

  * `trial`       - rolls out a short *sequence* so the plotted trial has a real
                    `prev_outcome`, and shows all eight input channels.
  * `psychometric`- choice curves additionally split by the previous trial's outcome, in a
                    2x2 (cued context) x (evidence stream) grid. The pooled two-panel
                    figure of the base task is also written, as `psychometric_pooled`. Each
                    target additionally contributes one logistic-regression fit of the
                    choices (relevant and irrelevant stimulus, previous choice,
                    win-stay/lose-switch, and a bias) to `history_glm.pkl`.
  * `historyglm`  - new: those regression weights across every checkpoint, smoothed.
  * `weights`     - new: per-block RMS weight norms across every checkpoint.

NOTE on the import below. The base module is loaded under the name 'pagan_analysis', NOT
'pagan'. examples/do.py loads the analysis module (line 157) *before* the model (line 163);
if this file registered sys.modules['pagan'], then examples/models/pagan_hist.py's
`from pagan import *` would receive this analysis module instead of
examples/models/pagan.py, and Model.__init__ would abort on its required-config check. Do
not "simplify" this to a sys.path insert plus `import pagan`.

NOTE on `do`. The base `do()` resolves `psychometric`, `plot_trial` etc. against *pagan.py's*
globals at call time, so rebinding those names here would not change what it calls. `do` is
therefore reimplemented in full below rather than delegated to. For the same reason the base
module is left unmodified: `seedvar_psychometric_metrics` and `checkpoint_psychometric_metrics`
call the pooled `psychometric` and expect its four-key return.
"""
from __future__ import absolute_import, division

import imp
import os
from   collections import OrderedDict

import numpy as np

from pyrl          import fittools, runtools, tasktools, utils
from pyrl.figtools import apply_alpha, Figure

_HERE = os.path.dirname(os.path.abspath(__file__))
_base = imp.load_source('pagan_analysis', os.path.join(_HERE, 'pagan.py'))

# `from pagan import *`, done by hand because the module was loaded under another name.
# Anything redefined below simply shadows the inherited binding in *this* namespace; the
# inherited functions keep resolving their own globals in the base module, which is what
# makes the untouched analyses genuinely the base analyses.
for _k, _v in vars(_base).items():
    if not _k.startswith('_'):
        globals()[_k] = _v
del _k, _v

#/////////////////////////////////////////////////////////////////////////////////////////
# Previous-trial outcome
#/////////////////////////////////////////////////////////////////////////////////////////

# Ordered to match the input-channel order of examples/models/pagan_hist.py, so that the
# legend of the psychometric figure and the channel order of the trial figure agree.
PREV_CONDS = ['L-win', 'L-lose', 'R-win', 'R-lose']

PREV_COLORS = {'L-win':  Figure.colors('strongblue'),
               'L-lose': Figure.colors('lightblue'),
               'R-win':  Figure.colors('strongred'),
               'R-lose': Figure.colors('salmon')}

def prev_condition(trial):
    """
    'L-win'/'L-lose'/'R-win'/'R-lose' for the trial that preceded `trial`, or None.

    `trial['prev_outcome']` is written by PolicyGradient.run_trials as (choice, undiscounted
    terminal reward), and is None for the first trial of a batch and after any trial that
    ended without a decision. Those trials carry all-zero history channels, so the network
    saw no history on them; they are excluded rather than pooled into a fifth condition.

    Win is `reward > 0`, the same threshold examples/models/pagan_hist.py uses to pick the
    input channel. Since `prev_outcome` is only set when a choice was made, the reward is
    R_CORRECT or R_INCORRECT; the no-decision R_TERMINAL cannot appear here.

    """
    prev = trial.get('prev_outcome')
    if prev is None:
        return None

    choice, reward = prev

    return '{}-{}'.format(choice, 'win' if reward > 0 else 'lose')

#/////////////////////////////////////////////////////////////////////////////////////////
# Trial generation
#/////////////////////////////////////////////////////////////////////////////////////////

def build_trials(pg, spec, trials_per_condition):
    """
    As pagan.build_trials, but in random order.

    The base version walks the condition grid with tasktools.unravel_index, so trial n's
    condition is a fixed function of n and consecutive trials share most of their fields
    (measured on a 14400-trial run: left_right_freq 75% shared, strength_loc 88%,
    strength_freq 96%). PolicyGradient.run_trials hands trial n-1's outcome to trial n, so
    under that ordering `prev_outcome` is correlated with the *current* stimulus and any
    history-split of the choices mixes a stimulus effect into the apparent history effect.
    Training is unaffected: it draws each condition independently via task.get_condition.

    Shuffling leaves the multiset of conditions, and hence the pooled psychometric, exactly
    as it was.

    """
    trials = _base.build_trials(pg, spec, trials_per_condition)
    pg.rng.shuffle(trials)

    return trials

#/////////////////////////////////////////////////////////////////////////////////////////
# Behaviour, split by the previous trial's outcome
#/////////////////////////////////////////////////////////////////////////////////////////

# A bin must hold at least this many trials to be plotted. The four-way history split divides
# each (context, stream, level) cell of the pooled psychometric by roughly four again, and
# unevenly, since a trained network wins far more often than it loses; without a floor the
# sparse cells are sampling noise and drag the fit around.
MIN_TRIALS_PER_BIN = 10

def _psychometric_history_results(trials, perf, min_trials=MIN_TRIALS_PER_BIN):
    """
    As the base `_psychometric_results`, but each (cued context, evidence stream) cell is
    further split by the previous trial's outcome.

    Returns results[<cued context><stream>][<prev condition>] = (x, p1), keeping the base
    function's 'll'/'lf'/'fl'/'ff' outer keys so the two can be read against one another.

    """
    counts = {cond: {p: {} for p in PREV_CONDS} for cond in ['ll', 'lf', 'fl', 'ff']}
    for n, trial in enumerate(trials):
        if not perf.decisions[n]:
            continue

        prev = prev_condition(trial)
        if prev is None:
            continue

        choice = 1 if perf.choices[n] == 'R' else 0
        prefix = 'l' if trial['context'] == 'loc' else 'f'

        for stream, integral in [('l', rightward(trial['integral_loc'])),
                                 ('f', rightward(trial['integral_freq']))]:
            level = integral_bin(integral)
            if level is None:
                continue
            evidence, choices = counts[prefix+stream][prev].setdefault(level, ([], []))
            evidence.append(integral)
            choices.append(choice)

    results = {}
    for k, by_prev in counts.items():
        results[k] = {}
        for prev, by_level in by_prev.items():
            levels = [lev for lev in sorted(by_level)
                      if len(by_level[lev][1]) >= min_trials]
            if not levels:
                continue
            x  = np.array([np.mean(by_level[lev][0]) for lev in levels])
            p1 = np.array([np.mean(by_level[lev][1]) for lev in levels])
            results[k][prev] = (x, p1)

    return results

def psychometric_history(trialsfile, plots=None, **kwargs):
    """
    Psychometric functions against realized evidence, one line per previous outcome.

    `plots` is keyed as the results dict is: '<cued context><evidence stream>', i.e. 'll',
    'lf', 'fl', 'ff'.

    """
    trials, A, R, M, perf = utils.load(trialsfile)
    results = _psychometric_history_results(
        trials, perf, min_trials=kwargs.get('min_trials', MIN_TRIALS_PER_BIN))

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    if plots is None:
        return results

    lw = kwargs.get('lw', 1.25)
    ms = kwargs.get('ms', 4)

    for k, by_prev in results.items():
        plot  = plots[k]
        x_all = []
        for prev in PREV_CONDS:          # fixed order, so colours and z-order are stable
            if prev not in by_prev:
                continue

            x, p1 = by_prev[prev]
            color = PREV_COLORS[prev]

            plot.plot(x, 100*p1, 'o', ms=ms, mew=0, mfc=color, zorder=10)
            props = dict(color=color, lw=lw, zorder=5, label=prev)

            # A two-point curve cannot constrain a four-parameter cumulative Gaussian.
            if len(x) >= 3:
                try:
                    popt, func = fittools.fit_psychometric(x, p1)

                    fit_x  = np.linspace(min(x), max(x), 201)
                    fit_p1 = func(fit_x, **popt)
                    plot.plot(fit_x, 100*fit_p1, **props)
                except (RuntimeError, TypeError, ValueError):
                    plot.plot(x, 100*p1, **props)
            else:
                plot.plot(x, 100*p1, **props)

            x_all.append(x)

        if x_all:
            plot.lim('x', x_all)
        plot.ylim(0, 100)
        plot.yticks([0, 50, 100])

    return results

def _psychometric_figure():
    """
    2x2 grid: columns are the cued context, rows the evidence stream. The diagonal ('ll'
    top-left, 'ff' bottom-right) is the relevant stream, the off-diagonal the ignored one.

    """
    fig = Figure(w=6.5, h=5.5)

    x0, y0 = 0.11, 0.09
    w,  h  = 0.36, 0.35
    dx, dy = 0.13, 0.13
    y1     = y0 + h + dy

    fig.add('ll', [x0,               y1, w, h])
    fig.add('fl', [fig[-1].right+dx, y1, w, h])
    fig.add('lf', [x0,               y0, w, h])
    fig.add('ff', [fig[-1].right+dx, y0, w, h])

    return fig

#/////////////////////////////////////////////////////////////////////////////////////////
# Choice logistic regression
#/////////////////////////////////////////////////////////////////////////////////////////

# Design-matrix columns, in order. The intercept is fitted alongside them and reported as a
# fifth parameter, 'bias'.
GLM_REGRESSORS = ['stim_rel', 'stim_irr', 'prev_choice', 'wsls']

# Ridge on the slopes only. A network at ceiling separates the classes perfectly, and there
# the unpenalised likelihood is maximised by |beta| -> inf, so IRLS never converges; this
# keeps such fits finite while leaving well-conditioned checkpoints essentially
# unregularised.
GLM_L2 = 1e-3

# Fewer decided trials than this and the fit is not worth reporting.
MIN_GLM_TRIALS = 50

def glm_design(trials, perf):
    """
    (X, y) for the choice regression over every *decided* trial of the batch.

    Columns of X are GLM_REGRESSORS followed by a constant:

      stim_rel     signed rightward evidence of the cued stream,
      stim_irr     the same for the ignored stream,
      prev_choice  +1 if the previous trial was a right choice, -1 if left,
      wsls         prev_choice * (+1 win, -1 lose), i.e. win-stay/lose-switch,
      1            intercept, the side bias.

    Trials with no usable history (the first of a batch, or one following a trial that ended
    without a decision) carry all-zero history *inputs* to the network, so they enter here
    with prev_choice = wsls = 0 rather than being dropped: they still inform the stimulus
    terms and the intercept, which is what estimating the bias from all trials in the
    evaluation batch requires.

    y is 1 for a right choice, 0 for a left one.

    """
    X, y = [], []
    for n, trial in enumerate(trials):
        if not perf.decisions[n]:
            continue

        x_loc, x_freq = rightward(trial['integral_loc']), rightward(trial['integral_freq'])
        if trial['context'] == 'loc':
            stim_rel, stim_irr = x_loc, x_freq
        else:
            stim_rel, stim_irr = x_freq, x_loc

        prev = prev_condition(trial)
        if prev is None:
            prev_choice = wsls = 0.0
        else:
            side, outcome = prev.split('-')
            prev_choice   = 1.0 if side == 'R' else -1.0
            wsls          = prev_choice*(1.0 if outcome == 'win' else -1.0)

        X.append([stim_rel, stim_irr, prev_choice, wsls, 1.0])
        y.append(1.0 if perf.choices[n] == 'R' else 0.0)

    return np.array(X, dtype=float), np.array(y, dtype=float)

def fit_logistic(X, y, l2=GLM_L2, max_iter=100, tol=1e-8):
    """
    Penalised maximum likelihood by Newton-Raphson (IRLS). The last column of X is taken to
    be the intercept and is left unpenalised.

    """
    penalty     = l2*np.ones(X.shape[1])
    penalty[-1] = 0

    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1/(1 + np.exp(-X.dot(beta)))
        w = np.clip(p*(1 - p), 1e-9, None)

        gradient = X.T.dot(y - p) - penalty*beta
        hessian  = (X.T*w).dot(X) + np.diag(penalty)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian).dot(gradient)

        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break

    return beta

def history_glm(trials, perf, l2=GLM_L2, min_trials=MIN_GLM_TRIALS):
    """
    Fit `glm_design`, returning the five parameters plus the number of trials behind them,
    or None when the batch cannot support a fit (too few decisions, or one-sided choices).

    The four slopes sit on very different natural scales: stim_rel/stim_irr are raw
    pulse-count integrals (typically +-50), while prev_choice/wsls are +-1 indicator codes.
    Raw coefficients are therefore not comparable across regressors -- a small coefficient on
    a wide-range column can still dominate the logit. Each slope is additionally reported as a
    standardized effect size, under a '_std' suffix: the coefficient times that regressor's SD
    in this evaluation batch, i.e. the actual log-odds swing driven by one SD of that input.
    'bias' (the intercept) has no natural SD and is reported once, unstandardized.

    """
    X, y = glm_design(trials, perf)
    if len(y) < min_trials or len(np.unique(y)) < 2:
        return None

    beta   = fit_logistic(X, y, l2=l2)
    record = OrderedDict(zip(GLM_REGRESSORS + ['bias'], [float(b) for b in beta]))

    sds = X[:, :-1].std(axis=0)
    for k, sd in zip(GLM_REGRESSORS, sds):
        record[k + '_std'] = record[k]*float(sd)

    record['n_trials'] = int(len(y))

    return record

GLM_FILE = 'history_glm.pkl'

def save_glm_record(datapath, iter_, record):
    """
    Merge one checkpoint's parameters into <datapath>/history_glm.pkl, replacing any record
    for the same iteration so that a re-run updates in place. The default target has no
    iteration (it is the best parameters) and is kept separately, under 'best'.

    """
    resultsfile = os.path.join(datapath, GLM_FILE)

    saved = {'regressors': GLM_REGRESSORS + ['bias'], 'checkpoints': [], 'best': None}
    if os.path.isfile(resultsfile):
        saved.update(utils.load(resultsfile))

    if iter_ is None:
        saved['best'] = record
    else:
        record = OrderedDict(record)
        record['iter'] = iter_
        saved['checkpoints'] = [r for r in saved['checkpoints'] if r['iter'] != iter_]
        saved['checkpoints'].append(record)
        saved['checkpoints'].sort(key=lambda r: r['iter'])

    utils.save(resultsfile, saved)

    return resultsfile

# The four series the summary figure shows. 'wsls' is the prev_choice x prev_reward
# interaction, labelled 'Prev. reward' because that is the term it isolates once prev_choice
# is already in the model. stim_irr is saved but not plotted.
#
# The three slopes are plotted as standardized effect sizes (_std, see history_glm), not raw
# coefficients: stim_rel's natural scale (+-50 pulses) otherwise dwarfs prev_choice/wsls's
# +-1 coding, making a raw-coefficient comparison misleading. 'bias' has no natural SD to
# standardize by and is plotted as-is; all four series are in log-odds units.
GLM_PLOT_SERIES = [('stim_rel_std',    'Stimulus (relevant)', 'k'),
                   ('prev_choice_std', 'Prev. choice',        '#9f1f63'),
                   ('wsls_std',        'Prev. reward',        '#662d91'),
                   ('bias',            'Side bias',           '#ec008c')]

# In checkpoints, not iterations: consecutive checkpoints are `checkfreq` iterations apart.
GLM_SMOOTH_SIGMA = 2

def gaussian_smooth(y, sigma):
    """Gaussian smoothing along the checkpoint axis, edges held at the end values."""
    y = np.asarray(y, dtype=float)
    if sigma <= 0 or len(y) < 2:
        return y

    half    = int(np.ceil(3*sigma))
    kernel  = np.exp(-0.5*(np.arange(-half, half+1)/float(sigma))**2)
    kernel /= kernel.sum()
    padded  = np.concatenate([np.repeat(y[0], half), y, np.repeat(y[-1], half)])

    return np.convolve(padded, kernel, mode='valid')

def plot_history_glm(resultsfile, figspath, name='history_glm', sigma=GLM_SMOOTH_SIGMA):
    """
    The four weights of GLM_PLOT_SERIES across checkpoints. The raw values are drawn faintly
    underneath the smoothed line, so that the smoothing cannot hide a trend it invented.

    """
    saved   = utils.load(resultsfile)
    records = saved['checkpoints']
    if not records:
        print("[ pagan_hist.historyglm ] No checkpoint records in {}".format(resultsfile))
        return

    iters = np.array([r['iter'] for r in records], dtype=float)

    fig  = Figure(w=5.5, h=3.6)
    plot = fig.add()

    for key, label, color in GLM_PLOT_SERIES:
        y = np.array([r[key] for r in records], dtype=float)
        plot.plot(iters, y, color=apply_alpha(color, 0.25), lw=0.75, zorder=4)
        plot.plot(iters, gaussian_smooth(y, sigma), color=color, lw=1.75, label=label,
                  zorder=8)

    plot.xlim(iters[0], iters[-1])
    plot.hline(0, color='0.6', lw=0.75, linestyle='--')
    plot.xlabel('Training iteration')
    plot.ylabel('Standardized effect (log-odds)')
    plot.legend(loc='upper left', fontsize=7)

    fig.save(path=figspath, name=name)
    fig.close()

    return records

#/////////////////////////////////////////////////////////////////////////////////////////
# Single trial, with real history
#/////////////////////////////////////////////////////////////////////////////////////////

# Trials rolled out before the one that gets plotted. run_trials threads `prev_outcome`
# through the batch it is handed, so a batch of this size gives the plotted trial a genuine
# history input instead of the all-zero channels a batch of one would produce. Kept small:
# this is a figure, not a statistic.
N_HISTORY_TRIALS = 10

# The four history channels reuse the psychometric palette, so the two figures can be read
# against one another.
TRIAL_CHANNELS = [('LMR',         Figure.colors('orange')),
                  ('HML',         Figure.colors('purple')),
                  ('LOC',         Figure.colors('blue')),
                  ('FREQ',        Figure.colors('green')),
                  ('PREV-L-WIN',  PREV_COLORS['L-win']),
                  ('PREV-L-LOSE', PREV_COLORS['L-lose']),
                  ('PREV-R-WIN',  PREV_COLORS['R-win']),
                  ('PREV-R-LOSE', PREV_COLORS['R-lose'])]

def plot_trial_sequence(pg, m, rng, figspath, name, n_trials=N_HISTORY_TRIALS, trials=None):
    """
    Observables, policy, actions and rewards for one trial of a short sequence.

    The sequence exists only so that the plotted trial has a previous outcome to show on its
    history channels; the earlier trials are not plotted. The latest trial that actually has
    a history is preferred, falling back to the last trial when nothing in the batch decided
    - which is the normal state of the earliest checkpoints.

    """
    if trials is None:
        trials = [pg.task.get_condition(rng, pg.dt) for _ in xrange(n_trials)]

    (U, Q, Q_b, Z, Z_b, A, R, M, init, init_b, x0, x0_b,
     perf) = pg.run_trials(trials)

    n = len(trials) - 1
    for i in xrange(len(trials)-1, -1, -1):
        if trials[i].get('prev_outcome') is not None:
            n = i
            break
    trial = trials[n]

    U = U[:,n,:]
    Z = Z[:,n,:]
    A = A[:,n,:]
    R = R[:,n]
    M = M[:,n]
    b = Z_b[:,n]
    t = int(np.sum(M))

    w  = 0.62
    h  = 0.18
    x  = 0.17
    dy = h + 0.05
    y0 = 0.08
    y1 = y0 + dy
    y2 = y1 + dy
    y3 = y2 + dy

    fig   = Figure(h=6.5)
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
    for label, color in TRIAL_CHANNELS:
        if label not in m.inputs:
            continue
        u = U[:t,m.inputs[label]]
        plot.plot(obs_time, u, 'o', ms=4, mew=0, mfc=color)
        plot.plot(obs_time, u, lw=1.25, color=color, label=label)

    plot.xlim(*xlim)
    # The pulse channels are signed integer counts, so autoscale symmetrically rather than
    # clamping to [0, 1]; the context and history channels are 0/1 and sit inside that range.
    umax = max(1, np.max(np.abs(U[:t])))
    plot.ylim(-umax, umax)
    plot.ylabel('Observables')

    color = Figure.colors('orange' if trial['correct_side'] == 'L' else 'purple')
    prev  = prev_condition(trial)
    plot.text_upper_right('{} | prev={} | I_loc={:+d} I_freq={:+d} $\\to$ {}'
                          .format(trial['context'], prev if prev else 'none',
                                  trial['integral_loc'], trial['integral_freq'],
                                  trial['correct_side']),
                          color=color, fontsize=6.5)

    props = {'prop': {'size': 6}, 'handlelength': 1.2,
             'handletextpad': 1.0, 'labelspacing': 0.55}
    plot.legend(bbox_to_anchor=(1.42, 1.0), **props)

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
    plot.legend(bbox_to_anchor=(1.45, 0.8), **props)

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
# Per-block weight norms across checkpoints
#/////////////////////////////////////////////////////////////////////////////////////////

# (label, network, parameter, input-channel names or None for the whole matrix). The three
# input blocks are separated because what the network learns to weight - stimulus against
# context cue against trial history - is the point of this task variant.
WEIGHT_BLOCKS = [
    ('Win stimulus', 'policy',   'Win',  ['LMR', 'HML']),
    ('Win context',  'policy',   'Win',  ['LOC', 'FREQ']),
    ('Win history',  'policy',   'Win',  ['PREV-L-WIN', 'PREV-L-LOSE',
                                          'PREV-R-WIN', 'PREV-R-LOSE']),
    ('Wrec policy',  'policy',   'Wrec', None),
    ('Wrec value',   'baseline', 'Wrec', None),
    ('Wout policy',  'policy',   'Wout', None),
    ]

def rms(W):
    """
    Per-element RMS, i.e. the Frobenius norm divided by sqrt(number of elements).

    The six blocks differ in size by more than an order of magnitude (2x100 for a Win group
    against 100x100 for Wrec), so a raw Frobenius norm would compare block sizes rather than
    typical weight magnitudes.

    """
    W = np.asarray(W, dtype=float)
    if W.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(W**2)))

def checkpoint_inputs(ckpt, fallback=None):
    """
    The input-channel map the checkpointed network was trained with.

    `inputs` is required config, so it is in every checkpoint's stored config. Preferring it
    over the current model module means a checkpoint is always indexed by its own layout.

    """
    inputs = ckpt.get('config', {}).get('inputs')

    return fallback if inputs is None else inputs

def weight_norms(ckpt, inputs):
    """
    Per-element RMS norm of each block of one checkpoint's parameters.

    Row indexing of `Win` is what makes this gates-agnostic: Win is (Nin, N) with gates off
    and (Nin, 3*N) with gates on, but the *rows* index input channels either way, so
    `Win[rows]` selects the same channels and simply spans the update/reset blocks as well
    when they exist. `Wrec` is (N, N) in both cases; `Wrec_gates`, which exists only with
    gates on, is deliberately not one of the six blocks.

    Missing parameters and channels absent from this run's input map (a plain pagan
    checkpoint, say, which has no PREV-* channels) give nan rather than raising, so a mixed
    set of checkpoints still plots.

    """
    norms = OrderedDict()
    for label, net, param, channels in WEIGHT_BLOCKS:
        params = dict(ckpt.get(net + '_params', {}))
        W      = params.get(param)
        if W is None:
            norms[label] = np.nan
            continue

        W = np.asarray(W)
        if channels is not None:
            rows = [inputs[c] for c in channels if c in inputs]
            if not rows or max(rows) >= W.shape[0]:
                norms[label] = np.nan
                continue
            W = W[rows]

        norms[label] = rms(W)

    return norms

def weight_norms_across_checkpoints(ckptpath, fallback_inputs=None):
    """
    (iters, {block label: array of norms}) over every checkpoint in `ckptpath`.

    Reads the checkpoint pickles directly rather than going through
    PolicyGradient.load_checkpoint: nothing here needs a network, and building a
    PolicyGradient would compile four Theano functions for no reason.

    """
    iters = sorted(int(f[5:-4]) for f in os.listdir(ckptpath)
                   if f.startswith('iter_') and f.endswith('.pkl'))
    if not iters:
        raise ValueError("no checkpoints in {}".format(ckptpath))

    labels = [b[0] for b in WEIGHT_BLOCKS]
    norms  = OrderedDict((label, []) for label in labels)
    for iter_ in iters:
        ckpt = utils.load(os.path.join(ckptpath, 'iter_{:06d}.pkl'.format(iter_)))
        n    = weight_norms(ckpt, checkpoint_inputs(ckpt, fallback_inputs))
        for label in labels:
            norms[label].append(n[label])

    return (np.asarray(iters),
            OrderedDict((k, np.asarray(v)) for k, v in norms.items()))

def plot_weights(iters, norms, figspath, name='weights'):
    """
    One row per block, each showing its per-element RMS norm across training.

    """
    labels = [b[0] for b in WEIGHT_BLOCKS]
    colors = [Figure.colors(c) for c in
              ['orange', 'blue', 'strongred', 'darkblue', 'green', 'purple']]

    x0, w = 0.17, 0.76
    y0, h = 0.055, 0.128
    dy    = 0.026

    fig = Figure(w=6, h=9)
    for i, label in enumerate(labels):
        # Row 0 at the top, so the figure reads in WEIGHT_BLOCKS order.
        y = y0 + (len(labels) - 1 - i)*(h + dy)
        fig.add(label, [x0, y, w, h])

    for i, label in enumerate(labels):
        plot = fig[label]
        y    = norms[label]
        ok   = np.isfinite(y)

        if ok.any():
            plot.plot(iters[ok], y[ok], lw=1.25, color=colors[i])
            plot.plot(iters[ok], y[ok], 'o', ms=3, mew=0, mfc=colors[i])
            plot.lim('y', [y[ok]], lower=0)
        else:
            plot.text_upper_left('(not present in these checkpoints)', color='0.5')

        plot.xlim(min(iters), max(iters))
        plot.ylabel('RMS')
        plot.text_upper_left(label, color=colors[i])
        if i == len(labels) - 1:
            plot.xlabel('Policy gradient updates')

    fig.save(path=figspath, name=name)
    fig.close()

#/////////////////////////////////////////////////////////////////////////////////////////

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
        kwargs    = dict(kv.split('=', 1) for kv in args if '=' in kv)
        n_history = int(kwargs.get('n', N_HISTORY_TRIALS))

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        # Resetting the RNG per checkpoint draws the same trial conditions every time, so
        # the figures differ only in what the network did with them.
        rng_state        = pg.rng.get_state()
        rng_action_state = pg.rng_action.get_state()
        for iter_, _, figspath in checkpoint_targets(config, args):
            pg.rng.set_state(rng_state)
            pg.rng_action.set_state(rng_action_state)
            if iter_ is not None:
                pg.load_checkpoint(iter_)
            plot_trial_sequence(pg, model.spec, pg.rng, figspath, 'trial',
                                n_trials=n_history)

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

    elif action == 'weights':
        ckptpath = os.path.join(config['datapath'], 'checkpoints')
        iters, norms = weight_norms_across_checkpoints(
            ckptpath, fallback_inputs=vars(config['model'].spec).get('inputs'))

        labels = [b[0] for b in WEIGHT_BLOCKS]
        lines  = ['{:>8s} '.format('iter')
                  + ' '.join('{:>14s}'.format(l) for l in labels)]
        for i, iter_ in enumerate(iters):
            lines.append('{:>8d} '.format(iter_)
                         + ' '.join('{:>14.5g}'.format(norms[l][i]) for l in labels))
        text = '\n'.join(lines) + '\n'
        print(text)

        utils.mkdir_p(config['figspath'])
        with open(os.path.join(config['figspath'], 'weights.txt'), 'w') as f:
            f.write(text)
        utils.save(os.path.join(config['datapath'], 'weight_norms.pkl'),
                   {'iters': iters, 'norms': norms, 'blocks': WEIGHT_BLOCKS})

        plot_weights(iters, norms, config['figspath'])

    #=====================================================================================

    elif action == 'psychometric':
        for iter_, trialspath, figspath in checkpoint_targets(config, args):
            trialsfile = runtools.behaviorfile(trialspath)

            #-----------------------------------------------------------------------------
            # Choice regression, saved for `historyglm`
            #-----------------------------------------------------------------------------

            try:
                trials, _, _, _, perf = utils.load(trialsfile)
                record = history_glm(trials, perf)
            except Exception as e:
                print("[ pagan_hist.psychometric ] No GLM for iter {}: {}".format(iter_, e))
                record = None

            if record is not None:
                save_glm_record(config['datapath'], iter_, record)
                print("[ pagan_hist.psychometric ] iter {} GLM  ".format(iter_)
                      + '  '.join('{}={:+.3f}'.format(k, record[k])
                                  for k in GLM_REGRESSORS + ['bias'])
                      + '  (n={})'.format(record['n_trials']))

            #-----------------------------------------------------------------------------
            # Split by the previous trial's outcome
            #-----------------------------------------------------------------------------

            fig = _psychometric_figure()

            # The earliest checkpoints never leave fixation, so there is no psychometric
            # curve to fit; report and move on rather than aborting the whole sweep.
            try:
                psychometric_history(trialsfile, fig.plots)
            except Exception as e:
                print("[ pagan_hist.psychometric ] Skipping iter {}: {}".format(iter_, e))
                fig.close()
                continue

            for k in ['ll', 'lf', 'fl', 'ff']:
                fig[k].ylabel('Percent right')
            fig['ll'].text_upper_center('Location context')
            fig['fl'].text_upper_center('Frequency context')
            fig['ll'].xlabel('Net rightward location pulses')
            fig['fl'].xlabel('Net rightward location pulses')
            fig['lf'].xlabel('Net rightward frequency pulses')
            fig['ff'].xlabel('Net rightward frequency pulses')

            # Legend as coloured text: there is no room outside the 2x2 grid, and the
            # upper-left corner of every panel is empty (P(right) is near 0 there).
            for j, prev in enumerate(PREV_CONDS):
                fig['ll'].text_upper_left('prev ' + prev, dy=-0.095*j,
                                          color=PREV_COLORS[prev], fontsize=6.5)

            fig.save(path=figspath, name='psychometric')
            fig.close()

            #-----------------------------------------------------------------------------
            # Pooled over history, for comparison with the base task
            #-----------------------------------------------------------------------------

            fig = Figure(w=6, h=2.7)
            fig.add('l', [0.12, 0.2, 0.36, 0.7])
            fig.add('f', [fig[-1].right+0.1, 0.2, 0.36, 0.7])
            try:
                psychometric(trialsfile, fig.plots)

                fig['l'].xlabel('Net rightward location pulses')
                fig['l'].ylabel('Percent right')
                fig['f'].xlabel('Net rightward frequency pulses')

                fig.save(path=figspath, name='psychometric_pooled')
            except Exception as e:
                print("[ pagan_hist.psychometric ] Skipping pooled, iter {}: {}"
                      .format(iter_, e))
            fig.close()

    #=====================================================================================

    elif action == 'historyglm':
        kwargs = dict(kv.split('=', 1) for kv in args if '=' in kv)
        sigma  = float(kwargs.get('sigma', GLM_SMOOTH_SIGMA))

        plot_history_glm(os.path.join(config['datapath'], GLM_FILE), config['figspath'],
                         sigma=sigma)

    #=====================================================================================

    elif action == 'statespace':
        for iter_, trialspath, figspath in checkpoint_targets(config, args):
            # Likewise: with no correct trials there is nothing to regress or sort.
            try:
                statespace(runtools.activityfile(trialspath), (figspath, 'statespace'))
            except Exception as e:
                print("[ pagan_hist.statespace ] Skipping iter {}: {}".format(iter_, e))

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
