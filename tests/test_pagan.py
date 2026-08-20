"""
Offline checks for examples/models/pagan.py.

Run from the repository root under the Python 2.7 environment:

  python tests/test_pagan.py

Checks 1-6 import only numpy and pyrl.tasktools, so they run in seconds. Check 7 builds the
policy gradient networks and therefore imports Theano; skip it with --no-theano.
"""
from __future__ import division, print_function

import imp
import os
import sys
import time
from collections import Counter

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pyrl import configs
from pyrl.performance import Performance2AFC

MODELFILE = os.path.join(REPO, 'examples', 'models', 'pagan.py')
m  = imp.load_source('pagan', MODELFILE)
DT = configs.default['dt']

#/////////////////////////////////////////////////////////////////////////////////////////
# Helpers
#/////////////////////////////////////////////////////////////////////////////////////////

def Tmax():
    return int(m.tmax/DT) + 1

def rollout(rng, trial, policy, R_TERMINAL=None):
    """
    Mirror PolicyGradient.run_trials: no input is given on the first step, get_step is
    called with t+1, and abort_on_last_t ends the trial at t = Tmax-1 without calling
    get_step. `policy(t, trial)` returns the action taken at time index t.

    Returns (us, rewards, status, t_last) where us[k] is the observation consumed at time
    index k+1 (i.e. the observation returned by the step taken at index k), and t_last is
    the last index at which an action was taken (run_trials breaks at the top of the next
    iteration, so its loop variable overshoots by one).
    """
    if R_TERMINAL is None:
        R_TERMINAL = m.R_ABORTED

    us      = []
    rewards = []

    t = t_last = 0
    u, reward, status = m.get_step(rng, DT, trial, t+1, policy(t, trial))
    us.append(u)
    rewards.append(reward)

    for t in range(1, Tmax()):
        if not status['continue']:
            break
        t_last = t
        a = policy(t, trial)
        if t == Tmax() - 1:
            u      = np.zeros(len(m.inputs))
            reward = R_TERMINAL
            status = {'continue': False, 'reward': reward}
        else:
            u, reward, status = m.get_step(rng, DT, trial, t+1, a)
        us.append(u)
        rewards.append(reward)

    return np.array(us), rewards, status, t_last

def fixate_then(choice_action, t_choice):
    """Policy: fixate, then take `choice_action` at time index `t_choice`."""
    def policy(t, trial):
        if t == t_choice:
            return choice_action
        return m.actions['FIXATE']
    return policy

def always_fixate(t, trial):
    return m.actions['FIXATE']

def action_for(side):
    return m.actions['CHOOSE-LEFT'] if side == 'L' else m.actions['CHOOSE-RIGHT']

def other(side):
    return 'R' if side == 'L' else 'L'

def check(name):
    print('[ok] ' + name)

#/////////////////////////////////////////////////////////////////////////////////////////
# 1. Module contract
#/////////////////////////////////////////////////////////////////////////////////////////

def test_module_contract():
    for k in configs.required:
        assert hasattr(m, k), 'missing required config key: {}'.format(k)

    # Every configs.default key is silently harvested from the module namespace, so an
    # innocuous module-level name like `N` would become a network hyperparameter.
    harvested = set(vars(m)) & set(configs.default)
    assert harvested == set(['R_ABORTED', 'max_iter', 'checkfreq']), \
        'unexpected config keys in module namespace: {}'.format(harvested)

    assert len(m.inputs)  == 4, m.inputs
    assert len(m.actions) == 3, m.actions
    assert m.n_conditions == 72, m.n_conditions
    assert m.n_gradient   == 72, m.n_gradient
    assert m.n_validation == 3600, m.n_validation
    assert m.tmax == 3300, m.tmax
    assert (m.R_ABORTED, m.R_CORRECT, m.R_INCORRECT) == (-1, +1, -1)

    check('module contract')

#/////////////////////////////////////////////////////////////////////////////////////////
# 2. Epoch and index invariants
#/////////////////////////////////////////////////////////////////////////////////////////

def test_epochs():
    rng   = np.random.RandomState(0)
    trial = m.get_condition(rng, DT)

    assert trial['epochs']['stimulus'] == list(range(0, 130)),   trial['epochs']['stimulus'][:5]
    assert trial['epochs']['decision'] == list(range(130, 330)), trial['epochs']['decision'][:5]
    assert len(trial['time']) == 331 == Tmax()
    assert trial['stim_t0'] == 1,   trial['stim_t0']
    assert trial['dec_t0']  == 130, trial['dec_t0']
    assert len(trial['lmr']) == len(trial['hml']) == 129
    assert trial['lmr'].dtype == np.int16
    assert trial['hml'].dtype == np.int16

    check('epoch and index invariants')

#/////////////////////////////////////////////////////////////////////////////////////////
# 3. Pulse statistics
#/////////////////////////////////////////////////////////////////////////////////////////

def test_pulse_statistics(n_trials=5000):
    rng    = np.random.RandomState(1)
    trials = [m.get_condition(rng, DT) for _ in range(n_trials)]

    nbins = len(trials[0]['lmr'])
    lam   = m.pulse_rate*DT/1000

    # Expected total pulses per trial = lam*nbins.
    expected_total = lam*nbins
    assert abs(expected_total - 51.6) < 0.01, expected_total

    # Mean integral vs nominal ratio.
    by_lr = {}
    by_hl = {}
    for t in trials:
        by_lr.setdefault(t['lr_ratio'], []).append(t['integral_loc'])
        by_hl.setdefault(t['hl_ratio'], []).append(t['integral_freq'])

    ratios = set([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    assert set(np.round(list(by_lr), 2)) == ratios, sorted(by_lr)
    assert set(np.round(list(by_hl), 2)) == ratios, sorted(by_hl)

    print('  {:>6} {:>8} {:>8} {:>8} {:>8}'.format('ratio', 'E[I]', 'mean', 'sd', 'n'))
    for ratio in sorted(by_lr):
        x        = np.array(by_lr[ratio], dtype=float)
        expected = expected_total*(2*ratio - 1)
        se       = x.std()/np.sqrt(len(x))
        print('  {:>6} {:>8.2f} {:>8.2f} {:>8.2f} {:>8d}'.format(
            ratio, expected, x.mean(), x.std(), len(x)))
        assert abs(x.mean() - expected) < 4*se, \
            'lr_ratio {}: mean {} vs expected {}'.format(ratio, x.mean(), expected)
        # Var(integral) = total expected pulses, independent of the ratio. The standard
        # error of a sample sd is sd/sqrt(2n), so allow 5 of them.
        sd    = np.sqrt(expected_total)
        se_sd = sd/np.sqrt(2*len(x))
        assert abs(x.std() - sd) < 5*se_sd, \
            'lr_ratio {}: sd {} vs expected {}'.format(ratio, x.std(), sd)

    # Per-bin second moments, within a condition (pooling across conditions would add the
    # spread of the per-condition means). Var(LMR) = Var(HML) = lam, and the covariance
    # Cov(LMR, HML) = lam*(2*lr - 1)*(2*hl - 1) is the signature of the four-sub-stream
    # construction: the two evidence channels are correlated within a trial.
    print('  {:>5} {:>5} {:>9} {:>9} {:>9} {:>9}'.format(
        'lr', 'hl', 'var(lmr)', 'var(hml)', 'cov', 'pred cov'))
    for lr, hl in [(0.75, 0.35), (0.25, 0.65), (0.55, 0.55)]:
        sub = [t for t in trials
               if abs(t['lr_ratio'] - lr) < 1e-9 and abs(t['hl_ratio'] - hl) < 1e-9]
        assert len(sub) > 50, len(sub)
        a = np.concatenate([t['lmr'] for t in sub]).astype(float)
        b = np.concatenate([t['hml'] for t in sub]).astype(float)
        measured  = np.mean(a*b) - a.mean()*b.mean()
        predicted = lam*(2*lr - 1)*(2*hl - 1)
        print('  {:>5} {:>5} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f}'.format(
            lr, hl, a.var(), b.var(), measured, predicted))
        assert abs(a.var() - lam) < 0.02, 'lmr per-bin var {} != {}'.format(a.var(), lam)
        assert abs(b.var() - lam) < 0.02, 'hml per-bin var {} != {}'.format(b.var(), lam)
        assert abs(measured - predicted) < 0.02, \
            'cov {} vs predicted {} at lr={}, hl={}'.format(measured, predicted, lr, hl)

    # Marginals of the sampled conditions.
    counts = Counter([t['context'] for t in trials])
    assert abs(counts['loc']/n_trials - 0.5) < 0.03, counts
    for key, values in [('left_right_loc', m.left_rights),
                        ('left_right_freq', m.left_rights),
                        ('strength_loc', m.strengths),
                        ('strength_freq', m.strengths)]:
        counts = Counter([t[key] for t in trials])
        assert set(counts) == set(values), (key, counts)
        for v in values:
            assert abs(counts[v]/n_trials - 1/len(values)) < 0.03, (key, counts)

    # The realized side agrees with the nominal side at a rate set by the strength, which
    # also pins down the sign convention (left_right = -1 -> majority left/high ->
    # positive integral -> 'L').
    print('  {:>8} {:>10} {:>8}'.format('strength', 'P(match)', 'n'))
    for strength, lo in [(0.45, 0.74), (0.35, 0.97), (0.25, 0.999)]:
        sub = [t for t in trials
               if t['context'] == 'loc' and abs(t['strength_loc'] - strength) < 1e-9]
        match = [np.sign(t['integral_loc']) == -np.sign(t['left_right_loc']) for t in sub]
        p = np.mean(match)
        print('  {:>8} {:>10.3f} {:>8d}'.format(strength, p, len(sub)))
        assert p >= lo, 'strength {}: P(match) = {} < {}'.format(strength, p, lo)

    check('pulse statistics')

#/////////////////////////////////////////////////////////////////////////////////////////
# 4. Rollouts
#/////////////////////////////////////////////////////////////////////////////////////////

def test_rollouts(n_trials=200):
    rng = np.random.RandomState(2)

    for _ in range(n_trials):
        trial = m.get_condition(rng, DT)
        nbins = len(trial['lmr'])

        #---------------------------------------------------------------------------------
        # Oracle: fixate through the stimulus, then choose the realized-correct side.
        #---------------------------------------------------------------------------------

        us, rewards, status, t_last = rollout(
            rng, trial, fixate_then(action_for(trial['correct_side']), trial['dec_t0']))
        assert rewards[-1] == m.R_CORRECT, rewards[-1]
        assert status['correct'] is True
        assert status['choice'] == trial['correct_side']
        assert status['t_choice'] == trial['dec_t0']
        assert status['continue'] is False
        assert t_last == trial['dec_t0']

        # us[k] is consumed at index k+1, so the delivered observations are us[:-1] shifted
        # by one: index k+1 carries pulse bin (k+1) - stim_t0.
        delivered = us[:-1]
        lmr_sum   = delivered[:, m.inputs['LMR']].sum()
        hml_sum   = delivered[:, m.inputs['HML']].sum()
        assert lmr_sum == trial['integral_loc'],  (lmr_sum, trial['integral_loc'])
        assert hml_sum == trial['integral_freq'], (hml_sum, trial['integral_freq'])

        # The cued context channel is on for exactly the steps carrying pulses.
        cued   = m.inputs['LOC'] if trial['context'] == 'loc' else m.inputs['FREQ']
        uncued = m.inputs['FREQ'] if trial['context'] == 'loc' else m.inputs['LOC']
        on     = set(np.where(delivered[:, cued] == 1)[0] + 1)
        assert on == set(range(trial['stim_t0'], trial['stim_t0'] + nbins)), sorted(on)[:5]
        assert delivered[:, uncued].sum() == 0

        # Nothing is delivered once the decision epoch starts.
        after = delivered[trial['dec_t0'] - 1:]
        assert np.all(after == 0), after[after != 0][:5]

        #---------------------------------------------------------------------------------
        # Wrong choice
        #---------------------------------------------------------------------------------

        _, rewards, status, _ = rollout(
            rng, trial,
            fixate_then(action_for(other(trial['correct_side'])), trial['dec_t0']))
        assert rewards[-1] == m.R_INCORRECT, rewards[-1]
        assert status['correct'] is False
        assert status['continue'] is False

        #---------------------------------------------------------------------------------
        # Choosing on the last legal step
        #---------------------------------------------------------------------------------

        t_last_legal = Tmax() - 2
        _, rewards, status, _ = rollout(
            rng, trial, fixate_then(action_for(trial['correct_side']), t_last_legal))
        assert rewards[-1] == m.R_CORRECT, rewards[-1]
        assert status['t_choice'] == t_last_legal

    #-------------------------------------------------------------------------------------
    # Breaking fixation
    #-------------------------------------------------------------------------------------

    trial = m.get_condition(rng, DT)
    _, rewards, status, t_last = rollout(
        rng, trial, fixate_then(m.actions['CHOOSE-LEFT'], 5))
    assert rewards[-1] == m.R_ABORTED, rewards[-1]
    assert status['continue'] is False
    assert 'correct' not in status
    assert t_last == 5

    perf = Performance2AFC()
    perf.update(trial, status)
    assert perf.n_trials == 1 and perf.n_decision == 0 and perf.n_correct == 0

    #-------------------------------------------------------------------------------------
    # Never deciding
    #-------------------------------------------------------------------------------------

    _, rewards, status, t_last = rollout(rng, trial, always_fixate)
    assert t_last == Tmax() - 1, t_last
    assert rewards[-1] == m.R_ABORTED, rewards[-1]
    assert 'correct' not in status

    perf = Performance2AFC()
    perf.update(trial, status)
    assert perf.n_trials == 1 and perf.n_decision == 0

    #-------------------------------------------------------------------------------------
    # Ties get a definite, randomly drawn correct side
    #-------------------------------------------------------------------------------------

    ties = []
    while len(ties) < 200:
        trial = m.get_condition(rng, DT)
        if trial['integral'] == 0:
            ties.append(trial)

    for trial in ties[:20]:
        _, rewards, status, _ = rollout(
            rng, trial, fixate_then(action_for(trial['correct_side']), trial['dec_t0']))
        assert rewards[-1] == m.R_CORRECT and status['correct'] is True
        _, rewards, status, _ = rollout(
            rng, trial,
            fixate_then(action_for(other(trial['correct_side'])), trial['dec_t0']))
        assert rewards[-1] == m.R_INCORRECT and status['correct'] is False

    p_left = np.mean([t['correct_side'] == 'L' for t in ties])
    assert abs(p_left - 0.5) < 0.12, p_left
    print('  ties: P(correct_side == L) = {:.3f} over {} tie trials'.format(p_left, len(ties)))

    check('rollouts')

#/////////////////////////////////////////////////////////////////////////////////////////
# 5. Forced conditions
#/////////////////////////////////////////////////////////////////////////////////////////

def test_forced_conditions():
    rng     = np.random.RandomState(3)
    forced  = {
        'context':         'freq',
        'left_right_loc':  -1,
        'left_right_freq': 1,
        'strength_loc':    0.25,
        'strength_freq':   0.45
        }
    trial = m.get_condition(rng, DT, forced)

    for k, v in forced.items():
        assert trial[k] == v, (k, trial[k], v)
    assert trial['lr_ratio'] == 0.75, trial['lr_ratio']
    assert trial['hl_ratio'] == 0.45, trial['hl_ratio']
    assert trial['integral'] == trial['integral_freq']

    check('forced conditions')

#/////////////////////////////////////////////////////////////////////////////////////////
# 6. Feasibility and speed
#/////////////////////////////////////////////////////////////////////////////////////////

def test_feasibility(n_trials=4000):
    rng    = np.random.RandomState(4)
    trials = [m.get_condition(rng, DT) for _ in range(n_trials)]

    integrals = np.array([t['integral'] for t in trials], dtype=float)
    print('  |integral|: mean {:.1f}, median {:.1f}, P(=0) {:.3f}'.format(
        np.abs(integrals).mean(), np.median(np.abs(integrals)),
        np.mean(integrals == 0)))

    print('  {:>6} {:>10}'.format('sigma', 'P(correct)'))
    for sigma in [2, 5, 8]:
        noisy   = integrals + rng.normal(scale=sigma, size=len(integrals))
        correct = []
        for t, x in zip(trials, noisy):
            side = 'L' if x > 0 else 'R'
            correct.append(side == t['correct_side'])
        p = np.mean(correct)
        print('  {:>6} {:>10.3f}'.format(sigma, p))
        if sigma == 8:
            # The termination criterion is p_correct >= 0.85, so an integrator carrying
            # 8 pulses of internal noise must still clear it with room to spare.
            assert p >= 0.87, 'noisy ideal observer only reaches {}'.format(p)

    # get_step must stay cheap: it runs Tmax times per trial in pure Python.
    trial = m.get_condition(rng, DT)
    n     = 200
    start = time.time()
    for _ in range(n):
        rollout(rng, trial, always_fixate)
    per_step = (time.time() - start)/(n*Tmax())
    print('  get_step: {:.1f} us/step'.format(per_step*1e6))
    assert per_step < 20e-6, '{} us/step'.format(per_step*1e6)

    check('feasibility and speed')

#/////////////////////////////////////////////////////////////////////////////////////////
# 7. End-to-end smoke test (imports Theano)
#/////////////////////////////////////////////////////////////////////////////////////////

def test_end_to_end():
    from pyrl.model import Model

    model  = Model(MODELFILE)
    config = dict(model.config)
    config.update({'seed': 3, 'policy_seed': 4, 'baseline_seed': 5})
    pg = model.get_pg(config, 3)

    assert pg.Tmax == 331, pg.Tmax
    assert pg.Nin  == 4,   pg.Nin
    assert pg.Nout == 3,   pg.Nout
    assert pg.R_TERMINAL == -1, pg.R_TERMINAL
    assert pg.baseline_net.config['bout'] == -1, pg.baseline_net.config['bout']

    trials = [pg.task.get_condition(pg.rng, pg.dt) for _ in range(8)]
    rvals  = pg.run_trials(trials, progress_bar=False)
    U, M   = rvals[0], rvals[7]
    assert U.shape == (331, 8, 4), U.shape

    lengths = M.sum(axis=0)
    print('  untrained trial lengths: {}'.format(list(lengths)))
    print('  mean reward: {:.3f}'.format(np.sum(rvals[6]*M)/len(trials)))

    check('end-to-end smoke test')

#/////////////////////////////////////////////////////////////////////////////////////////
# 8. Checkpoint round-trip (imports Theano, needs a trained run)
#/////////////////////////////////////////////////////////////////////////////////////////

SAVEFILE = os.path.join(REPO, 'examples', 'work', 'data', 'pagan', 'pagan.pkl')

def test_checkpoints():
    from pyrl.model import Model
    from pyrl       import utils

    if not os.path.isfile(SAVEFILE):
        print('  {} not found, skipping.'.format(SAVEFILE))
        return

    model = Model(MODELFILE)
    pg    = model.get_pg(SAVEFILE, 1)

    iters = pg.list_checkpoints()
    print('  checkpoints: {}'.format(iters))
    assert len(iters) > 1, iters

    def assert_params_match(actual, expected, label):
        assert set(actual) == set(expected), (sorted(actual), sorted(expected))
        for k in expected:
            assert np.allclose(actual[k], expected[k], atol=1e-6), \
                '{}: parameter {} does not match'.format(label, k)

    # A checkpoint's parameters must land in the live networks, both of them.
    ckpt = pg.load_checkpoint(iters[0])
    assert_params_match(pg.policy_net.get_values(),   ckpt['policy_params'],   'policy')
    assert_params_match(pg.baseline_net.get_values(), ckpt['baseline_params'], 'baseline')

    # Loading the best iteration must reproduce the savefile's best parameters, which is
    # also what a fresh `load='best'` PolicyGradient would hold.
    save = utils.load(SAVEFILE)
    assert save['best_iter'] in iters, (save['best_iter'], iters)
    pg.load_checkpoint(save['best_iter'])
    assert_params_match(pg.policy_net.get_values(), save['best_policy_params'], 'best')

    # Swapping parameters must not trigger a Theano recompile: the second round trip runs
    # against already-compiled functions, so it cannot be much slower than the first.
    times = []
    for iter_ in iters[:2]:
        start = time.time()
        pg.load_checkpoint(iter_)
        pg.run_trials([pg.task.get_condition(pg.rng, pg.dt) for _ in range(8)],
                      progress_bar=False)
        times.append(time.time() - start)
    print('  load + 8 trials: {:.2f} s, {:.2f} s'.format(*times))
    assert times[1] < max(2*times[0], 5), times

    check('checkpoint round-trip')

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    test_module_contract()
    test_epochs()
    test_pulse_statistics()
    test_rollouts()
    test_forced_conditions()
    test_feasibility()
    if '--no-theano' not in sys.argv:
        test_end_to_end()
        test_checkpoints()
    print('\nAll checks passed.')
