import numpy as np

required = ['inputs', 'actions', 'tmax', 'n_gradient', 'n_validation']
default  = {
    'Performance':           None,
    'N':                     100,
    'p0':                    0.1,
    'baseline_N':            100,
    'baseline_p0':           1,
    'lr':                    0.004,
    'baseline_lr':           0.004,
    'max_iter':              100000,
    'fix':                   [],
    'baseline_fix':          [],
    'target_reward':         np.inf,
    'mode':                  'episodic',
    'network_type':          'gru',
    'baseline_network_type': 'gru',
    'f_hidden':              'relu',
    'gates':                 True,
    'Wrec_init':             'gamma',
    'Wout':                  0,
    'baseline_Wout':         0,
    'R_ABORTED':             -1,
    'R_TERMINAL':            None,
    'abort_on_last_t':       True,
    'checkfreq':             50,
    'dt':                    10,
    'tau':                   100,
    'tau_reward':            np.inf,
    'var_rec':               0.01,
    'baseline_var_rec':      0.01,
    'L2_r':                  0,
    'baseline_L2_r':         0,
    'L2_x':                  0,
    'beta_entropy':          0,
    'Win':                   1,
    'baseline_Win':          1,
    'bout':                  0,
    'baseline_bout':         None,
    'Win_mask':              None,
    'baseline_Win_mask':     None,
    'rho':                   2,
    'baseline_rho':          2,
    'L1_Wrec':               0,
    'L2_Wrec':               0,
    'policy_seed':           1,
    'baseline_seed':         2,

    # Independently seeded RNG streams. `None` means "fall back to the single --seed",
    # which reproduces the legacy derivation. See `pyrl.model.derive_seeds`.
    'weight_seed':           None,
    'stim_seed':             None,
    'action_seed':           None
    }

def derive_seeds(seed, config):
    """
    Resolve the three independent RNG streams from `seed` and any explicitly set
    `weight_seed`/`stim_seed`/`action_seed`.

    (1) weight init   -> policy_seed, baseline_seed  (both networks)
    (2) stimulus      -> seed                        (trial/condition generation)
    (3) action sample -> action_rng_seed, and the recurrent-noise stream

    Streams 2 and 3 together are the "training seed" but stay separately controllable.
    With no stream seed set they all fall back to `seed`, reproducing the legacy
    3s / 3s+1 / 3s+2 derivation for weight init and trials. The action stream is offset
    out of the 3k, 3k+1, 3k+2 block so that it can never coincide with another stream.

    """
    w = seed if config.get('weight_seed') is None else config['weight_seed']
    s = seed if config.get('stim_seed')   is None else config['stim_seed']
    a = seed if config.get('action_seed') is None else config['action_seed']

    return {'weight_seed':     w,
            'stim_seed':       s,
            'action_seed':     a,
            'policy_seed':     3*w + 1,
            'baseline_seed':   3*w + 2,
            'seed':            3*s,
            'action_rng_seed': 3*a + 1000000,
            'noise_rng_seed':  3*a + 2000000}
