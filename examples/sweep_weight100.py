#!/usr/bin/env python
"""
100-weight-seed sweep for the pagan task, at a single fixed training seed.

Unlike sweep_seeds.py (10 weight seeds x 4 training seeds, final checkpoint only), this
sweep holds `stim_seed`/`action_seed` fixed and varies only `weight_seed`, training every
seed for the full `max_iter` budget (no early stopping -- see `terminate()` in
examples/models/pagan.py) with checkpoints saved every `checkfreq` iterations. The
`--analyze` pass rolls out lightweight in-memory metrics (no trials_behavior.pkl written)
for every checkpoint of every seed via the `checkpointsweep` analysis action, writing one
small `checkpoint_metrics.pkl` per seed.

Usage (from the repository root, under the Python 2.7 environment):

    python examples/sweep_weight100.py --dry-run        # print the commands
    python examples/sweep_weight100.py --jobs 8         # train the grid
    python examples/sweep_weight100.py --jobs 8 --analyze

"""
from __future__ import absolute_import, division, print_function

import argparse
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool

HERE     = os.path.dirname(os.path.abspath(__file__))
REPO     = os.path.dirname(HERE)
DO       = os.path.join(HERE, 'do.py')
MODEL    = os.path.join(HERE, 'models', 'pagan.py')
ANALYSIS = os.path.join(HERE, 'analysis', 'pagan.py')
LOGS     = os.path.join(REPO, 'logs')

WEIGHT_SEEDS = list(range(1, 101))
STIM_SEED    = 1
ACTION_SEED  = 1
TRIALS_PER_CONDITION = 50

# Held fixed across every cell. `dt` stays at its default of 10, so tau=10 gives alpha=1.
BASE_CONFIG = [
    'gates=False',
    'f_hidden=tanh',
    'Wrec_init=gaussian',
    'tau=10',
    'rho=0.9',
    'baseline_rho=0.9',
    'var_rec=0',
    'baseline_var_rec=0',
    'L2_x=0.001',
    'beta_entropy=0',
    'max_iter=1000',
    'checkfreq=10',
    'p0=0.5',
    ]

def cell_name(w):
    return 'weight100_w{:03d}'.format(w)

def grid():
    return WEIGHT_SEEDS

def train_command(w):
    config = ['weight_seed={}'.format(w),
              'stim_seed={}'.format(STIM_SEED),
              'action_seed={}'.format(ACTION_SEED)] + BASE_CONFIG

    cmd = [sys.executable, '-u', DO, '--suffix', '_' + cell_name(w)]
    for kv in config:
        cmd += ['--config', kv]

    return cmd + [MODEL, 'train']

def analyze_commands(w):
    return [[sys.executable, '-u', DO, '--suffix', '_' + cell_name(w), MODEL,
             'run', ANALYSIS, 'checkpointsweep', str(TRIALS_PER_CONDITION)]]

def savefile_for(w):
    name = 'pagan_' + cell_name(w)

    return os.path.join(HERE, 'work', 'data', name, name + '.pkl')

def run(name, cmds):
    """
    Run a cell's commands in order, teeing everything to logs/<name>.log. Returns
    (name, ok).

    """
    # do.py is run as a script, so sys.path[0] is examples/, not the repository root;
    # `pyrl` is not installed into the environment and has to be found via PYTHONPATH.
    env = dict(os.environ)
    env['PYTHONPATH'] = REPO + os.pathsep + env.get('PYTHONPATH', '')

    logfile = os.path.join(LOGS, name + '.log')
    with open(logfile, 'w') as log:
        for cmd in cmds:
            log.write('$ ' + ' '.join(cmd) + '\n')
            log.flush()
            if subprocess.call(cmd, cwd=REPO, env=env, stdout=log,
                               stderr=subprocess.STDOUT) != 0:
                return name, False

    return name, True

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobs', type=int, default=8, help="cells to run concurrently")
    p.add_argument('--analyze', action='store_true',
                   help="roll out checkpoint metrics instead of training")
    p.add_argument('--force', action='store_true',
                   help="re-run cells whose savefile already exists")
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    if not os.path.isdir(LOGS):
        os.makedirs(LOGS)

    jobs = []
    for w in grid():
        name   = 'pagan_' + cell_name(w)
        exists = os.path.isfile(savefile_for(w))

        if a.analyze:
            if not exists:
                print('[skip] {}: not trained yet'.format(name))
                continue
            cmds = analyze_commands(w)
        else:
            if exists and not a.force:
                print('[skip] {}: already trained'.format(name))
                continue
            cmds = [train_command(w)]

        jobs.append((name, cmds))

    if a.dry_run:
        for name, cmds in jobs:
            for cmd in cmds:
                print(' '.join(cmd))
        print('\n{} cells, {} commands'.format(len(jobs), sum(len(c) for _, c in jobs)))
        return 0

    if not jobs:
        print('Nothing to do.')
        return 0

    print('Running {} cells with {} concurrent jobs.'.format(len(jobs), a.jobs))
    pool    = ThreadPool(a.jobs)
    results = pool.map(lambda job: run(*job), jobs)
    pool.close()
    pool.join()

    print('')
    failed = 0
    for name, ok in sorted(results):
        print('{:6s} {}'.format('ok' if ok else 'FAILED', name))
        failed += not ok
    print('\n{}/{} succeeded. Logs in {}'.format(len(results) - failed, len(results), LOGS))

    return 1 if failed else 0

if __name__ == '__main__':
    sys.exit(main())
