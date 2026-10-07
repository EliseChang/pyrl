#!/usr/bin/env python
"""
100-weight-seed sweep for the pagan_hist task under Gaussian init, ungated, at a single
fixed training seed.

Uses the RNN hyperparameters validated on the pagan (Nin=4) `pagan_nullaction_gaussianinit`
run - Wrec_init='gaussian', gates=False, f_hidden='tanh', tau=100, rho=baseline_rho=1.15,
var_rec=0, L2_x=0.001 - carried over unchanged to pagan_hist (Nin=8), with the single change
p0=0.1 -> 0.5 (recurrent connection probability). Unlike sweep_hist100.py, BASE_CONFIG here is
complete: it pins gates=False, f_hidden=tanh and Wrec_init=gaussian explicitly so the runs are
ungated (sweep_hist100.py omitted these and inherited the framework default gates=True).

Nin=8 is set automatically from pagan_hist.inputs; with Wrec_init=gaussian the input weights
are drawn IID N(0, 1/Nin), so Win std = 1/sqrt(8) ~ 0.354 (vs 1/sqrt(4)=0.5 in the pagan
source) - an automatic consequence of the larger channel count.

Early stopping (terminate() in pagan_hist.py, P(decision)>=0.99 and P(correct)>=0.95) is on,
so seeds finish at different iterations and checkpoints are not aligned across seeds.

Usage (from the repository root, under the Python 2.7 environment):

    python examples/sweep_histgauss100.py --dry-run        # print the commands
    python examples/sweep_histgauss100.py --jobs 3         # train the grid

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
MODEL    = os.path.join(HERE, 'models', 'pagan_hist.py')
LOGS     = os.path.join(REPO, 'logs')

WEIGHT_SEEDS = list(range(1, 101))
STIM_SEED    = 1
ACTION_SEED  = 1

# Held fixed across every cell. Complete (unlike sweep_hist100.py's BASE_CONFIG): gates,
# f_hidden and Wrec_init are pinned so the runs are ungated Gaussian-init.
BASE_CONFIG = [
    'Wrec_init=gaussian',
    'gates=False',
    'f_hidden=tanh',
    'tau=100',
    'rho=1.15',
    'baseline_rho=1.15',
    'p0=0.5',
    'baseline_p0=1',
    'var_rec=0',
    'baseline_var_rec=0',
    'L2_x=0.001',
    'lr=0.004',
    'baseline_lr=0.004',
    'beta_entropy=0',
    'Win=1',
    'max_iter=500',
    'checkfreq=10',
    ]

def cell_name(w):
    return 'histgauss100_w{:03d}'.format(w)

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

def savefile_for(w):
    name = 'pagan_hist_' + cell_name(w)

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
    p.add_argument('--jobs', type=int, default=3, help="cells to run concurrently")
    p.add_argument('--force', action='store_true',
                   help="re-run cells whose savefile already exists")
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    if not os.path.isdir(LOGS):
        os.makedirs(LOGS)

    jobs = []
    for w in grid():
        name   = 'pagan_hist_' + cell_name(w)
        exists = os.path.isfile(savefile_for(w))

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
