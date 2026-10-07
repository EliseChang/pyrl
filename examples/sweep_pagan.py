#!/usr/bin/env python
"""
Run the vanilla-RNN hyperparameter grid for the pagan task.

Every cell is a non-gated, non-leaky RNN by construction: `gates=False` clamps the update
and reset gates to 1, and `tau=dt` makes `alpha = dt/tau = 1`. The grid crosses

    activation  relu | tanh
    rho         0.8 (contracting) | 1.0 (critical)
    var_rec     0.01 (default) | 0 (no recurrent noise)

over three seeds, for 24 runs. `rho` is applied to `Wrec` alone once the gates are off, so
it is exactly the spectral radius of the recurrent connectivity. The value network shares
the activation and the gates-off construction but is pinned at `baseline_rho=0.8` in every
cell, so that a cell never fails because its critic was unstable.

Usage (from the repository root, under the Python 2.7 environment):

    python examples/sweep_pagan.py --dry-run        # print the commands
    python examples/sweep_pagan.py --jobs 4         # train the grid
    python examples/sweep_pagan.py --jobs 4 --analyze

`--analyze` runs the best-parameter analysis pipeline for each cell instead of training.
Cells whose savefile already exists are skipped unless `--force` is given.

"""
from __future__ import absolute_import, division, print_function

import argparse
import itertools
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool

HERE  = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(HERE)
DO    = os.path.join(HERE, 'do.py')
MODEL = os.path.join(HERE, 'models', 'pagan.py')
LOGS  = os.path.join(REPO, 'logs')

# The grid. `dt` stays at its default of 10, so tau=10 gives alpha=1.
ACTIVATIONS = ['relu', 'tanh']
RHOS        = [0.8, 1.0]
VAR_RECS    = [0.01, 0.0]
SEEDS       = [100, 200, 300]

MAX_ITER     = 500
BASELINE_RHO = 0.8

def cell_name(f_hidden, rho, var_rec, seed):
    return '{}_rho{:02d}_{}_s{}'.format(f_hidden, int(round(10*rho)),
                                        'noise' if var_rec > 0 else 'nonoise', seed)

def grid():
    for f_hidden, rho, var_rec, seed in itertools.product(ACTIVATIONS, RHOS,
                                                          VAR_RECS, SEEDS):
        yield f_hidden, rho, var_rec, seed

def train_command(f_hidden, rho, var_rec, seed):
    suffix = '_' + cell_name(f_hidden, rho, var_rec, seed)

    return [sys.executable, '-u', DO,
            '--suffix', suffix,
            '--seed', str(seed),
            '--config', 'gates=False',
            '--config', 'tau=10',
            '--config', 'f_hidden={}'.format(f_hidden),
            '--config', 'rho={}'.format(rho),
            '--config', 'baseline_rho={}'.format(BASELINE_RHO),
            '--config', 'var_rec={}'.format(var_rec),
            '--config', 'baseline_var_rec={}'.format(var_rec),
            '--config', 'max_iter={}'.format(MAX_ITER),
            MODEL, 'train']

def analyze_commands(f_hidden, rho, var_rec, seed):
    suffix   = '_' + cell_name(f_hidden, rho, var_rec, seed)
    analysis = os.path.join('analysis', 'pagan')

    # Rollouts first, then the figures that consume them. `--dt-save 50` keeps the
    # activity file to a manageable size; do NOT pass `--dt`, which would change the
    # integration step without changing tau and so break alpha = 1.
    cmds = [[sys.executable, '-u', DO, '--suffix', suffix, MODEL,
             'run', analysis, 'trials-b'],
            [sys.executable, '-u', DO, '--suffix', suffix, '--dt-save', '50', MODEL,
             'run', analysis, 'trials-a']]
    for figure in ['performance', 'psychometric', 'statespace', 'trial']:
        cmds.append([sys.executable, '-u', DO, '--suffix', suffix, MODEL,
                     'run', analysis, figure])

    return cmds

def savefile_for(f_hidden, rho, var_rec, seed):
    name = 'pagan_' + cell_name(f_hidden, rho, var_rec, seed)

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
    p.add_argument('--jobs', type=int, default=4, help="cells to run concurrently")
    p.add_argument('--analyze', action='store_true',
                   help="run the analysis pipeline instead of training")
    p.add_argument('--force', action='store_true',
                   help="re-run cells whose savefile already exists")
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    if not os.path.isdir(LOGS):
        os.makedirs(LOGS)

    jobs = []
    for cell in grid():
        name     = 'pagan_' + cell_name(*cell)
        savefile = savefile_for(*cell)
        exists   = os.path.isfile(savefile)

        if a.analyze:
            # Analysis needs a trained model, so a missing savefile is an error here,
            # not something to skip past silently.
            if not exists:
                print('[skip] {}: not trained yet'.format(name))
                continue
            cmds = analyze_commands(*cell)
        else:
            if exists and not a.force:
                print('[skip] {}: already trained'.format(name))
                continue
            cmds = [train_command(*cell)]

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
