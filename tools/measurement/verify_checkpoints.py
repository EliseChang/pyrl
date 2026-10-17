#!/usr/bin/env python
"""
Verify the checkpoints of one or more run folders before measuring them.

Run under the Python 2.7 environment (the checkpoints are Python-2 pickles referencing the
task module, so examples/models must be importable):

  python tools/measurement/verify_checkpoints.py pagan_hist_tscost_g06tl55cho_w002 ...
  python tools/measurement/verify_checkpoints.py --glob 'pagan_hist_tscost_g06tl55cho_w*'

Why this exists. On 16-17 Oct 2026 two successive transfers of the tl55cho runs from the Mac
arrived cross-contaminated: a `checkpoints/` directory nominally belonging to seed 2 contained
checkpoints whose stored `config['weight_seed']` was 3. The first time it was an rsync mistake
(trailing slashes flattened both runs into one directory, and identical filenames overwrote each
other); the second time 13 files were still seed 3, reproducibly, so the source itself was
contaminated. Nothing downstream notices: the rollout happily loads whatever weights are in the
file, and the resulting record is a chimera of two networks with no outward sign. The only
defence is to check the seed stored inside every checkpoint against the seed in the run name,
which is what this does.

Checks, per run:
  - every iter_*.pkl is readable
  - d['iter'] matches the iteration in the filename
  - d['config']['weight_seed'] matches the _wNNN in the run folder name
  - the iteration series has no gaps (reported, not fatal: a run can legitimately be measured
    with gaps, it just yields fewer rollout points)

Exit status is 0 only if every run is free of unreadable, mismatched and wrong-seed files.
Gaps alone do not fail it.

  --quarantine DIR   move offending files there instead of only reporting them. The run is then
                     clean but has gaps; measure_run.sh will use however many checkpoints remain.
"""
from __future__ import division, print_function

import argparse
import collections
import glob as globmod
import os
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('PYRL_REPO', os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'examples', 'models'))


def seed_from_name(run):
    """The weight seed a run folder claims, from its trailing _wNNN."""
    i = run.rfind('_w')
    if i < 0:
        return None
    tail = run[i + 2:]
    return int(tail) if tail.isdigit() else None


def verify(run, data, quarantine=None):
    cd = os.path.join(data, run, 'checkpoints')
    expect = seed_from_name(run)
    out = {'run': run, 'expect': expect, 'n': 0, 'unreadable': [], 'bad_seed': [],
           'bad_iter': [], 'gaps': [], 'ok': False}
    if expect is None:
        print("  %s: cannot read a seed from the run name -- skipping" % run)
        return out
    if not os.path.isdir(cd):
        print("  %s: no checkpoints/ directory" % run)
        return out

    files = sorted(f for f in os.listdir(cd)
                   if f.startswith('iter_') and f.endswith('.pkl'))
    out['n'] = len(files)
    iters = []
    for f in files:
        p = os.path.join(cd, f)
        want_iter = int(f[5:-4])
        iters.append(want_iter)
        try:
            d = pickle.load(open(p, 'rb'))
        except Exception as e:
            out['unreadable'].append((f, str(e)[:70]))
            continue
        got = d.get('config', {}).get('weight_seed')
        if got != expect:
            out['bad_seed'].append((f, got))
        if d.get('iter') != want_iter:
            out['bad_iter'].append((f, d.get('iter')))

    # gaps, against the modal step of the series
    if len(iters) > 2:
        diffs = collections.Counter(b - a for a, b in zip(iters, iters[1:]))
        step = diffs.most_common(1)[0][0]
        if step > 0:
            full = set(range(min(iters), max(iters) + 1, step))
            out['gaps'] = sorted(full - set(iters))

    dirty = out['unreadable'] or out['bad_seed'] or out['bad_iter']
    out['ok'] = not dirty

    print("  %s" % run)
    print("     expect weight_seed=%d | %d checkpoints | iters %s..%s"
          % (expect, out['n'], min(iters) if iters else '-', max(iters) if iters else '-'))
    if out['bad_seed']:
        seeds = collections.Counter(s for _, s in out['bad_seed'])
        print("     WRONG SEED: %d files %s" % (len(out['bad_seed']), dict(seeds)))
        for f, s in out['bad_seed'][:8]:
            print("        %s -> seed %s" % (f, s))
        if len(out['bad_seed']) > 8:
            print("        ... and %d more" % (len(out['bad_seed']) - 8))
    if out['unreadable']:
        print("     UNREADABLE: %d %s" % (len(out['unreadable']), out['unreadable'][:3]))
    if out['bad_iter']:
        print("     ITER MISMATCH: %d %s" % (len(out['bad_iter']), out['bad_iter'][:3]))
    if out['gaps']:
        print("     gaps (not fatal): %d missing iterations %s%s"
              % (len(out['gaps']), out['gaps'][:12],
                 ' ...' if len(out['gaps']) > 12 else ''))
    print("     VERDICT: %s" % ("CLEAN" if out['ok'] else "DIRTY -- do not measure as is"))

    if quarantine and dirty:
        qd = os.path.join(quarantine, run)
        if not os.path.isdir(qd):
            os.makedirs(qd)
        moved = 0
        for f, _ in out['bad_seed'] + out['bad_iter'] + out['unreadable']:
            src = os.path.join(cd, f)
            if os.path.exists(src):
                os.rename(src, os.path.join(qd, f))
                moved += 1
        print("     quarantined %d files -> %s" % (moved, qd))
        out['ok'] = True
        out['quarantined'] = moved
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--glob', default=None,
                    help="shell glob over run folder names, e.g. 'pagan_hist_tscost_g06tl55cho_w*'")
    ap.add_argument('--data', default=os.path.join(REPO, 'examples', 'work', 'data'))
    ap.add_argument('--quarantine', default=None,
                    help="move offending files here instead of only reporting them")
    a = ap.parse_args()

    runs = list(a.runs)
    if a.glob:
        runs += [os.path.basename(p) for p in sorted(globmod.glob(os.path.join(a.data, a.glob)))
                 if os.path.isdir(p)]
    runs = sorted(set(runs))
    if not runs:
        print("no run folders given (use names or --glob)")
        return 2

    print("verify_checkpoints: %d run(s) under %s" % (len(runs), a.data))
    results = [verify(r, a.data, a.quarantine) for r in runs]
    bad = [r['run'] for r in results if not r['ok']]
    print()
    print("%d of %d clean" % (len(results) - len(bad), len(results)))
    if bad:
        print("DIRTY: %s" % ', '.join(bad))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
