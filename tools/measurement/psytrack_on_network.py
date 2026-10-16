#!/usr/bin/env python3
"""PsyTrack on a reward-trained network, across its whole training history.

Why. The rat story in the talk is told with PsyTrack (Roy et al. 2021): smooth, trial-by-trial
weights of choice on evidence and history, with the weights allowed to jump between sessions.
To compare a network with a rat on the same axes, fit the same model to the network.

How. Elise's pyrl saves, at every checkpoint, a rollout of N condition-balanced trials under the
frozen weights of that checkpoint (`trials-b N iters=all`), rolled out sequentially so that the
previous-outcome input is real. Each checkpoint is treated as one session (with checkfreq 5 and
72 trials per update, one checkpoint stands for 360 training trials), the sessions are
concatenated in training order, and PsyTrack is fitted with a per-trial random walk (sigma) and
a per-session jump (sigDay), exactly as rat_strategy_discovery/baselines/psytrack_baseline.py
does for the rats. Regressors follow that file: bias, rel (evidence of the cued feature, signed,
+ = right), irr (evidence of the other feature), histChoice (previous choice, +1 right), wsls
(previous choice x previous reward, +1 win). Continuous regressors are z-scored over the whole
record. Trials without a choice are dropped.

Input: the rollout folder of one run,
    $SCRATCH/work/pyrl/examples/<run>/checkpoints/iter_NNNNNN/trials_behavior.pkl
(Python 2 pickles: [trials, A, R, M, perf]; loaded with latin1 and a tolerant unpickler, since
`perf` is a pyrl class we do not need.)

Output: <out>/<run>_psytrack.npz (wMode, hyp, names, session index per trial, trial index) and a
figure <out>/<run>_psytrack.png in the style of the rat PsyTrack plots.

Usage, on rebel with the rat_strategy_discovery venv (psytrack is installed there):
    python psytrack_on_network.py --run pagan_hist_curriculum_curr_w001 [--run ...] \
        --scratch /mnt/big/pyrl_work/scratch --out /mnt/big/pyrl_work/psytrack

Split mode, for farming the fits out to other machines (the rollouts are 1.8 GB per run, the
record a few MB, and the fit is one to five CPU-hours):
    python psytrack_on_network.py --run <run> --extract --out <dir>     # rebel: writes <run>_record.npz
    python psytrack_on_network.py --record <dir>/<run>_record.npz --out <dir>   # anywhere: fits it
Fits are single-process numpy; cap BLAS threads (OMP_NUM_THREADS=2) when running many at once.
"""
import argparse
import glob
import os
import pickle
import re

import numpy as np

try:
    import psytrack as psy
except ImportError:  # the figure-only path does not need psytrack
    psy = None


class _Tolerant(pickle.Unpickler):
    """Unpickle Python 2 pyrl rollouts without pyrl: unknown classes become plain objects."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (object,), {"__setstate__": lambda self, s: self.__dict__.update(s)})


def load_rollout(path):
    with open(path, "rb") as f:
        trials, A, R, M, perf = _Tolerant(f, encoding="latin1").load()
    return trials, np.asarray(A), np.asarray(R), np.asarray(M)


def choice_of(trial_actions, trial_rewards, mask):
    """The choice of one rollout trial. Actions are one-hot (Tmax, 3): fixate, left, right. A
    trial's valid steps are 0..T-1 with T = mask.sum(); the trial ends at the step of the choice,
    so the last valid action is the choice (fixate there means it never decided). Returns +1
    right, -1 left, 0 none, and the terminal reward."""
    T = int(mask.sum())
    a = int(np.argmax(trial_actions[T - 1]))
    ch = -1 if a == 1 else (+1 if a == 2 else 0)
    return ch, float(trial_rewards[:T].sum())


def build_record(run_dir):
    """Concatenate the per-checkpoint rollouts into one PsyTrack record."""
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "iter_*", "trials_behavior.pkl")))
    if not ckpts:
        raise SystemExit("no rollouts under %s" % run_dir)
    rel, irr, prev_choice, wsls, y, session, iters = [], [], [], [], [], [], []
    ctx, e_loc_all, e_frq_all, rew = [], [], [], []      # extras for the explorer
    n_total, n_choice, ck_iters = [], [], []             # opportunities and attempts per checkpoint
    for si, path in enumerate(ckpts):
        it = int(re.search(r"iter_(\d+)", path).group(1))
        trials, A, R, M = load_rollout(path)
        # A, R, M are (Tmax, n_trials); trials is a list of dicts
        n_total.append(len(trials)); ck_iters.append(it); n_choice.append(0)
        for n, tr in enumerate(trials):
            ch, rw = choice_of(A[:, n], R[:, n], M[:, n])
            # history as the network saw it: pyrl threads (choice, reward) of the previous
            # trial onto the trial dict (None on the first trial and after a no-decision trial)
            prev = tr.get("prev_outcome")
            if prev is None:
                last_choice, last_reward = 0, 0.0
            else:
                last_choice, last_reward = (+1 if prev[0] == "R" else -1), float(prev[1])
            # evidence: positive integral means LEFT in pagan.py; flip so + = right
            e_loc, e_frq = -float(tr["integral_loc"]), -float(tr["integral_freq"])
            r_, i_ = (e_loc, e_frq) if tr["context"] == "loc" else (e_frq, e_loc)
            if ch != 0:
                n_choice[-1] += 1
                ctx.append(0 if tr["context"] == "loc" else 1)
                e_loc_all.append(e_loc); e_frq_all.append(e_frq); rew.append(rw)
                rel.append(r_); irr.append(i_); y.append(1 if ch > 0 else 0)
                prev_choice.append(last_choice)
                wsls.append(last_choice * (1 if last_reward > 0 else -1) if last_choice != 0 else 0)
                session.append(si); iters.append(it)
    rel, irr = np.array(rel), np.array(irr)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    X = {"bias": np.ones(len(y)), "rel": z(rel), "irr": z(irr),
         "histChoice": np.array(prev_choice, float), "wsls": np.array(wsls, float)}
    # raw per-trial fields, not regressors: kept on the function so --extract can save them
    build_record.extras = {"context": np.array(ctx, np.int8), "e_loc": np.array(e_loc_all),
                           "e_frq": np.array(e_frq_all), "reward": np.array(rew), "rel_raw": rel, "irr_raw": irr,
                           "ck_iters": np.array(ck_iters), "ck_n_total": np.array(n_total), "ck_n_choice": np.array(n_choice)}
    return X, np.array(y), np.array(session), np.array(iters)


def fit(X, y, session):
    names = sorted(X.keys())
    inputs = {k: X[k][:, None] for k in names}
    dayLength = np.bincount(session)
    dat = {"inputs": inputs, "y": y, "dayLength": dayLength[dayLength > 0]}
    weights = {k: 1 for k in names}
    hyper = {"sigInit": 2 ** 4.0, "sigma": [2 ** -4.0] * len(names), "sigDay": [2 ** -4.0] * len(names)}
    hyp, evd, wMode, hess = psy.hyperOpt(dat, hyper, weights, ["sigma", "sigDay"])
    # +/- 1 std credible interval of every weight at every trial, from the hessian; this is
    # the band drawn around the rats' trajectories (psytrack_analyses/code/build_rat_html.py)
    W_std = psy.getCredibleInterval(hess)
    return names, wMode, hyp, evd, W_std


# The rats' colours (psytrack_analyses/code/build_rat_html.py WEIGHT_CLRS), so that a network
# figure and a rat figure read the same way side by side.
WEIGHT_CLRS = {"bias": "tab:blue", "rel": "tab:red", "irr": "tab:green",
               "histChoice": "tab:orange", "wsls": "tab:purple"}
# Pre-training phases of pagan_hist_curriculum, in checkpoints (checkfreq 5): the rollouts of
# checkpoints 0 (iter 0) and 1 (iter 5) fall in 'fixed', 2-4 in 'blocks', 5-6 in 'random'.
# Shades follow the rat report's phase shades.
CURRICULUM_PHASES = [("fixed", 0, 2, "#e8f0fe"), ("blocks", 2, 5, "#e9f7ef"),
                     ("random", 5, 7, "#fef3e7")]


def plot(run, names, wMode, session, iters, out, W_std=None):
    """One network, all weights on one axis across its whole training record, drawn exactly
    like the rats' full-trajectory figure: white background, the rats' colours, a +/- 1 std
    band when the fit saved one, a dashed zero line, legend along the bottom. The x axis is
    the concatenated rollout trial; every checkpoint boundary is a faint vertical line (the
    networks' 'sessions'), and for a curriculum run the pre-training phases are shaded and
    separated by red dashed lines, like the rats' stage lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    x = np.arange(len(session))
    fig, ax = plt.subplots(figsize=(13, 6))
    bounds = np.where(np.diff(session) != 0)[0] + 1          # first trial of each checkpoint
    # checkpoint boundaries: faint, like session boundaries in the rat plots
    for b in bounds:
        ax.axvline(b, color="#dddddd", lw=0.4, zorder=0)
    if "curriculum_curr" in run:
        starts = np.concatenate([[0], bounds, [len(x)]])
        for name, c0, c1, shade in CURRICULUM_PHASES:
            if c1 <= len(starts) - 1:
                ax.axvspan(starts[c0], starts[c1], color=shade, zorder=0)
                ax.text((starts[c0] + starts[c1]) / 2, 0.99, name, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=8, fontweight="bold", color="#444", rotation=90)
                ax.axvline(starts[c1], color="red", ls="--", lw=0.8, zorder=1)
        ax.text(starts[7] + 0.01 * len(x) if len(starts) > 7 else 0, 0.99, "task",
                transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8,
                fontweight="bold", color="#444")
    for k, name in enumerate(names):
        w = wMode[k]
        ax.plot(x, w, color=WEIGHT_CLRS.get(name, "k"), label=name, lw=1.3)
        if W_std is not None:
            e = W_std[k]
            ax.fill_between(x, w - e, w + e, color=WEIGHT_CLRS.get(name, "k"), alpha=0.25, lw=0)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, len(x))
    # a second row of ticks in training trials, so the x axis can be read against the rats'
    ticks = bounds[::10]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%d\n(%d)" % (t, iters[t] * 72) for t in ticks], fontsize=8)
    ax.set_xlabel("Trial of the concatenated checkpoint rollouts   (in brackets: training trials seen)")
    ax.set_ylabel("Logistic weight")
    ax.legend(loc="lower center", ncol=len(names), fontsize=8, framealpha=0.9)
    fig.suptitle("%s: %d rollout trials over %d checkpoints" % (run, len(x), session.max() + 1),
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--record", nargs="*", default=[], help="fit these <run>_record.npz files instead of rollouts")
    ap.add_argument("--extract", action="store_true", help="only build the record and save it as <run>_record.npz")
    ap.add_argument("--replot", nargs="*", default=[], help="redraw these existing <run>_psytrack.npz files")
    ap.add_argument("--scratch", default="/mnt/big/pyrl_work/scratch")
    ap.add_argument("--out", default="/mnt/big/pyrl_work/psytrack")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for npz in a.replot:
        z = np.load(npz)
        run = os.path.basename(npz).replace("_psytrack.npz", "")
        plot(run, [str(n) for n in z["names"]], z["wMode"], z["session"], z["iters"],
             os.path.join(a.out, run + "_psytrack.png"), W_std=z["W_std"] if "W_std" in z.files else None)
        print("replotted", run, "with band" if "W_std" in z.files else "(no band saved in this fit)")
    jobs = [("run", r) for r in a.run] + [("record", r) for r in a.record]
    for kind, item in jobs:
        if kind == "run":
            run = item
            run_dir = os.path.join(a.scratch, "work", "pyrl", "examples", run)
            X, y, session, iters = build_record(run_dir)
            if a.extract:
                np.savez(os.path.join(a.out, run + "_record.npz"), y=y, session=session, iters=iters,
                         **{"X_" + k: v for k, v in X.items()},
                         **{"raw_" + k: v for k, v in build_record.extras.items()})
                print(run, len(y), "choices over", session.max() + 1, "checkpoints: record saved")
                continue
        else:
            z = np.load(item)
            run = os.path.basename(item).replace("_record.npz", "")
            y, session, iters = z["y"], z["session"], z["iters"]
            X = {k[2:]: z[k] for k in z.files if k.startswith("X_")}
        print(run, len(y), "choices over", session.max() + 1, "checkpoints")
        names, wMode, hyp, evd, W_std = fit(X, y, session)
        np.savez(os.path.join(a.out, run + "_psytrack.npz"), wMode=wMode, W_std=W_std, names=np.array(names),
                 session=session, iters=iters, y=y, sigma=hyp["sigma"], sigDay=hyp["sigDay"], evd=evd)
        plot(run, names, wMode, session, iters, os.path.join(a.out, run + "_psytrack.png"), W_std=W_std)
        final = {n: round(float(wMode[k, -1]), 2) for k, n in enumerate(names)}
        print("  final weights:", final)


if __name__ == "__main__":
    main()
