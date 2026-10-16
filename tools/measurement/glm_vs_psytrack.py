#!/usr/bin/env python3
"""PsyTrack versus faster alternatives, on one network's training record.

Why. A PsyTrack fit of one network takes one to five CPU-hours, almost
all of it in hyperOpt: the search over the ten hyperparameters (a per-trial drift sigma and a
per-session jump sigDay for each of five weights), each step of which solves a sparse system
of size 5 x 110,000. For a network that is overkill. A checkpoint is a frozen policy: nothing
changes within its rollout, so the only honest model is "one weight vector per checkpoint",
which is a logistic regression on 1,800 trials and takes milliseconds. The earlier GLM
diverged only on checkpoints whose choices were perfectly separable (a pure side habit); a
small L2 prior removes that and gives bounded weights with standard errors.

Three fits of the same record, timed:
  1. PsyTrack as run (loaded from the saved fit; its time is read from the farm log)
  2. PsyTrack with hyperparameters fixed (one MAP solve, no hyperOpt)
  3. per-checkpoint ridge logistic regression (Newton's method, L2 prior, +/- 1 SE from the
     Hessian), i.e. PsyTrack with sigma = 0 and sigDay = infinity plus a Gaussian prior
and a figure overlaying them, in the rats' format.

Usage (rebel, rat_strategy_discovery venv):
    python glm_vs_psytrack.py --record /mnt/big/pyrl_work/psytrack/records/<run>_record.npz \
        --psytrack /mnt/big/pyrl_work/psytrack/<run>_psytrack.npz --out <png>
"""
import argparse
import os
import time

import numpy as np

WEIGHT_CLRS = {"bias": "tab:blue", "rel": "tab:red", "irr": "tab:green",
               "histChoice": "tab:orange", "wsls": "tab:purple"}


def ridge_logistic(X, y, lam=1.0, iters=50):
    """Logistic regression of y (0/1) on columns of X with a Gaussian prior of precision lam
    on every weight. Newton's method; returns the weights and their standard errors from the
    inverse Hessian at the optimum. With lam > 0 the solution is finite even when the classes
    are perfectly separable, which is exactly the case that broke the unpenalised GLM."""
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + lam * w
        H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(d)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ w))
    H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(d)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return w, se


def per_checkpoint_glm(X, y, session, lam=1.0):
    names = sorted(X.keys())
    M = np.column_stack([X[k] for k in names])
    W, SE = [], []
    for s in np.unique(session):
        m = session == s
        w, se = ridge_logistic(M[m], y[m], lam)
        W.append(w); SE.append(se)
    return names, np.array(W).T, np.array(SE).T          # (n_weights, n_checkpoints)


def psytrack_fixed(X, y, session, sigma=2 ** -4.0, sigDay=2 ** 0.0):
    """One PsyTrack MAP solve with the hyperparameters fixed: no search over them."""
    import psytrack as psy
    names = sorted(X.keys())
    dat = {"inputs": {k: X[k][:, None] for k in names}, "y": y,
           "dayLength": np.bincount(session)[np.bincount(session) > 0]}
    hyper = {"sigInit": 2 ** 4.0, "sigma": [sigma] * len(names), "sigDay": [sigDay] * len(names)}
    weights = {k: 1 for k in names}
    from psytrack.getMAP import getMAP   # psy.getMAP is the submodule, not the function
    wMode, hess, logEvd, llstruct = getMAP(dat, hyper, weights)
    return names, wMode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--psytrack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--no-fixed", action="store_true",
                    help="skip the fixed-hyperparameter PsyTrack solve (45 min per network)")
    a = ap.parse_args()
    z = np.load(a.record)
    y, session, iters = z["y"], z["session"], z["iters"]
    X = {k[2:]: z[k] for k in z.files if k.startswith("X_")}
    run = os.path.basename(a.record).replace("_record.npz", "")

    t0 = time.time()
    names, W, SE = per_checkpoint_glm(X, y, session, a.lam)
    t_glm = time.time() - t0
    print("ridge GLM, %d checkpoints: %.2f s" % (W.shape[1], t_glm))

    t0 = time.time()
    W_fixed, t_fixed = None, None
    try:
        if a.no_fixed:
            raise RuntimeError("skipped")
        names_f, W_fixed = psytrack_fixed(X, y, session)
        t_fixed = time.time() - t0
        print("PsyTrack, fixed hyperparameters (one MAP solve): %.1f s" % t_fixed)
    except Exception as e:  # skipped, or getMAP not exported in some versions
        print("PsyTrack fixed-hyperparameter solve:", e)

    p = np.load(a.psytrack)
    p_names = [str(n) for n in p["names"]]
    wMode = p["wMode"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    x = np.arange(len(session))
    starts = np.concatenate([[0], np.where(np.diff(session) != 0)[0] + 1, [len(x)]])
    centres = (starts[:-1] + starts[1:]) / 2
    p_major = np.array([max(y[starts[j]:starts[j + 1]].mean(), 1 - y[starts[j]:starts[j + 1]].mean())
                        for j in range(len(centres))])
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.3 * len(names)), sharex=True)
    for ax, name in zip(axes, names):
        # grey: checkpoints where one choice exceeds 95% of trials, so the weights are not
        # identifiable and both methods return their prior
        for j in np.where(p_major > 0.95)[0]:
            ax.axvspan(starts[j], starts[j + 1], color="#dddddd", zorder=0, lw=0)
        c = WEIGHT_CLRS.get(name, "k")
        k = p_names.index(name)
        ax.plot(x, wMode[k], color=c, lw=1.6, label="PsyTrack, hyperOpt (hours)")
        if W_fixed is not None:
            ax.plot(x, W_fixed[names_f.index(name)], color="k", lw=0.9, ls="--", alpha=0.7,
                    label="PsyTrack, fixed hyperparameters (%.0f s)" % t_fixed)
        i = names.index(name)
        ax.errorbar(centres, W[i], yerr=SE[i], fmt="o", ms=3, color=c, mec="k", mew=0.4,
                    ecolor="#666", elinewidth=0.7, capsize=0,
                    label="ridge GLM per checkpoint, +/- 1 SE (%.2f s)" % t_glm)
        ax.axhline(0, color="k", ls="--", lw=0.6)
        ax.set_ylabel(name)
        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.9)
    ticks = starts[:-1][::10]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(["%d\n(%d)" % (t, iters[t] * 72) for t in ticks], fontsize=8)
    axes[-1].set_xlabel("Trial of the concatenated checkpoint rollouts   (in brackets: training trials seen)")
    fig.suptitle("%s: PsyTrack versus faster alternatives" % run, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    # agreement, numerically: correlation and RMS difference between PsyTrack and the GLM,
    # taking PsyTrack's mean within each checkpoint
    for i, name in enumerate(names):
        k = p_names.index(name)
        pm = np.array([wMode[k, starts[j]:starts[j + 1]].mean() for j in range(len(centres))])
        ok = np.isfinite(W[i])
        print("  %-10s corr(PsyTrack, GLM) = %.3f   rms diff = %.2f   GLM range %.1f..%.1f"
              % (name, np.corrcoef(pm[ok], W[i][ok])[0, 1], np.sqrt(np.mean((pm[ok] - W[i][ok]) ** 2)),
                 W[i][ok].min(), W[i][ok].max()))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
