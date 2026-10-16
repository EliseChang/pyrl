#!/usr/bin/env python3
"""Mechanism of a pyrl (Song-Wang) reward-trained RNN at every checkpoint: fixed points per
context, the line attractor's eigenvalue, the selection vector, the Nature-2025 triangle.

Why. The mechanisms of these networks (their fixed points, for the actor and for the critic)
have to be studied before the talk. This reuses gatsby_talk/solution_space/solution_space.py
(the Python port of the Nature decomposition) on the actor's saved parameters, so the RL
networks are placed in the same triangle as the vanilla ones and the poster's rows (fixed
points in PC space across training, largest eigenvalue over training) can be drawn for them.

The mapping. pyrl's vanilla policy (gru.py, gates off) updates, with alpha = dt/tau,
    x_t = (1 - alpha) x_{t-1} + alpha (r_{t-1} Wrec + u_t Win + bin),    r = tanh(x)
(row-vector convention: r.dot(Wrec)). A fixed point under constant input u satisfies
    x = r Wrec + u Win + bin,  r = tanh(x)   ->   r = tanh(Wrec^T r + Win^T u + bin)
which is solution_space's discrete map h = tanh(wR h + wI u + bR) with wR = Wrec^T, wI = Win^T,
bR = bin. The leaky Jacobian is (1 - alpha) I + alpha J_discrete, so eigenvectors are shared and
an eigenvalue of 1 (the line attractor) is 1 in both; solution_space's "largest eigenvalue" is
reported as the discrete one and mapped with lam_leaky = 1 - alpha + alpha * lam.
Inputs: pagan_hist channels (LMR, HML, LOC, FREQ, 4 history channels); the context-only
fixed points use LOC or FREQ = 1, everything else 0 (as in the Nature analysis). Readout: the
scalar wO = Wout[:, RIGHT] - Wout[:, LEFT] (right minus left logit; sign convention as in
solution_space.from_torch_rnn but with + = right).

Per checkpoint it writes: fixed points per context (rows), the chosen pair nearest the decision
boundary, leading eigenvalues (discrete and leaky), r, l1, l2, the three terms and the triangle
coordinates per feature, and the selection-vector angle between contexts. Output is one npz
per run under <out>/<run>_mechanism.npz; the explorer reads it.

Usage (rebel, rat_strategy_discovery venv; the pyrl checkpoints are Python 2 pickles):
    python pyrl_mechanism.py --run pagan_hist_curriculum_curr_w001 [--every 1] \
        --data ~/pyrl_work/examples/work/data --out /mnt/big/pyrl_work/mechanism
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "solution_space"))   # in the gatsby_talk tree
sys.path.insert(0, _here)                                                 # or next to this file (rebel)
import solution_space as ss  # noqa: E402

LMR, HML, LOC, FREQ = 0, 1, 2, 3
LEFT, RIGHT = 1, 2


class _Tolerant(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (object,), {"__setstate__": lambda self, s: self.__dict__.update(s)})


def load_pkl(path):
    with open(path, "rb") as f:
        return _Tolerant(f, encoding="latin1").load()


def from_pyrl(params, nin_task=4):
    """pyrl policy params (Win (Nin,N), bin (N,), Wrec (N,N), Wout (N,Nout), bout) -> the
    solution_space convention (wR (N,N), bR (N,), wI (N,4), wO (N,), bO)."""
    Win, bin_, Wrec = np.asarray(params["Win"]), np.asarray(params["bin"]), np.asarray(params["Wrec"])
    Wout, bout = np.asarray(params["Wout"]), np.asarray(params["bout"])
    wI = Win[:nin_task, :].T                         # (N, 4): LMR, HML, LOC, FREQ
    wO = Wout[:, RIGHT] - Wout[:, LEFT]
    bO = float(bout[RIGHT] - bout[LEFT])
    return dict(wR=Wrec.T, bR=bin_, wI=wI, wO=wO, bO=bO)


# Slow points rather than exact fixed points. On a trained RL network the strict tolerance
# (1e-3 RMS one-step movement) finds nothing: in the LOC context every start collapses to one
# stable point, in the FRQ context the autonomous dynamics never settle (spectral radius ~1.8).
# With 1e-2 the search returns the slow manifold (Sussillo and Barak's slow points), which is
# the object the linearisation is about. The achieved loss is recorded per checkpoint.
FP_TOL = 1e-2


def forward_settle(wR, bR, u, alpha, R0, n_steps=1500):
    """Run the leaky dynamics from states R0 (rows, r-space) under constant input u and return
    the settled r. Stable slow points are reached this way even when the gradient search from
    random starts fails (which it does on trained RL networks)."""
    X = np.arctanh(np.clip(R0, -0.999, 0.999))
    for _ in range(n_steps):
        X = (1 - alpha) * X + alpha * (np.tanh(X) @ wR.T + u + bR)
    return np.tanh(X)


def candidates(wR, bR, u, alpha, rng, visited=None, n_random=150):
    """Starting points for the slow-point search: states the network actually visited in the
    rollout at this checkpoint (if given), their forward-settled versions, and random points."""
    N = len(bR)
    parts = [rng.standard_normal((n_random, N)) * 0.5]
    if visited is not None and len(visited):
        idx = rng.choice(len(visited), size=min(200, len(visited)), replace=False)
        V = np.asarray(visited[idx], float)
        parts += [V, forward_settle(wR, bR, u, alpha, V)]
    return np.vstack(parts)


def analyse_params(params, alpha, rng, h_init=None, n_init=300, n_iter=2000, visited=None):
    net = from_pyrl(params)
    wR, bR, wI, wO, bO = net["wR"], net["bR"], net["wI"], net["wO"], net["bO"]
    out = {}
    F = {}
    for ctx, col in (("loc", LOC), ("frq", FREQ)):
        u = wI[:, col]
        H0 = candidates(wR, bR, u, alpha, rng, visited)
        if h_init and h_init.get(ctx) is not None:
            H0 = np.vstack([H0, h_init[ctx]])
        F[ctx] = ss.find_fixed_points(wR, bR, u, n_iter=n_iter, lr=0.02, rng=rng, h_init=H0, tol=FP_TOL)
        out["fp_" + ctx] = F[ctx]
        out["fp_loss_" + ctx] = ss.fixed_point_loss(wR, bR, u, F[ctx]) if len(F[ctx]) else np.array([])
        # The slow manifold itself, for drawing: states the network visited during the stimulus,
        # run forward under the context input until they settle, kept where they are slow
        # (no deduplication, so a line attractor shows as a line). Up to 150 points.
        if visited is not None and len(visited):
            idx = rng.choice(len(visited), size=min(150, len(visited)), replace=False)
            S = forward_settle(wR, bR, u, alpha, np.asarray(visited[idx], float), n_steps=800)
            sl = ss.fixed_point_loss(wR, bR, u, S)
            out["settled_" + ctx] = S[sl < 5 * FP_TOL]
            out["settled_loss_" + ctx] = sl[sl < 5 * FP_TOL]
        else:
            out["settled_" + ctx] = np.zeros((0, len(bR)))
    dec = ss.decompose(wR, bR, wI, wO, bO, F["loc"], F["frq"], fixed_point_threshold=FP_TOL, ctx_cols=(LOC, FREQ), feat_cols=(LMR, HML))
    if dec is None:
        out["ok"] = False
        return out, F
    out["ok"] = True
    for k in ("te1", "te2", "tri1", "tri2", "f1", "f2", "r", "l1", "l2", "eig1", "eig2", "tot1", "tot2"):
        out[k] = np.asarray(dec[k])
    out["eig1_leaky"] = 1 - alpha + alpha * float(dec["eig1"]); out["eig2_leaky"] = 1 - alpha + alpha * float(dec["eig2"])
    # selection-vector angle between contexts (degrees), the quantity Angus suggested instead of Schur
    l1, l2 = dec["l1"], dec["l2"]
    out["sel_angle"] = float(np.degrees(np.arccos(np.clip(l1 @ l2 / (np.linalg.norm(l1) * np.linalg.norm(l2)), -1, 1))))
    out["n_fp_loc"], out["n_fp_frq"] = len(F["loc"]), len(F["frq"])
    return out, F


def analyse_vanilla(fn, out_dir):
    """A sup_*.npz of train_supervised.py (31 checkpoints of a torch RNN + Linear): weights as in
    selection_vector_tracking.py (fixation channel folded into the bias, inputs LMR, HML, ctx1,
    ctx2 in columns 1-4, read-out = left minus right logit), discrete dynamics (alpha = 1)."""
    import json
    z = np.load(fn, allow_pickle=True)
    k = {n[5:]: z[n] for n in z.files if n.startswith("ckpt_")}
    cfg = json.loads(str(z["config"])); n_ck = k["rnn.weight_hh_l0"].shape[0]
    rng = np.random.default_rng(0)
    rows, prev = [], None
    name = os.path.basename(fn).replace(".npz", "")
    for i in range(n_ck):
        W_ih, W_hh = k["rnn.weight_ih_l0"][i].astype(float), k["rnn.weight_hh_l0"][i].astype(float)
        b = (k["rnn.bias_ih_l0"][i] + k["rnn.bias_hh_l0"][i]).astype(float); Wo = k["lin.weight"][i].astype(float)
        wR, bR, wI = W_hh, b + W_ih[:, 0], W_ih[:, 1:5]
        wO = Wo[2] - Wo[1]                                   # right minus left, as for pyrl
        params = {"_direct": (wR, bR, wI, wO, 0.0)}
        res, F = analyse_direct(wR, bR, wI, wO, 0.0, 1.0, rng, h_init=prev)
        prev = {c: (np.vstack([F[c], rng.standard_normal((100, wR.shape[0])) * 0.5]) if len(F[c]) else None) for c in F}
        res["iter"] = i * cfg.get("ckpt_every", 1)
        rows.append(res)
        print("%s ckpt %2d (%d steps): fps %d/%d %s" % (name, i, res["iter"], res.get("n_fp_loc", 0), res.get("n_fp_frq", 0),
              ("eig %.3f/%.3f angle %.0f tri1 %s" % (res["eig1"], res["eig2"], res["sel_angle"], np.round(res["tri1"], 2))) if res["ok"] else "no pair"), flush=True)
    keys = sorted({kk for r in rows for kk in r})
    pack = {}
    for kk in keys:
        vals = [r.get(kk) for r in rows]
        pack[kk] = np.array([np.nan if v is None else v for v in vals], float) if all(np.isscalar(v) or v is None for v in vals) else np.array(vals, dtype=object)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, name + "_vanilla_mechanism.npz"), **pack, allow_pickle=True)
    print("wrote", name)


def analyse_direct(wR, bR, wI, wO, bO, alpha, rng, h_init=None, visited=None):
    """The analysis on explicit (wR, bR, wI, wO, bO); analyse_params is this after from_pyrl."""
    out, F = {}, {}
    for ctx, col in (("loc", LOC), ("frq", FREQ)):
        u = wI[:, col]
        H0 = candidates(wR, bR, u, alpha, rng, visited)
        if h_init and h_init.get(ctx) is not None:
            H0 = np.vstack([H0, h_init[ctx]])
        F[ctx] = ss.find_fixed_points(wR, bR, u, n_iter=2000, lr=0.02, rng=rng, h_init=H0, tol=FP_TOL)
        out["fp_" + ctx] = F[ctx]
        out["fp_loss_" + ctx] = ss.fixed_point_loss(wR, bR, u, F[ctx]) if len(F[ctx]) else np.array([])
        S = forward_settle(wR, bR, u, alpha, rng.standard_normal((150, len(bR))) * 0.5, n_steps=800)
        sl = ss.fixed_point_loss(wR, bR, u, S)
        out["settled_" + ctx] = S[sl < 5 * FP_TOL]; out["settled_loss_" + ctx] = sl[sl < 5 * FP_TOL]
    dec = ss.decompose(wR, bR, wI, wO, bO, F["loc"], F["frq"], fixed_point_threshold=FP_TOL, ctx_cols=(LOC, FREQ), feat_cols=(LMR, HML))
    if dec is None:
        out["ok"] = False; return out, F
    out["ok"] = True
    for kk in ("te1", "te2", "tri1", "tri2", "f1", "f2", "r", "l1", "l2", "eig1", "eig2", "tot1", "tot2"):
        out[kk] = np.asarray(dec[kk])
    out["eig1_leaky"] = 1 - alpha + alpha * float(dec["eig1"]); out["eig2_leaky"] = 1 - alpha + alpha * float(dec["eig2"])
    l1, l2 = dec["l1"], dec["l2"]
    out["sel_angle"] = float(np.degrees(np.arccos(np.clip(l1 @ l2 / (np.linalg.norm(l1) * np.linalg.norm(l2)), -1, 1))))
    out["n_fp_loc"], out["n_fp_frq"] = len(F["loc"]), len(F["frq"])
    return out, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--data", default=os.path.expanduser("~/pyrl_work/examples/work/data"))
    ap.add_argument("--out", default="/mnt/big/pyrl_work/mechanism")
    ap.add_argument("--every", type=int, default=1, help="use every k-th checkpoint")
    ap.add_argument("--max-ckpts", type=int, default=40, help="subsample to at most this many checkpoints")
    ap.add_argument("--net", choices=["policy", "baseline"], default="policy")
    ap.add_argument("--activity-scratch", default=None,
                    help="scratch root with trials_activity.pkl rollouts; visited states seed the search")
    ap.add_argument("--vanilla", nargs="*", default=[],
                    help="sup_*.npz files of the NeuroGym supervised networks (solution_space format): the "
                         "same analysis with alpha = 1, so RL and vanilla networks are compared under one method")
    a = ap.parse_args()
    for fn in a.vanilla:
        analyse_vanilla(fn, a.out)
    if not a.run:
        return
    os.makedirs(a.out, exist_ok=True)
    for run in a.run:
        cps = sorted(glob.glob(os.path.join(a.data, run, "checkpoints", "iter_*.pkl")),
                     key=lambda p: int(re.search(r"iter_(\d+)", p).group(1)))
        if len(cps) > a.max_ckpts:
            idx = np.unique(np.linspace(0, len(cps) - 1, a.max_ckpts).round().astype(int))
            cps = [cps[i] for i in idx]
        cps = cps[::a.every]
        cfg = load_pkl(cps[0])["config"]
        alpha = float(cfg["dt"]) / float(cfg["tau"])
        rng = np.random.default_rng(0)
        rows, prev = [], None
        act = {}
        if a.activity_scratch:
            for p in glob.glob(os.path.join(a.activity_scratch, "work", "pyrl", "examples", run, "checkpoints", "iter_*", "trials_activity.pkl")):
                act[int(re.search(r"iter_(\d+)", p).group(1))] = p
        for cp in cps:
            d = load_pkl(cp)
            params = d["policy_params"] if a.net == "policy" else d["baseline_params"]
            visited = None
            if act:
                near = min(act, key=lambda k: abs(k - d["iter"]))
                dd = load_pkl(act[near])
                r = dd[8] if a.net == "policy" else dd[9]          # r_p or r_v: (T, n, N)
                t0 = int(dd[0][0]["stim_t0"]); t1 = int(dd[0][0]["dec_t0"])
                visited = r[t0:t1].reshape(-1, r.shape[-1])
            res, F = analyse_params(params, alpha, rng, h_init=prev, visited=visited)
            # warm-start the next checkpoint's search from this one's fixed points plus random
            prev = {k: np.vstack([F[k], rng.standard_normal((100, F[k].shape[1] if len(F[k]) else len(params["bin"]))) * 0.5]) if len(F[k]) else None for k in F}
            res["iter"] = d["iter"]
            rows.append(res)
            print("%s iter %5d: fps loc/frq %d/%d  %s" % (run, d["iter"], res.get("n_fp_loc", 0), res.get("n_fp_frq", 0),
                  ("eig %.3f/%.3f (leaky %.3f/%.3f) angle %.1f tri1 %s" % (res["eig1"], res["eig2"], res["eig1_leaky"], res["eig2_leaky"], res["sel_angle"], np.round(res["tri1"], 2))) if res["ok"] else "no fixed point pair"), flush=True)
        keys = sorted({k for r in rows for k in r})
        pack = {}
        for k in keys:
            vals = [r.get(k) for r in rows]
            if all(np.isscalar(v) or v is None for v in vals):
                pack[k] = np.array([np.nan if v is None else v for v in vals], float)
            else:
                pack[k] = np.array(vals, dtype=object)
        np.savez(os.path.join(a.out, "%s_%s_mechanism.npz" % (run, a.net)), **pack, allow_pickle=True)
        print("wrote", run)


if __name__ == "__main__":
    main()
