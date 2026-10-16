#!/usr/bin/env python3
"""Compact mechanism and critic measures from pyrl activity rollouts (trials-a), per checkpoint.

Why. The mechanisms of the actor and, especially, of the critic are a
desideratum before the talk. A trials-a rollout saves, for every step of every trial, the
inputs, the policy's action probabilities, the critic's value, the actions, and both networks'
hidden states (43 MB per checkpoint). This script reduces one rollout to a few kB of measures
and the explorer draws them across checkpoints.

Input per checkpoint: <scratch>/work/pyrl/examples/<run>/checkpoints/iter_NNNNNN/trials_activity.pkl
  [trials, U (T,n,Nin), Z (T,n,Nout), Z_b (T,n), A (T,n,Nout), P (T,n), M (T,n), perf, r_p (T,n,N), r_v (T,n,N)]
Per trial: context, integral_loc/freq (+ = left in pyrl; flipped here to + = right), correct side,
stim_t0, dec_t0. Choice = argmax of the last valid action; hit = choice == correct side.

Measures (all per checkpoint):
  critic
    v_mean_t_hit, v_mean_t_miss     mean value trajectory over time, hit and miss trials (T,)
    v_start, v_end                   mean value at the first step and at the last valid step
    pe_terminal                      mean(hit - value at the last valid step): prediction error
    v_vs_abs_evidence                slope of value at the end of the stimulus on |relevant integral|
    v_vs_signed_evidence             slope on the signed relevant integral (does the critic code side)
    v_side_gap                       mean value on trials the actor will choose R minus L
    critic_pr                        participation ratio of r_v during the stimulus (dimensionality)
  actor
    actor_pr                         participation ratio of r_p during the stimulus
    pca_var                          variance fractions of the top 5 PCs of r_p (stimulus steps)
    pca_mean, pca_comp               the PCA basis (for projecting slow points later)
    choice_axis, evidence_axis       unit vectors in r_p space at the end of the stimulus: the LDA
                                     direction for eventual choice, and the regression direction on
                                     the signed relevant integral; their angle in degrees
    evidence_axis_ctx                the evidence axes per context and the angle between them
    critic_reads_evidence            correlation between the critic's value at stimulus end and the
                                     actor's projection on the evidence axis (|.|)
    traj_pc                          mean trajectory in the top 3 PCs by (context, eventual choice) (2,2,T,3)
Usage:
    python activity_summary.py --run <run> --scratch <scratch> --out <dir>    (any python3 with numpy)
"""
import argparse
import glob
import os
import pickle
import re

import numpy as np


class _Tolerant(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (object,), {"__setstate__": lambda self, s: self.__dict__.update(s)})


def load(path):
    with open(path, "rb") as f:
        return _Tolerant(f, encoding="latin1").load()


def participation_ratio(X):
    C = np.cov(X.T)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if len(ev) else np.nan


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def summarise(path):
    trials, U, Z, Z_b, A, P, M, perf, r_p, r_v = load(path)
    T, n, N = r_p.shape
    M = np.asarray(M, bool)
    last = np.array([int(M[:, i].sum()) - 1 for i in range(n)])          # index of the last valid step
    choice = np.array([int(np.argmax(A[last[i], i])) for i in range(n)])   # 1 = L, 2 = R, 0 = never decided
    ctx = np.array([0 if tr["context"] == "loc" else 1 for tr in trials])
    e_loc = -np.array([tr["integral_loc"] for tr in trials], float)        # + = right
    e_frq = -np.array([tr["integral_freq"] for tr in trials], float)
    rel = np.where(ctx == 0, e_loc, e_frq)
    correct = np.array([2 if tr["correct_side"] == "R" else 1 for tr in trials])
    hit = (choice == correct).astype(float)
    decided = choice != 0
    stim_t0 = int(trials[0]["stim_t0"]); dec_t0 = int(trials[0]["dec_t0"])
    stim_end = dec_t0 - 1
    out = {"n_trials": n, "p_decided": float(decided.mean()), "p_hit": float(hit[decided].mean()) if decided.any() else np.nan}
    # ---- critic
    V = np.asarray(Z_b, float)
    out["v_mean_t_hit"] = np.where(M[:, (hit == 1) & decided].any(1), np.nanmean(np.where(M[:, (hit == 1) & decided], V[:, (hit == 1) & decided], np.nan), 1), np.nan) if ((hit == 1) & decided).any() else np.full(T, np.nan)
    out["v_mean_t_miss"] = np.where(M[:, (hit == 0) & decided].any(1), np.nanmean(np.where(M[:, (hit == 0) & decided], V[:, (hit == 0) & decided], np.nan), 1), np.nan) if ((hit == 0) & decided).any() else np.full(T, np.nan)
    out["v_start"] = float(V[1].mean())
    v_last = np.array([V[last[i], i] for i in range(n)])
    out["v_end"] = float(v_last[decided].mean()) if decided.any() else np.nan
    out["pe_terminal"] = float((hit - v_last)[decided].mean()) if decided.any() else np.nan
    v_se = V[stim_end]
    X = np.column_stack([np.abs(rel), np.ones(n)]); out["v_vs_abs_evidence"] = float(np.linalg.lstsq(X, v_se, rcond=None)[0][0])
    X = np.column_stack([rel, np.ones(n)]); out["v_vs_signed_evidence"] = float(np.linalg.lstsq(X, v_se, rcond=None)[0][0])
    out["v_side_gap"] = float(v_se[choice == 2].mean() - v_se[choice == 1].mean()) if (choice == 2).any() and (choice == 1).any() else np.nan
    stim_steps = slice(stim_t0, dec_t0)
    Rv = r_v[stim_steps].reshape(-1, N); out["critic_pr"] = participation_ratio(Rv)
    # ---- actor
    Rp = r_p[stim_steps].reshape(-1, N); out["actor_pr"] = participation_ratio(Rp)
    mean = Rp.mean(0); Xc = Rp - mean
    Uu, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    out["pca_var"] = var[:5]; out["pca_mean"] = mean; out["pca_comp"] = Vt[:5]
    r_end = r_p[stim_end]                                                  # (n, N) at the end of the stimulus
    # choice axis: LDA with shrinkage between eventual L and R choosers
    if (choice == 1).sum() > 2 and (choice == 2).sum() > 2:
        mL, mR = r_end[choice == 1].mean(0), r_end[choice == 2].mean(0)
        Sw = np.cov(r_end[decided].T) + 0.1 * np.eye(N)
        ca = unit(np.linalg.solve(Sw, mR - mL))
    else:
        ca = np.full(N, np.nan)
    out["choice_axis"] = ca
    # evidence axis: ridge regression of r_end on the signed relevant integral (and per context)
    def ev_axis(mask):
        if mask.sum() < 5: return np.full(N, np.nan)
        x = rel[mask] - rel[mask].mean(); Y = r_end[mask] - r_end[mask].mean(0)
        return unit((x @ Y) / (x @ x + 1e-9))
    ea = ev_axis(np.ones(n, bool)); out["evidence_axis"] = ea
    ea_loc, ea_frq = ev_axis(ctx == 0), ev_axis(ctx == 1)
    out["evidence_axis_ctx"] = np.vstack([ea_loc, ea_frq])
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))) if np.isfinite(a).all() and np.isfinite(b).all() else np.nan
    out["angle_choice_evidence"] = ang(ca, ea)
    out["angle_evidence_ctx"] = ang(ea_loc, ea_frq)
    proj = (r_end - r_end.mean(0)) @ ea if np.isfinite(ea).all() else np.full(n, np.nan)
    out["critic_reads_evidence"] = float(abs(np.corrcoef(proj, v_se)[0, 1])) if np.isfinite(proj).all() and proj.std() > 0 else np.nan
    # mean trajectories in the top 3 PCs by context and eventual choice
    traj = np.full((2, 2, T, 3), np.nan)
    for c in (0, 1):
        for k, ch in enumerate((1, 2)):
            m = (ctx == c) & (choice == ch)
            if m.sum() >= 2:
                traj[c, k] = ((r_p[:, m] - mean) @ Vt[:3].T).mean(1)
    out["traj_pc"] = traj
    out["stim_t0"], out["dec_t0"] = stim_t0, dec_t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.scratch, "work", "pyrl", "examples", a.run, "checkpoints", "iter_*", "trials_activity.pkl")),
                   key=lambda p: int(re.search(r"iter_(\d+)", p).group(1)))
    rows = []
    for p in paths:
        it = int(re.search(r"iter_(\d+)", p).group(1))
        s = summarise(p); s["iter"] = it; rows.append(s)
        print("%s iter %5d: hit %.2f  V start %.2f end %.2f  PE %.3f  V~|ev| %.3f  critic reads ev %.2f  actor PR %.1f critic PR %.1f  angle(choice,ev) %.0f  angle(ev ctx) %.0f"
              % (a.run, it, s["p_hit"], s["v_start"], s["v_end"], s["pe_terminal"], s["v_vs_abs_evidence"], s["critic_reads_evidence"], s["actor_pr"], s["critic_pr"], s["angle_choice_evidence"], s["angle_evidence_ctx"]), flush=True)
    if not rows:
        print("no activity rollouts for", a.run); return
    os.makedirs(a.out, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    pack = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        try:
            pack[k] = np.array(vals, float) if np.isscalar(vals[0]) else np.array(vals)
        except Exception:
            pack[k] = np.array(vals, dtype=object)
    np.savez(os.path.join(a.out, a.run + "_activity.npz"), **pack)
    print("wrote", a.run, len(rows), "checkpoints")


if __name__ == "__main__":
    main()
