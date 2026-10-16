"""
solution_space.py: Python port of the Nature 2025 solution-space decomposition (Pagan et al.),
from flexible_decision_making_rnn/private/compute_mechanism.m, localize_model.m,
analysis_fixed_points.m and compute_terms_from_eigs.m. Written 2026-10-06 so that networks trained
in NeuroGym (supervised or RL) can be placed in the barycentric triangle.

THE NETWORK CLASS. A discrete-time tanh RNN without leak,
        h[t+1] = tanh(wR h[t] + bR + wI u[t]),     out = wO h + bO
with inputs u = (LMR, HML, ctx1, ctx2). torch.nn.RNN(nonlinearity='tanh') has exactly this form
(W_hh h + b_hh + W_ih x + b_ih, with bR = b_hh + b_ih), so NeuroGym-trained vanilla RNNs map onto it
directly; an RL policy's extra input channels (fixation, previous reward/action) are held at their
"no input" value when the context-only fixed points are found.

THE DECOMPOSITION (localize_model.m). In each context c, at the fixed point closest to the decision
boundary (|wO f + bO| smallest among stable fixed points):
    sat_c   = 1 - tanh(wR f_c + bR + wI[:, ctx_c])^2        saturation factor, per unit
    J_c     = sat_c[:, None] * wR                           Jacobian
    r_c     = right eigenvector of J_c for the largest eigenvalue (the line attractor)
    l_c     = left eigenvector for the same eigenvalue (the selection vector), signed so r.wO > 0
    in_c    = sat_c * wI[:, feature]                        effective input of one feature
The context difference of the feature's final effect on the line attractor, l1.in1 - l2.in2,
is split exactly into three terms, with r = mean(r1, r2), s_c = l_c - r:
    mechanism 1  mean(in) . (l1 - l2)                       "selection vector modulation"
    mechanism 2  r . (in1 - in2)                            "direct input modulation"
    mechanism 3  mean(s) . (in1 - in2)                      "indirect input modulation"
compute_mechanism.m returns these as (jacdif, indif_norm, indif_nonnorm) with r normalised to unit
length and the sign of r fixed by r.wO > 0; the triangle shows |terms| / sum|terms|.
Feature 1 (LMR) is relevant in context 1 (ctx index 2, i.e. wI[:, 2]); feature 2 (HML) in context 2.
"""
import numpy as np


def fixed_point_loss(wR, bR, wI_ctx, F):
    """RMS one-step movement of each candidate fixed point (rows of F). Small = stable fixed point."""
    new = np.tanh(F @ wR.T + bR[None, :] + wI_ctx[None, :])
    return np.sqrt(np.mean((new - F) ** 2, axis=1))


def find_fixed_points(wR, bR, wI_ctx, n_init=400, n_iter=3000, lr=0.05, rng=None, h_init=None, tol=1e-3):
    """Find fixed points of h = tanh(wR h + bR + wI_ctx) by gradient descent on the squared one-step
    movement from many random (or supplied) initial conditions. Returns the unique points with
    loss < tol (rows). This is the Sussillo-Barak recipe; it replaces the JAX fixed-point search of
    the original training code, which is not kept with the saved networks."""
    rng = np.random.default_rng(0) if rng is None else rng
    N = wR.shape[0]
    H = (rng.standard_normal((n_init, N)) * 0.5) if h_init is None else np.array(h_init, float)
    for _ in range(n_iter):
        pre = H @ wR.T + bR[None, :] + wI_ctx[None, :]
        new = np.tanh(pre)
        mov = new - H                                           # q = 0.5 ||mov||^2 ; dq/dH = (J^T - I) mov
        grad = (mov * (1 - new ** 2)) @ wR - mov                # (J^T mov) - mov, J = diag(1-new^2) wR
        H = H - lr * grad
    loss = fixed_point_loss(wR, bR, wI_ctx, H)
    H = H[loss < tol]
    if len(H) == 0:
        return H
    # Deduplicate: greedy merge of points closer than 1e-2 in RMS distance.
    keep = []
    for h in H:
        if all(np.sqrt(np.mean((h - k) ** 2)) > 1e-2 for k in keep):
            keep.append(h)
    return np.array(keep)


def _linearise(wR, bR, wI, f, ctx_col, wO):
    """Saturation factor, line attractor r and selection vector l at fixed point f in one context."""
    sat = 1 - np.tanh(wR @ f + bR + wI[:, ctx_col]) ** 2
    J = sat[:, None] * wR
    D, V = np.linalg.eig(J)
    order = np.argsort(-D.real)                                 # MATLAB eig is unordered; compute_mechanism takes
    k = order[0]                                                # D(1,1), which for these matrices is the largest.
    r = V[:, k].real
    l = np.linalg.inv(V)[k, :].real
    if np.corrcoef(r, wO)[0, 1] < 0:
        r, l = -r, -l
    return sat, r, l, D.real[k]


def decompose(wR, bR, wI, wO, bO, F1, F2, fixed_point_threshold=0.002, ctx_cols=(2, 3), feat_cols=(0, 1)):
    """compute_mechanism.m. F1, F2: candidate fixed points (rows) in context 1 and 2.
    Returns dict with te1, te2 (each the 3 raw terms for feature 1 and feature 2), the chosen fixed
    points, the leading eigenvalues, and the normalised triangle coordinates."""
    wO = np.asarray(wO).reshape(-1); bO = float(np.asarray(bO).reshape(-1)[0]) if np.size(bO) else 0.0
    chosen = []
    for F, c in ((F1, ctx_cols[0]), (F2, ctx_cols[1])):
        loss = fixed_point_loss(wR, bR, wI[:, c], F)
        good = np.where(loss < fixed_point_threshold)[0]
        if len(good) == 0:
            return None
        out = F[good] @ wO + bO
        chosen.append(F[good[np.argmin(np.abs(out))]])
    f1, f2 = chosen
    sat1, r1, l1, ev1 = _linearise(wR, bR, wI, f1, ctx_cols[0], wO)
    sat2, r2, l2, ev2 = _linearise(wR, bR, wI, f2, ctx_cols[1], wO)
    # compute_mechanism.m applies the readout sign check twice (per fixed point and again after),
    # then averages and normalises r.
    if r1 @ wO < 0: r1, l1 = -r1, -l1
    if r2 @ wO < 0: r2, l2 = -r2, -l2
    r = (r1 + r2) / 2; r = r / np.linalg.norm(r)
    if r @ wO < 0: r = -r

    def terms(in1, in2):
        s1, s2 = l1 - r, l2 - r
        mui, difi = (in1 + in2) / 2, in1 - in2
        difl, mus = l1 - l2, (s1 + s2) / 2
        return np.array([mui @ difl, r @ difi, mus @ difi]), l1 @ in1 - l2 @ in2

    te1, tot1 = terms(sat1 * wI[:, feat_cols[0]], sat2 * wI[:, feat_cols[0]])
    te2, tot2 = terms(sat1 * wI[:, feat_cols[1]], sat2 * wI[:, feat_cols[1]])
    te2, tot2 = -te2, -tot2                                     # sign convention of compute_mechanism.m
    tri = lambda t: np.abs(t) / np.sum(np.abs(t))
    return dict(te1=te1, te2=te2, tot1=tot1, tot2=tot2, tri1=tri(te1), tri2=tri(te2), f1=f1, f2=f2,
                eig1=ev1, eig2=ev2, r=r, l1=l1, l2=l2)


def load_matlab_model(dire):
    """A saved network from flexible_decision_making_rnn (wO, wR, wI, h0, bO, bR, f1, f2 as .npy)."""
    g = lambda n: np.load(f'{dire}/{n}.npy')
    return dict(wR=g('wR'), bR=g('bR').reshape(-1), wI=g('wI'), wO=g('wO').reshape(-1), bO=g('bO'), F1=g('f1'), F2=g('f2'))


def from_torch_rnn(state_dict, in_cols=(1, 2, 3, 4), prefix_rnn='rnn', prefix_out='lin'):
    """Map a torch nn.RNN + Linear state dict (as saved by train_supervised.py) onto (wR, bR, wI, wO, bO).
    in_cols: which input channels are (LMR, HML, ctx1, ctx2) in the NeuroGym observation; other
    channels (fixation etc.) are folded into bR at their value during the stimulus period (fixation = 1).
    """
    sd = {k: np.asarray(v) for k, v in state_dict.items()}
    W_ih, W_hh = sd[f'{prefix_rnn}.weight_ih_l0'], sd[f'{prefix_rnn}.weight_hh_l0']
    b = sd[f'{prefix_rnn}.bias_ih_l0'] + sd[f'{prefix_rnn}.bias_hh_l0']
    wI = W_ih[:, list(in_cols)]
    fix_cols = [c for c in range(W_ih.shape[1]) if c not in in_cols]
    bR = b + (W_ih[:, fix_cols].sum(1) if fix_cols else 0)     # fixation channel on (=1) during the stimulus
    W_out, b_out = sd[f'{prefix_out}.weight'], sd[f'{prefix_out}.bias']
    wO = W_out[1] - W_out[2]; bO = b_out[1] - b_out[2]           # left minus right logit as the scalar read-out
    return dict(wR=W_hh, bR=bR, wI=wI, wO=wO, bO=bO)
