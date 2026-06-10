"""Phase 1: recover topic geometry from the binary vote matrix.

Estimator ladder: PCA (sanity) -> MDS (workhorse init) -> full MLE (Adam,
analytic gradients) -> dimensionality selection by held-out log-likelihood.
"""
import numpy as np
from scipy.spatial.distance import cdist

from .votes import sigmoid
from .world import gauge_normalize


# ----------------------------- cheap estimators -----------------------------

def recover_pca(Y, dim):
    Z = Y - Y.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    return gauge_normalize(Vt[:dim].T * S[:dim] / np.sqrt(Y.shape[0]))


def recover_mds(Y, dim):
    """Classical MDS on a co-voting dissimilarity between topic columns."""
    Z = Y - Y.mean(0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Z.T)
    # constant columns -> 0 correlation
    C = np.nan_to_num(C)
    np.fill_diagonal(C, 1.0)
    D2 = 2.0 * (1.0 - C)                     # squared-dissimilarity proxy
    n = D2.shape[0]
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:dim]
    L = np.sqrt(np.clip(vals[idx], 0.0, None))
    return gauge_normalize(vecs[:, idx] * L)


# ------------------------------- full MLE ----------------------------------

def mle_fit(Y, dim, mask=None, eps=0.03, n_iter=1500, lr=0.05,
            lam_v=1e-3, lam_b=1e-3, lam_c=1e-3, lam_s=1e-2,
            init_T=None, seed=0, verbose=False):
    """Fit the logistic ideal-point model by penalized maximum likelihood.

        logit_ij = b_i + c_j - ||v_i - t_j||^2 / (2 sigma_i^2)
        P_ij     = eps + (1 - 2 eps) * sigmoid(logit_ij)

    mask: float/bool (N, X); 1 = observed (used for training). Held-out
    entries (mask == 0) are simply excluded from the gradient.
    Gauge is re-pinned every iteration (centroid 0, unit RMS topics); the
    per-voter scale parameter s_i = log sigma_i absorbs the rescaling.
    """
    N, X = Y.shape
    rng = np.random.default_rng(seed)
    M = np.ones((N, X)) if mask is None else mask.astype(float)
    Yf = Y.astype(float)

    T = gauge_normalize(init_T.copy() if init_T is not None
                        else rng.normal(0, 0.5, (X, dim)))
    row = (Yf * M).sum(1, keepdims=True)
    V = (Yf * M) @ T / (row + 1.0) + 0.01 * rng.normal(size=(N, dim))
    b = np.zeros(N)
    c = np.zeros(X)
    s = np.zeros(N)                                   # s_i = log sigma_i

    params = [V, T, b, c, s]
    m = [np.zeros_like(p) for p in params]            # Adam moments
    v2 = [np.zeros_like(p) for p in params]
    b1, b2, ae = 0.9, 0.999, 1e-8

    for it in range(1, n_iter + 1):
        k = 0.5 * np.exp(-2.0 * s)                    # 1 / (2 sigma^2)
        d2 = cdist(V, T, "sqeuclidean")
        q = sigmoid(b[:, None] + c[None, :] - d2 * k[:, None])
        P = eps + (1.0 - 2.0 * eps) * q

        # dNLL / dlogit  (masked)
        dl = -M * (Yf / P - (1 - Yf) / (1 - P)) * (1 - 2 * eps) * q * (1 - q)
        rs = dl.sum(1)                                # (N,)
        dlk = dl * k[:, None]

        gV = -2.0 * k[:, None] * (rs[:, None] * V - dl @ T) + 2 * lam_v * V
        gT = -2.0 * (dlk.sum(0)[:, None] * T - dlk.T @ V)
        gb = rs + 2 * lam_b * b
        gc = dl.sum(0) + 2 * lam_c * c
        gs = 2.0 * k * (dl * d2).sum(1) + 2 * lam_s * s

        for p, g, mm, vv in zip(params, [gV, gT, gb, gc, gs], m, v2):
            mm += (1 - b1) * (g - mm)
            vv += (1 - b2) * (g * g - vv)
            mhat = mm / (1 - b1 ** it)
            vhat = vv / (1 - b2 ** it)
            p -= lr * mhat / (np.sqrt(vhat) + ae)

        # --- re-pin the gauge; sigma absorbs the scale ---
        mu = T.mean(0)
        T -= mu
        V -= mu
        r = np.sqrt((T ** 2).sum(1).mean())
        T /= r
        V /= r
        s -= np.log(r)

        if verbose and it % 300 == 0:
            nll = -(M * (Yf * np.log(P) + (1 - Yf)
                    * np.log(1 - P))).sum() / M.sum()
            print(f"  iter {it:5d}  train NLL/entry = {nll:.4f}")

    k = 0.5 * np.exp(-2.0 * s)
    d2 = cdist(V, T, "sqeuclidean")
    P = eps + (1 - 2 * eps) * \
        sigmoid(b[:, None] + c[None, :] - d2 * k[:, None])
    return dict(T=T, V=V, b=b, c=c, log_sigma=s, P=P)


# -------------------------- dimensionality selection ------------------------

def select_dimension(Y, dims, eps=0.03, holdout_frac=0.10, seed=0,
                     n_iter=800, verbose=True):
    """Mask a random fraction of votes, fit each candidate dim, score held-out
    mean log-likelihood. Returns {dim: heldout_ll} (higher is better)."""
    from .metrics import mean_loglik
    rng = np.random.default_rng(seed)
    mask = (rng.random(Y.shape) > holdout_frac)       # True = train
    out = {}
    for d in dims:
        init = recover_mds(Y, d)
        fit = mle_fit(Y, d, mask=mask, eps=eps, n_iter=n_iter,
                      init_T=init, seed=seed)
        out[d] = mean_loglik(Y, fit["P"], mask=~mask)
        if verbose:
            print(f"  dim {d}: held-out LL = {out[d]:.4f}")
    return out, mask
