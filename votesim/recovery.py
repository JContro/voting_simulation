"""Phase 1: recover topic geometry from the binary vote matrix.

Estimator ladder: PCA (sanity) -> MDS (workhorse init) -> full MLE (Adam,
analytic gradients) -> dimensionality selection by held-out log-likelihood.

Changes from v1:
  - Lower default LR (0.05 -> 0.01), cosine annealing schedule
  - Stronger L2 regularisation (lam_v 1e-3 -> 3e-2, lam_s 1e-2 -> 3e-1)
  - Early stopping on held-out validation log-likelihood
  - Extended iterations (1500 -> 3000)
  - New: ordinal MLE (mle_fit_ordinal) for Likert-scale responses
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


# ------------------------------ helpers ------------------------------------

def _validation_split(Y, holdout_frac=0.10, seed=0):
    """Create a train/val mask pair for held-out log-likelihood scoring."""
    rng = np.random.default_rng(seed)
    mask = (rng.random(Y.shape) > holdout_frac)   # True = train
    return mask, ~mask


# ------------------------------- full MLE ----------------------------------

def _cosine_lr(it, warmup=200, lr_max=0.01, lr_min=0.001, total=3000):
    """Cosine annealing with linear warmup."""
    if it < warmup:
        return lr_max * it / max(warmup, 1)
    t = (it - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * t))


def mle_fit(Y, dim, mask=None, eps=0.03, n_iter=3000, lr=0.01,
            lam_v=3e-2, lam_b=1e-2, lam_c=1e-2, lam_s=3e-1,
            val_mask=None, patience=5,
            init_T=None, seed=0, verbose=False):
    """Fit the logistic ideal-point model by penalized maximum likelihood.

        logit_ij = b_i + c_j - ||v_i - t_j||^2 / (2 sigma_i^2)
        P_ij     = eps + (1 - 2 eps) * sigmoid(logit_ij)

    mask: float/bool (N, X); 1 = observed (used for training). Held-out
    entries (mask == 0) are simply excluded from the gradient.
    val_mask: bool (N, X); used for early stopping validation LL.
    patience: number of val-LL checks without improvement before reverting.

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

    # ---------- early stopping state ----------
    best_val_ll = -np.inf
    best_params = None
    stall = 0

    # ---------- Adam state ----------
    params = [V, T, b, c, s]
    m = [np.zeros_like(p) for p in params]
    v2 = [np.zeros_like(p) for p in params]
    b1, b2, ae = 0.9, 0.999, 1e-8

    for it in range(1, n_iter + 1):
        # --- cosine-annealed learning rate ---
        eta = _cosine_lr(it, lr_max=lr, total=n_iter)

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
            p -= eta * mhat / (np.sqrt(vhat) + ae)

        # --- re-pin the gauge; sigma absorbs the scale ---
        mu = T.mean(0)
        T -= mu
        V -= mu
        r = np.sqrt((T ** 2).sum(1).mean())
        T /= r
        V /= r
        s -= np.log(r)

        # --- early stopping on validation LL ---
        if val_mask is not None and it % 100 == 0:
            val_ll = -np.inf
            with np.errstate(divide="ignore", invalid="ignore"):
                Pv = eps + (1 - 2 * eps) * sigmoid(
                    b[:, None] + c[None, :] - cdist(V, T, "sqeuclidean") * k[:, None]
                )
                Pv = np.clip(Pv, 1e-12, 1 - 1e-12)
                val_ll = (Yf * np.log(Pv) + (1 - Yf) * np.log(1 - Pv))[val_mask].mean()
            if val_ll > best_val_ll + 1e-6:
                best_val_ll = val_ll
                best_params = [p.copy() for p in params]
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    if verbose:
                        print(f"  early stopping at iter {it}  val LL = {val_ll:.4f}")
                    for p, bp in zip(params, best_params):
                        p[:] = bp
                    break

        if verbose and it % 500 == 0:
            nll = -(M * (Yf * np.log(P) + (1 - Yf)
                    * np.log(1 - P))).sum() / M.sum()
            print(f"  iter {it:5d}  train NLL/entry = {nll:.4f}")

    k = 0.5 * np.exp(-2.0 * s)
    d2 = cdist(V, T, "sqeuclidean")
    P = eps + (1 - 2 * eps) * \
        sigmoid(b[:, None] + c[None, :] - d2 * k[:, None])
    return dict(T=T, V=V, b=b, c=c, log_sigma=s, P=P)


# -------------------------- ordinal MLE ------------------------------------

def _softmax(x, axis=-1):
    """Numerically stable softmax."""
    xmax = x.max(axis=axis, keepdims=True)
    e = np.exp(x - xmax)
    return e / e.sum(axis=axis, keepdims=True)


def mle_fit_ordinal(Y, dim, n_categories=5, thresholds=None,
                    mask=None,
                    eps=0.03, n_iter=3000, lr=0.01,
                    lam_v=3e-2, lam_b=1e-2, lam_c=1e-2, lam_s=3e-1,
                    val_mask=None, patience=5,
                    init_T=None, seed=0, verbose=False):
    """Ordinal logistic ideal-point MLE.

    Y: (N, X) integer codes 0 .. n_categories-1.
    Same gauge convention as mle_fit(). Thresholds are learned if not
    provided (initialised at equally spaced quantiles of N(0,1)).
    """
    N, X = Y.shape
    K = n_categories
    rng = np.random.default_rng(seed)

    # thresholds: K-1 ordered cutpoints
    if thresholds is None:
        th = np.linspace(-1.5, 1.5, K - 1)
    else:
        th = np.array(thresholds)
    # enforce ordering by storing deltas and a base
    th_base = th[0]
    th_deltas = np.diff(th)                # K-2 positive deltas
    th_free = np.concatenate([[th_base], np.log(th_deltas)])

    # topic / voter init (same as binary MLE)
    init_T_ = gauge_normalize(init_T.copy() if init_T is not None
                              else rng.normal(0, 0.5, (X, dim)))
    row = Y.sum(1, keepdims=True)
    T = init_T_.copy()
    V = Y @ T / (row + 1.0) + 0.01 * rng.normal(size=(N, dim))
    b = np.zeros(N)
    c = np.zeros(X)
    s = np.zeros(N)

    M = np.ones((N, X)) if mask is None else mask.astype(float)
    Yf = Y.astype(float)

    params = [V, T, b, c, s, th_free]
    m = [np.zeros_like(p) for p in params]
    v2 = [np.zeros_like(p) for p in params]
    b1, b2, ae = 0.9, 0.999, 1e-8

    # one-hot encode Y for fast indexing
    Yoh = np.eye(K, dtype=float)[Y]       # (N, X, K)

    # early stopping state
    best_val_ll = -np.inf
    best_params = None
    stall = 0

    for it in range(1, n_iter + 1):
        eta = _cosine_lr(it, lr_max=lr, total=n_iter)

        k = 0.5 * np.exp(-2.0 * s)
        d2 = cdist(V, T, "sqeuclidean")
        logit = b[:, None] + c[None, :] - d2 * k[:, None]   # (N, X)

        # reconstruct ordered thresholds
        th = np.zeros(K - 1)
        th[0] = th_free[0]
        cum = th_free[0]
        for j in range(1, K - 1):
            cum += np.exp(th_free[j])
            th[j] = cum

        # ordinal probabilities (N, X, K)
        cum_probs = np.zeros((N, X, K - 1))
        for kk in range(K - 1):
            cum_probs[:, :, kk] = sigmoid(th[kk] - logit)
        P0 = np.zeros((N, X, K))
        P0[:, :, 0] = cum_probs[:, :, 0]
        for kk in range(1, K - 1):
            P0[:, :, kk] = cum_probs[:, :, kk - 1] - cum_probs[:, :, kk]
        P0[:, :, K - 1] = 1.0 - cum_probs[:, :, -1]
        P = eps / K + (1 - eps) * P0     # lapse-rate floor
        # note: P sums to 1 over K per (i,j)

        # gradient of NLL w.r.t. logit
        # d(NLL) = -[I(y=k) - P_k] * dlogit   summed over k appropriately
        # For ordinal: dNLL/dlogit = sum_k [I(y=k) - P_k] * d(P_k)/dlogit / P_k
        # But the multinomial gradient w.r.t. logit is:
        # dNLL/dlogit = sum_k (I(y=k) - P_k) * (P_k contributions from logit)
        # For the ordered logit: d(NLL)/dlogit = -(F_kp1' - F_k') aggregated
        #
        # Simpler approach: use the chain rule via dlog_loss
        # NLL_i = -sum_k y_ik * log(P_ik)
        # d NLL_i / d logit = -sum_k (y_ik / P_ik) * dP_ik/dlogit
        #
        # dP_0/dlogit = -sig'(th_0 - logit) = -sig(t0)*(1-sig(t0))  [k=0]
        # dP_k/dlogit = (sig'(th_k - logit) - sig'(th_{k-1} - logit))  [1<=k<K-1]
        # dP_{K-1}/dlogit = sig'(th_{K-2} - logit)                   [k=K-1]

        # precompute sigmoid values and derivatives
        sig_th = np.zeros((N, X, K - 1))
        sigp_th = np.zeros((N, X, K - 1))
        for kk in range(K - 1):
            sig_th[:, :, kk] = sigmoid(th[kk] - logit)
            sigp_th[:, :, kk] = sig_th[:, :, kk] * (1 - sig_th[:, :, kk])

        # dP/dlogit for each category
        dP_dlogit = np.zeros((N, X, K))
        dP_dlogit[:, :, 0] = -sigp_th[:, :, 0]
        for kk in range(1, K - 1):
            dP_dlogit[:, :, kk] = sigp_th[:, :, kk] - sigp_th[:, :, kk - 1]
        dP_dlogit[:, :, K - 1] = sigp_th[:, :, K - 2]

        # dNLL/dlogit = -sum_k (y_ik / P_ik) * dP_ik/dlogit
        dl = np.zeros((N, X))
        for kk in range(K):
            dl += -(Yoh[:, :, kk] / np.clip(P[:, :, kk], 1e-12, None)) * dP_dlogit[:, :, kk]
        # mask (if provided)
        dl = dl * M

        rs = dl.sum(1)
        dlk = dl * k[:, None]

        gV = -2.0 * k[:, None] * (rs[:, None] * V - dl @ T) + 2 * lam_v * V
        gT = -2.0 * (dlk.sum(0)[:, None] * T - dlk.T @ V)
        gb = rs + 2 * lam_b * b
        gc = dl.sum(0) + 2 * lam_c * c
        gs = 2.0 * k * (dl * d2).sum(1) + 2 * lam_s * s

        # gradient w.r.t. threshold parameters
        gth = np.zeros(K - 1)
        for kk in range(K - 1):
            # contribution of th_kk to all categories involving it
            contrib = np.zeros((N, X))
            # th_kk appears in P[:,:,kk] and P[:,:,kk+1]
            # dP_kk/dth_kk = sigp(th_kk - logit)
            # dP_{kk+1}/dth_kk = -sigp(th_kk - logit)
            dp = sigp_th[:, :, kk]
            # for each (i,j): -sum yk/Pk * dPk/dth_kk
            # = -(y_kk/P_kk * sigp + y_{kk+1}/P_{kk+1} * (-sigp))
            contrib = -(Yoh[:, :, kk] / np.clip(P[:, :, kk], 1e-12, None)) * dp \
                      + (Yoh[:, :, kk + 1] / np.clip(P[:, :, kk + 1], 1e-12, None)) * dp
            contrib = contrib * M
            gth[kk] = contrib.sum()

        # chain-rule for th_free (deltas)
        gth_free = np.zeros_like(th_free)
        gth_free[0] = gth[0]
        cumulative_grad = gth[0]
        for j in range(1, K - 1):
            cumulative_grad += gth[j]
            gth_free[j] = cumulative_grad * np.exp(th_free[j])

        for p, g, mm, vv in zip(params, [gV, gT, gb, gc, gs, gth_free], m, v2):
            mm += (1 - b1) * (g - mm)
            vv += (1 - b2) * (g * g - vv)
            mhat = mm / (1 - b1 ** it)
            vhat = vv / (1 - b2 ** it)
            p -= eta * mhat / (np.sqrt(vhat) + ae)

        # --- re-pin the gauge ---
        mu = T.mean(0)
        T -= mu
        V -= mu
        r = np.sqrt((T ** 2).sum(1).mean())
        T /= r
        V /= r
        s -= np.log(r)

        # --- early stopping ---
        if val_mask is not None and it % 100 == 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_prob = np.log(np.clip(P, 1e-12, 1.0))
                val_ll = (Yoh * log_prob).sum(axis=-1)[val_mask].mean() \
                    if val_mask is not None else -np.inf
            if val_ll > best_val_ll + 1e-6:
                best_val_ll = val_ll
                best_params = [p.copy() for p in params]
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    if verbose:
                        print(f"  early stopping at iter {it}  val LL = {val_ll:.4f}")
                    for p, bp in zip(params, best_params):
                        p[:] = bp
                    break

        if verbose and it % 500 == 0:
            log_prob = np.log(np.clip(P, 1e-12, 1.0))
            nll = -(Yoh * log_prob).sum(axis=-1).mean()
            print(f"  iter {it:5d}  train NLL/entry = {nll:.4f}")

    # reconstruct final thresholds
    th_final = np.zeros(K - 1)
    th_final[0] = th_free[0]
    cum = th_free[0]
    for j in range(1, K - 1):
        cum += np.exp(th_free[j])
        th_final[j] = cum

    return dict(T=T, V=V, b=b, c=c, log_sigma=s,
                thresholds=th_final, P=P)


# -------------------------- dimensionality selection ------------------------

def select_dimension(Y, dims, type="binary", eps=0.03, holdout_frac=0.10,
                     seed=0, n_iter=800, verbose=True):
    """Mask a random fraction of votes, fit each candidate dim, score held-out
    mean log-likelihood. Returns {dim: heldout_ll} (higher is better).

    type: "binary" or "ordinal" — selects which MLE to use.
    """
    from .metrics import mean_loglik
    mask, val_mask = _validation_split(Y, holdout_frac=holdout_frac, seed=seed)
    out = {}
    for d in dims:
        init = recover_mds(Y, d)
        if type == "ordinal":
            n_cat = int(Y.max() + 1)
            fit = mle_fit_ordinal(Y, d, n_categories=n_cat, eps=eps,
                                  mask=mask, n_iter=n_iter,
                                  val_mask=val_mask,
                                  init_T=init, seed=seed)
            # held-out LL for ordinal
            log_prob = np.log(np.clip(fit["P"], 1e-12, 1.0))
            Yoh = np.eye(n_cat)[Y]
            out[d] = (Yoh * log_prob).sum(axis=-1)[~mask].mean()
        else:
            fit = mle_fit(Y, d, mask=mask, eps=eps, n_iter=n_iter,
                          val_mask=val_mask, init_T=init, seed=seed)
            out[d] = mean_loglik(Y, fit["P"], mask=~mask)
        if verbose:
            print(f"  dim {d}: held-out LL = {out[d]:.4f}")
    return out, mask