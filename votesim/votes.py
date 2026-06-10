"""Response models: binary and ordinal approval probabilities + sampling."""
import numpy as np
from scipy.spatial.distance import cdist


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))   # numerically stable


# --------------------------- binary response --------------------------------

def vote_probs(T, V, sigma, b, c, eps=0.03, kernel="logistic"):
    """P(approve) for every (voter, topic) pair.

    T: (X, d) topics | V: (N, d) voters | sigma, b: (N,) | c: (X,)
    Returns (N, X) matrix of probabilities in [eps, 1 - eps].
    """
    d2 = cdist(V, T, "sqeuclidean")                       # (N, X)
    k = 1.0 / (2.0 * sigma ** 2)                          # (N,)
    if kernel == "gaussian":                              # original design
        core = np.exp(-d2 * k[:, None])
    else:                                                 # improved design
        core = sigmoid(b[:, None] + c[None, :] - d2 * k[:, None])
    return eps + (1.0 - 2.0 * eps) * core


def sample_votes(P, rng):
    return (rng.random(P.shape) < P).astype(np.int8)


# --------------------------- ordinal response --------------------------------

def _ordered_thresholds(th_free):
    """Reconstruct ordered thresholds from free parameterisation.

    th_free[0] = th_0 (base), th_free[1:] = log(delta) for positive diffs.
    Returns (K-1,) array with th_0 < th_1 < ... < th_{K-2}.
    """
    K = len(th_free)
    th = np.zeros(K)
    th[0] = th_free[0]
    cum = th_free[0]
    for j in range(1, K):
        cum += np.exp(th_free[j])
        th[j] = cum
    return th


def vote_probs_ordinal(T, V, sigma, b, c, thresholds, eps=0.03):
    """Ordinal response probabilities for a K-category Likert item.

    Args:
        T: (X, d) topic positions.
        V: (N, d) voter ideal points.
        sigma: (N,) voter tolerances.
        b: (N,) voter acquiescence biases.
        c: (X,) topic valences.
        thresholds: (K-1,) array of ascending cutpoints.
        eps: lapse/guess rate (shared across categories).

    Returns:
        (N, X, K) array where sum over K = 1 for each (voter, topic).
    """
    d2 = cdist(V, T, "sqeuclidean")                     # (N, X)
    k = 1.0 / (2.0 * sigma ** 2)
    logit = b[:, None] + c[None, :] - d2 * k[:, None]  # (N, X)
    K = len(thresholds) + 1

    # cumulative probabilities: P(response <= k) = sigmoid(th_k - logit)
    cum = np.zeros((V.shape[0], T.shape[0], K - 1))
    for kk in range(K - 1):
        cum[:, :, kk] = sigmoid(thresholds[kk] - logit)

    # category probabilities
    probs = np.zeros((V.shape[0], T.shape[0], K))
    probs[:, :, 0] = cum[:, :, 0]
    for kk in range(1, K - 1):
        probs[:, :, kk] = cum[:, :, kk] - cum[:, :, kk - 1]
    probs[:, :, K - 1] = 1.0 - cum[:, :, -1]

    # lapse-rate floor across all categories
    return eps / K + (1 - eps) * probs


def sample_ordinal(P, rng):
    """Sample ordinal categories from (N, X, K) probability tensor."""
    N, X, K = P.shape
    flat = rng.random((N, X))
    cum = np.cumsum(P, axis=-1)
    for k in range(K):
        if k == 0:
            sel = flat < cum[:, :, 0]
        else:
            sel = (flat >= cum[:, :, k - 1]) & (flat < cum[:, :, k])
        P[sel] = k
    return P.astype(np.int8)[:, :, 0]  # (N, X) integer codes