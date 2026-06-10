"""Gauge-aware evaluation: Procrustes alignment, distance correlation, log-lik."""
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def procrustes_transform(A, B):
    """Optimal similarity transform f (rotation+scale+translation) mapping A -> B.
    Returns a callable applicable to any points living in A's space."""
    muA, muB = A.mean(0), B.mean(0)
    A0, B0 = A - muA, B - muB
    R, raw_scale = orthogonal_procrustes(A0, B0)
    s = raw_scale / max((A0 ** 2).sum(), 1e-12)

    def f(X):
        return s * ((np.atleast_2d(X) - muA) @ R) + muB

    return f


def aligned_rmse(T_hat, T_true):
    f = procrustes_transform(T_hat, T_true)
    return float(np.sqrt(((f(T_hat) - T_true) ** 2).sum(1).mean()))


def distance_spearman(T_hat, T_true):
    """Gauge-free: rank correlation of the two topic-distance matrices."""
    return float(spearmanr(pdist(T_hat), pdist(T_true)).statistic)


def mean_loglik(Y, P, mask=None):
    """Mean Bernoulli log-likelihood; mask=boolean selects entries to score."""
    P = np.clip(P, 1e-9, 1.0 - 1e-9)
    ll = Y * np.log(P) + (1 - Y) * np.log(1 - P)
    return float(ll.mean() if mask is None else ll[mask].mean())
