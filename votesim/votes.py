"""Response model: approval probabilities and Bernoulli vote sampling."""
import numpy as np
from scipy.spatial.distance import cdist


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))   # numerically stable


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
