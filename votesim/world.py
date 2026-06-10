"""Generative engine: sample topics, voters, tolerances, biases."""
import numpy as np

from .config import SimConfig


def gauge_normalize(T: np.ndarray) -> np.ndarray:
    """Pin the gauge: topic centroid at origin, unit RMS norm."""
    T = T - T.mean(axis=0, keepdims=True)
    rms = np.sqrt((T ** 2).sum(axis=1).mean())
    return T / max(rms, 1e-12)


def _mixture_params(cfg: SimConfig):
    """Cluster means/scales drawn from a *separate* deterministic stream so that
    phase-2 test voters come from the same population as phase-1 voters."""
    rng = np.random.default_rng(cfg.seed + 99_991)
    if cfg.population == "clustered":
        means = rng.normal(0.0, 1.2 * cfg.s_v,
                           size=(cfg.n_clusters, cfg.alpha))
        scales = np.full(cfg.n_clusters, 0.4 * cfg.s_v)
        weights = rng.dirichlet(np.full(cfg.n_clusters, 5.0))
    elif cfg.population == "polarized":
        direction = rng.normal(size=cfg.alpha)
        direction /= np.linalg.norm(direction)
        means = np.stack(
            [1.2 * cfg.s_v * direction, -1.2 * cfg.s_v * direction])
        scales = np.full(2, 0.35 * cfg.s_v)
        weights = np.array([0.5, 0.5])
    else:  # blob
        means = np.zeros((1, cfg.alpha))
        scales = np.array([cfg.s_v])
        weights = np.array([1.0])
    return means, scales, weights


def sample_voters(cfg: SimConfig, n: int, rng: np.random.Generator):
    """Sample voter ideal points, tolerances, and acquiescence biases."""
    means, scales, weights = _mixture_params(cfg)
    z = rng.choice(len(weights), size=n, p=weights)
    V = means[z] + rng.normal(size=(n, cfg.alpha)) * scales[z][:, None]
    sigma = np.clip(
        rng.lognormal(cfg.mu_log_sigma, cfg.tau, size=n), *cfg.sigma_bounds
    )
    if cfg.kernel == "logistic":
        b = rng.normal(0.0, cfg.sd_bias, size=n)
    else:
        b = np.zeros(n)
    return V, sigma, b


def make_world(cfg: SimConfig) -> dict:
    """Build the full ground-truth world."""
    rng = np.random.default_rng(cfg.seed)
    T = gauge_normalize(rng.normal(size=(cfg.n_topics, cfg.alpha)))
    V, sigma, b = sample_voters(cfg, cfg.n_voters, rng)
    if cfg.kernel == "logistic":
        c = rng.normal(0.0, cfg.sd_valence, size=cfg.n_topics)
    else:
        c = np.zeros(cfg.n_topics)
    return dict(cfg=cfg, rng=rng, T=T, V=V, sigma=sigma, b=b, c=c)
