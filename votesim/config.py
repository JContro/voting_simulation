"""Experiment configuration."""
from dataclasses import dataclass


@dataclass
class SimConfig:
    # --- latent space ---
    alpha: int = 2                 # true latent dimensionality
    n_topics: int = 60             # X
    n_voters: int = 2000           # N (phase-1 calibration voters)

    # --- voter tolerance sigma_i ~ LogNormal(mu, tau), clipped ---
    mu_log_sigma: float = 0.0      # typical tolerance (log scale)
    tau: float = 0.4               # population heterogeneity  <-- headline knob
    sigma_bounds: tuple = (0.2, 5.0)

    # --- voter population geometry ---
    s_v: float = 1.0               # voter cloud scale
    population: str = "blob"       # "blob" | "clustered" | "polarized"
    n_clusters: int = 3            # used if population == "clustered"

    # --- response model ---
    # "logistic" (improved) | "gaussian" (original)
    kernel: str = "logistic"
    eps: float = 0.03              # lapse / guess rate
    # sd of voter acquiescence bias b_i (logistic only)
    sd_bias: float = 0.5
    sd_valence: float = 0.5        # sd of topic valence c_j (logistic only)

    seed: int = 0
