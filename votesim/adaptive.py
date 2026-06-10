"""Phase 2: adaptive voter localization.

Particle approximation of p(v, log sigma, b | answers) with three question
policies: random, uncertainty sampling, expected information gain (EIG/BALD).
"""
import numpy as np

from .votes import vote_probs


def _entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


class ParticleFilter:
    """Posterior over a single voter's latent parameters given a topic map."""

    def __init__(self, T, c, prior, n_particles=3000, eps=0.03,
                 kernel="logistic", rng=None):
        self.T, self.c, self.eps, self.kernel = T, c, eps, kernel
        self.rng = rng or np.random.default_rng()
        self.prior = prior
        P, d = n_particles, T.shape[1]
        self.v = self.rng.normal(0, prior["s_v"], (P, d))
        self.ls = self.rng.normal(prior["mu_log_sigma"], prior["tau"], P)
        self.ls = np.clip(self.ls, *np.log(prior["sigma_bounds"]))
        self.b = self.rng.normal(0, prior["sd_bias"], P) \
            if kernel == "logistic" else np.zeros(P)
        self.w = np.full(P, 1.0 / P)
        self._probs = None

    @property
    def probs(self):
        """(P, X) approval probability of every topic under every particle."""
        if self._probs is None:
            self._probs = vote_probs(self.T, self.v, np.exp(self.ls),
                                     self.b, self.c, self.eps, self.kernel)
        return self._probs

    def predictive(self):
        return self.w @ self.probs                    # (X,)

    def update(self, j, y):
        like = self.probs[:, j] if y else 1.0 - self.probs[:, j]
        self.w *= like
        tot = self.w.sum()
        if tot <= 0:                                  # pathological; reset
            self.w[:] = 1.0 / len(self.w)
        else:
            self.w /= tot
        if 1.0 / (self.w ** 2).sum() < 0.5 * len(self.w):
            self._resample()

    def _resample(self, jitter=0.2):
        P = len(self.w)
        pos = (self.rng.random() + np.arange(P)) / P
        idx = np.searchsorted(np.cumsum(self.w), pos)
        idx = np.clip(idx, 0, P - 1)
        for name in ("v", "ls", "b"):
            arr = getattr(self, name)[idx].copy()
            sd = arr.std(axis=0) if arr.ndim > 1 else arr.std()
            arr += self.rng.normal(size=arr.shape) * jitter * sd
            setattr(self, name, arr)
        self.ls = np.clip(self.ls, *np.log(self.prior["sigma_bounds"]))
        self.w = np.full(P, 1.0 / P)
        self._probs = None

    def mean_v(self):
        return self.w @ self.v

    def cov_trace(self):
        mu = self.mean_v()
        return float((self.w[:, None] * (self.v - mu) ** 2).sum())


# ------------------------------- policies -----------------------------------

def pick_random(pf, asked, rng):
    pool = np.setdiff1d(np.arange(pf.T.shape[0]), asked)
    return int(rng.choice(pool))


def pick_uncertainty(pf, asked, rng):
    pred = pf.predictive().copy()
    pred[asked] = np.inf                               # exclude asked
    return int(np.argmin(np.abs(pred - 0.5)))


def pick_eig(pf, asked, rng):
    pred = pf.predictive()                             # (X,)
    ig = _entropy(pred) - pf.w @ _entropy(pf.probs)    # marginal H - E[cond H]
    ig[asked] = -np.inf
    return int(np.argmax(ig))


POLICIES = {"random": pick_random,
            "uncertainty": pick_uncertainty,
            "EIG": pick_eig}


# ------------------------------ evaluation ----------------------------------

def run_session(T_model, c_model, prior, T_true, c_true,
                v_true, sigma_true, b_true, policy, n_questions,
                n_particles, eps, kernel, transform, rng):
    """One voter, one policy. Answers come from the TRUE world; the posterior
    only sees the model map. `transform` maps model space -> true space."""
    pf = ParticleFilter(T_model, c_model, prior, n_particles, eps, kernel, rng)
    p_row = vote_probs(T_true, v_true[None, :], np.array([sigma_true]),
                       np.array([b_true]), c_true, eps, kernel)[0]
    asked, errs = [], [np.linalg.norm(transform(pf.mean_v())[0] - v_true)]
    for _ in range(n_questions):
        j = policy(pf, asked, rng)
        y = int(rng.random() < p_row[j])
        pf.update(j, y)
        asked.append(j)
        errs.append(np.linalg.norm(transform(pf.mean_v())[0] - v_true))
    return np.array(errs)


def evaluate_policies(cfg, T_model, c_model, T_true, c_true, transform,
                      n_test_voters=100, n_questions=30, n_particles=3000,
                      seed=123):
    """Localization-error curves (n_voters, n_questions+1) per policy."""
    from .world import sample_voters
    rng = np.random.default_rng(seed)
    Vt, St, Bt = sample_voters(cfg, n_test_voters, rng)
    prior = dict(s_v=cfg.s_v, mu_log_sigma=cfg.mu_log_sigma, tau=cfg.tau,
                 sd_bias=cfg.sd_bias, sigma_bounds=cfg.sigma_bounds)
    results = {}
    for name, policy in POLICIES.items():
        curves = [run_session(T_model, c_model, prior, T_true, c_true,
                              Vt[i], St[i], Bt[i], policy, n_questions,
                              n_particles, cfg.eps, cfg.kernel, transform,
                              np.random.default_rng(seed + 7 * i + hash(name) % 1000))
                  for i in range(n_test_voters)]
        results[name] = np.stack(curves)
        print(f"  policy {name:12s} median final error = "
              f"{np.median(results[name][:, -1]):.3f}")
    return results
