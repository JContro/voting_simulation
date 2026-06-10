#!/usr/bin/env python3
"""Ordinal (Likert) response demo.

Generates ordinal 5-category responses, recovers topics with the ordinal
MLE, and evaluates adaptive questioning with ordinal-specific policies.

Usage:
    python run_ordinal_demo.py [--seed 1] [--n_voters 1000]

Figures:
    figures/fig_ordinal_recovery.png   — topic recovery scatter
    figures/fig_ordinal_adaptive.png   — adaptive localization curves
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

from votesim import SimConfig, make_world, vote_probs, sample_votes
from votesim.votes import vote_probs_ordinal, sample_ordinal
from votesim.metrics import procrustes_transform, aligned_rmse, distance_spearman
from votesim.recovery import recover_mds, mle_fit_ordinal, select_dimension
from votesim.adaptive import evaluate_policies

os.makedirs("figures", exist_ok=True)

parser = argparse.ArgumentParser(description="Ordinal response demo")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--n_voters", type=int, default=1000, help="calibration voters")
args = parser.parse_args()

# --- build the world (same latent space, ordinal response) ---
print("Generating ordinal world...")
cfg = SimConfig(alpha=2, n_topics=60, n_voters=args.n_voters,
                tau=0.4, kernel="logistic", eps=0.03, seed=args.seed)
world = make_world(cfg)

# Ordinal thresholds (centered, symmetric)
thresholds = np.array([-1.5, -0.5, 0.5, 1.5])
n_categories = len(thresholds) + 1

P_ord = vote_probs_ordinal(world["T"], world["V"], world["sigma"],
                           world["b"], world["c"], thresholds, cfg.eps)
Y_ord = sample_ordinal(P_ord, np.random.default_rng(cfg.seed + 100))
print(f"  Ordinal response shape: {Y_ord.shape}, categories: {n_categories}")
for k in range(n_categories):
    print(f"    category {k}: {(Y_ord == k).mean():.3f}")

# --- Phase 1: ordinal MLE ---
print("\n--- Phase 1: ordinal topic recovery ---")
T_mds = recover_mds(Y_ord, cfg.alpha)   # MDS on the ordinal matrix still works
fit = mle_fit_ordinal(Y_ord, cfg.alpha, n_categories=n_categories,
                      thresholds=thresholds, eps=cfg.eps,
                      init_T=T_mds, n_iter=2000, seed=cfg.seed, verbose=True)

rmse = aligned_rmse(fit["T"], world["T"])
ds = distance_spearman(fit["T"], world["T"])
print(f"\n  MLE (ordinal) RMSE = {rmse:.4f}, dist-Spearman = {ds:.4f}")

# figure 1: topic recovery
f = procrustes_transform(fit["T"], world["T"])
Ta = f(fit["T"])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(*world["T"][:, :2].T, c="k", s=30, label="true topics")
ax.scatter(*Ta[:, :2].T, c="tab:red", s=30, marker="x", label="ordinal MLE (aligned)")
for a, bpt in zip(world["T"], Ta):
    ax.plot([a[0], bpt[0]], [a[1], bpt[1]], "gray", lw=0.5, alpha=0.5)
ax.set_title("Topic recovery — ordinal response")
ax.legend()
fig.savefig("figures/fig_ordinal_recovery.png", dpi=150, bbox_inches="tight")
print("Figure: figures/fig_ordinal_recovery.png saved")

# --- Phase 2: adaptive questioning with ordinal responses ---
print("\n--- Phase 2: adaptive voter localization (ordinal) ---")


def identity(x):
    return np.atleast_2d(x)


to_true = procrustes_transform(fit["T"], world["T"])

# We need to pass ordinal config to evaluate_policies. It reads
# cfg.response_mode and cfg.thresholds, so set them:
cfg.response_mode = "ordinal"
cfg.thresholds = thresholds

print("  Oracle topic map:")
res_true = evaluate_policies(cfg, world["T"], world["c"],
                             world["T"], world["c"], identity,
                             n_test_voters=50, n_questions=20,
                             n_particles=2000)

print("  Recovered topic map:")
res_reco = evaluate_policies(cfg, fit["T"], fit["c"],
                             world["T"], world["c"], to_true,
                             n_test_voters=50, n_questions=20,
                             n_particles=2000)

fig, ax = plt.subplots(figsize=(7, 5))
colors = {"random": "tab:gray", "uncertainty": "tab:blue", "EIG": "tab:red"}
for res, ls, tag in [(res_true, "-", "true map"), (res_reco, "--", "recovered map")]:
    for name, E in res.items():
        med = np.median(E, axis=0)
        ax.plot(med, ls, color=colors[name], label=f"{name} ({tag})")
        if ls == "-":
            q1, q3 = np.percentile(E, [25, 75], axis=0)
            ax.fill_between(range(len(med)), q1, q3,
                            color=colors[name], alpha=0.12)
ax.set_xlabel("questions asked")
ax.set_ylabel("median localization error")
ax.set_title("Adaptive localization — ordinal responses")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig_ordinal_adaptive.png", dpi=150, bbox_inches="tight")
print("Figure: figures/fig_ordinal_adaptive.png saved")