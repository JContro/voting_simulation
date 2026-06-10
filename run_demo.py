#!/usr/bin/env python3
"""Full pipeline demo: world -> phase-1 recovery -> phase-2 adaptive sampling.

Usage:
    python run_demo.py [--seed N] [--population blob|polarized]

Runs in ~2-4 minutes on a laptop. Figures land in ./figures/.
For paper-grade numbers, use run_sweep.py (sweeps over 20+ seeds).
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from votesim import SimConfig, make_world, vote_probs, sample_votes
from votesim.metrics import (procrustes_transform, aligned_rmse,
                             distance_spearman, mean_loglik)
from votesim.recovery import recover_pca, recover_mds, mle_fit, select_dimension
from votesim.adaptive import evaluate_policies

os.makedirs("figures", exist_ok=True)

# --- CLI ---
parser = argparse.ArgumentParser(description="Spatial voting simulation demo")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--population", type=str, default="blob",
                    choices=["blob", "polarized", "clustered"],
                    help="voter population geometry")
args = parser.parse_args()
print(f"run_demo.py  seed={args.seed}  population={args.population}")

# ============================ build the world ================================
cfg = SimConfig(alpha=2, n_topics=60, n_voters=2000, tau=0.4,
                kernel="logistic", eps=0.03, seed=args.seed,
                population=args.population)
world = make_world(cfg)
P_true = vote_probs(world["T"], world["V"], world["sigma"],
                    world["b"], world["c"], cfg.eps, cfg.kernel)
Y = sample_votes(P_true, world["rng"])
approval = Y.mean()
print(f"world: alpha={cfg.alpha}, X={cfg.n_topics}, N={cfg.n_voters}, "
      f"approval rate = {approval:.3f}, population={cfg.population}")

# ====================== PHASE 1: topic recovery ==============================
print("\n--- Phase 1: topic recovery ---")
T_pca = recover_pca(Y, cfg.alpha)
T_mds = recover_mds(Y, cfg.alpha)
fit = mle_fit(Y, cfg.alpha, eps=cfg.eps, init_T=T_mds, n_iter=3000,
              seed=cfg.seed, verbose=True)

oracle_ll = mean_loglik(Y, P_true)
print(f"\n{'method':8s} {'RMSE(aligned)':>14s} {'dist-Spearman':>14s} {'LL/entry':>10s}")
for name, Th in [("PCA", T_pca), ("MDS", T_mds), ("MLE", fit["T"])]:
    ll = f"{mean_loglik(Y, fit['P']):10.4f}" if name == "MLE" else " " * 10
    print(f"{name:8s} {aligned_rmse(Th, world['T']):14.4f} "
          f"{distance_spearman(Th, world['T']):14.4f} {ll}")
print(f"{'oracle':8s} {'':14s} {'':14s} {oracle_ll:10.4f}   <- noise floor")

# figure 1: true vs aligned recovered topic map
f = procrustes_transform(fit["T"], world["T"])
Ta = f(fit["T"])
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(*world["T"][:, :2].T, c="k", s=30, label="true topics")
ax.scatter(*Ta[:, :2].T, c="tab:red", s=30, marker="x", label="MLE (aligned)")
for a, bpt in zip(world["T"], Ta):
    ax.plot([a[0], bpt[0]], [a[1], bpt[1]], "gray", lw=0.5, alpha=0.5)
ax.set_title("Topic recovery (Procrustes-aligned)")
ax.legend()
fig.savefig("figures/fig1_topic_recovery.png", dpi=150, bbox_inches="tight")

# ===================== dimensionality selection ==============================
print("\n--- Dimensionality selection (held-out log-likelihood) ---")
dims = [1, 2, 3, 4, 5]
ll_by_dim, _ = select_dimension(Y, dims, eps=cfg.eps, seed=cfg.seed)
best = max(ll_by_dim, key=ll_by_dim.get)
print(f"selected dim = {best} (true = {cfg.alpha})")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(dims, [ll_by_dim[d] for d in dims], "o-")
ax.axvline(cfg.alpha, color="gray", ls="--", label="true dim")
ax.set_xlabel("fitted dimension")
ax.set_ylabel("held-out mean log-likelihood")
ax.legend()
fig.savefig("figures/fig2_dim_selection.png", dpi=150, bbox_inches="tight")

# ================= PHASE 2: adaptive voter localization ======================


def identity(x):
    return np.atleast_2d(x)


to_true = procrustes_transform(fit["T"], world["T"])

print("\n--- Phase 2 (oracle topic map) ---")
res_true = evaluate_policies(cfg, world["T"], world["c"],
                             world["T"], world["c"], identity)
print("\n--- Phase 2 (recovered topic map, end-to-end) ---")
res_reco = evaluate_policies(cfg, fit["T"], fit["c"],
                             world["T"], world["c"], to_true)

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
ax.set_ylabel("median localization error  $\\|\\hat{v} - v\\|$")
ax.set_title("Adaptive voter localization")
ax.legend(fontsize=8)
fig.savefig("figures/fig3_adaptive_policies.png", dpi=150, bbox_inches="tight")
print("\nFigures saved to ./figures/")

# ====================== polarized population plot ============================
if args.population in ("polarized", "clustered"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*world["V"][::10, :2].T, c=world["sigma"][::10],
               s=10, cmap="viridis", alpha=0.6)
    ax.scatter(*world["T"][:, :2].T, c="k", s=40, marker="s",
               label="topics")
    ax.set_title(f"Voters by tolerance ({cfg.population} population)")
    cbar = fig.colorbar(ax.collections[0], ax=ax, label="tolerance $\\sigma$")
    fig.savefig("figures/fig4_population.png", dpi=150, bbox_inches="tight")
    print("Figure: fig4_population.png saved")