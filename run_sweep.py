#!/usr/bin/env python3
"""Multi-seed experiment sweep.

Runs the full phase-1 topic-recovery pipeline over seeds 0 .. n_seeds-1
and aggregates metrics with median + IQR bands. Logs everything to Weights &
Biases (if ``WANDB_API_KEY`` is set in the environment or :file:`.env`).

Usage:
    python run_sweep.py [--n_seeds 20] [--n_voters 2000]

Figures:
    figures/fig_sweep_recovery.png   — RMSE + dist-Spearman by method
    figures/fig_sweep_dimselect.png  — frequency of selected dimension

Output:
    sweep_results.json  — raw per-seed data for further analysis
"""
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from votesim import SimConfig, make_world, vote_probs, sample_votes
from votesim.metrics import aligned_rmse, distance_spearman, mean_loglik
from votesim.recovery import recover_pca, recover_mds, mle_fit
from votesim.wandb_logger import (
    init_run,
    log_metrics,
    log_figure,
    log_summary,
    finish_run,
)

# ── load .env for WANDB_API_KEY (docker-compose may not pass it automatically) ──
env_path = Path(__file__).with_name(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.makedirs("figures", exist_ok=True)

parser = argparse.ArgumentParser(description="Multi-seed experiment sweep")
parser.add_argument("--n_seeds", type=int, default=20, help="number of seeds")
parser.add_argument("--n_voters", type=int, default=2000, help="voters per seed")
parser.add_argument("--wandb_project", type=str, default="voting-simulation",
                    help="W&B project name (default: voting-simulation)")
parser.add_argument("--wandb_disabled", action="store_true",
                    help="Skip W&B logging even if API key is present")
args = parser.parse_args()

# ── shared config (used for both simulation and W&B run config) ──
SWEEP_CONFIG = dict(
    n_seeds=args.n_seeds,
    n_voters=args.n_voters,
    alpha=2,
    n_topics=60,
    tau=0.4,
    kernel="logistic",
    eps=0.03,
    dim_search=[1, 2, 3, 4, 5],
)

# ── W&B initialisation ──
_wandb_active = False
_wandb_url = None
if not args.wandb_disabled:
    _wandb_active, _wandb_url = init_run(
        project=args.wandb_project,
        config=SWEEP_CONFIG,
        notes=f"sweep over {args.n_seeds} seeds, {args.n_voters} voters each",
        tags=["sweep", f"n_seeds={args.n_seeds}"],
    )
if _wandb_active:
    print(f"W&B logging active — {_wandb_url}")
else:
    print("W&B logging skipped (set WANDB_API_KEY in .env to enable)")

dims_to_try = [1, 2, 3, 4, 5]

records = []

for s in range(args.n_seeds):
    if (s + 1) % 5 == 0:
        print(f"\n--- seed {s + 1}/{args.n_seeds} ---")
    else:
        print(f"  seed {s + 1}/{args.n_seeds}")

    cfg = SimConfig(alpha=2, n_topics=60, n_voters=args.n_voters,
                    tau=0.4, kernel="logistic", eps=0.03, seed=s)
    world = make_world(cfg)
    P_true = vote_probs(world["T"], world["V"], world["sigma"],
                        world["b"], world["c"], cfg.eps, cfg.kernel)
    Y = sample_votes(P_true, world["rng"])
    approval = float(Y.mean())

    # ---- recovery ----
    T_pca = recover_pca(Y, cfg.alpha)
    T_mds = recover_mds(Y, cfg.alpha)
    fit = mle_fit(Y, cfg.alpha, eps=cfg.eps, init_T=T_mds, n_iter=3000, seed=s)

    rmse_pca = aligned_rmse(T_pca, world["T"])
    rmse_mds = aligned_rmse(T_mds, world["T"])
    rmse_mle = aligned_rmse(fit["T"], world["T"])
    ds_pca = distance_spearman(T_pca, world["T"])
    ds_mds = distance_spearman(T_mds, world["T"])
    ds_mle = distance_spearman(fit["T"], world["T"])
    train_ll = mean_loglik(Y, fit["P"])
    oracle_ll = mean_loglik(Y, P_true)

    # ---- dimensionality selection ----
    from votesim.recovery import _validation_split
    mask, val_mask = _validation_split(Y, holdout_frac=0.10, seed=s)
    bll = {}
    for d in dims_to_try:
        init = recover_mds(Y, d)
        f = mle_fit(Y, d, mask=mask, eps=cfg.eps, n_iter=800,
                    val_mask=val_mask, init_T=init, seed=s)
        P = f["P"]
        P = np.clip(P, 1e-12, 1 - 1e-12)
        bll[d] = float(
            (Y * np.log(P) + (1 - Y) * np.log(1 - P))[val_mask].mean())
    best_dim = max(bll, key=bll.get)

    records.append(dict(
        seed=s, approval=approval,
        rmse_pca=rmse_pca, rmse_mds=rmse_mds, rmse_mle=rmse_mle,
        ds_pca=ds_pca, ds_mds=ds_mds, ds_mle=ds_mle,
        train_ll=train_ll, oracle_ll=oracle_ll,
        selected_dim=best_dim,
        heldout_ll={str(d): float(bll[d]) for d in dims_to_try},
    ))

    # ── per-seed W&B logging ──
    log_metrics({
        "seed": s,
        "approval_rate": approval,
        "rmse_pca": rmse_pca,
        "rmse_mds": rmse_mds,
        "rmse_mle": rmse_mle,
        "dist_spearman_pca": ds_pca,
        "dist_spearman_mds": ds_mds,
        "dist_spearman_mle": ds_mle,
        "train_loglik": train_ll,
        "oracle_loglik": oracle_ll,
        "selected_dim": best_dim,
        **{f"heldout_ll_dim{d}": float(bll[d]) for d in dims_to_try},
    }, step=s)

# ---- aggregate ----
def median_iqr(arr):
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))

print("\n" + "=" * 60)
print(f"Aggregated over {args.n_seeds} seeds:")
print(f"{'metric':20s} {'median':>8s} {'[Q1':>8s} {'Q3]':>8s}")
print("-" * 60)
for key in ["rmse_pca", "rmse_mds", "rmse_mle",
            "ds_pca", "ds_mds", "ds_mle", "approval"]:
    vals = np.array([r[key] for r in records])
    med, q1, q3 = median_iqr(vals)
    print(f"{key:20s} {med:8.4f} {q1:8.4f} {q3:8.4f}")

# ---- dim selection frequency ----
dim_counts = np.bincount([r["selected_dim"] for r in records],
                         minlength=6)
print(f"\nDimensionality selection frequency (true dim=2):")
for d in range(1, 6):
    freq = dim_counts[d] / len(records)
    print(f"  dim {d}: {freq * 100:5.1f}%  ({dim_counts[d]}/{len(records)})")

# ---- save raw data ----
with open("sweep_results.json", "w") as f:
    json.dump(records, f, indent=2)
print(f"\nRaw data saved to sweep_results.json")

# ---- aggregate metrics logged to W&B ----
if _wandb_active:
    agg = {}
    for key in ["rmse_pca", "rmse_mds", "rmse_mle",
                "ds_pca", "ds_mds", "ds_mle", "approval"]:
        vals = np.array([r[key] for r in records])
        med, q1, q3 = median_iqr(vals)
        agg[f"{key}_median"] = med
        agg[f"{key}_q1"] = q1
        agg[f"{key}_q3"] = q3
    agg["dim_select_accuracy"] = dim_counts[2] / len(records)
    log_summary(agg)
    log_metrics({**agg, "stage": "aggregate"}, step=args.n_seeds)

# ---- figure: recovery metrics ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# RMSE panel
rmse_data = [np.array([r["rmse_pca"] for r in records]),
             np.array([r["rmse_mds"] for r in records]),
             np.array([r["rmse_mle"] for r in records])]
bp1 = axes[0].boxplot(rmse_data, tick_labels=["PCA", "MDS", "MLE"],
                       widths=0.5, patch_artist=True)
for patch, color in zip(bp1["boxes"], ["tab:orange", "tab:green", "tab:red"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for i, d in enumerate(rmse_data):
    axes[0].scatter(np.full_like(d, i + 1) + 0.05 * (np.random.rand(len(d)) - 0.5),
                    d, s=12, alpha=0.4, color="k")
axes[0].set_ylabel("aligned RMSE")
axes[0].set_title("Topic recovery error")
axes[0].grid(axis="y", alpha=0.3)

# dist-Spearman panel
ds_data = [np.array([r["ds_pca"] for r in records]),
           np.array([r["ds_mds"] for r in records]),
           np.array([r["ds_mle"] for r in records])]
bp2 = axes[1].boxplot(ds_data, tick_labels=["PCA", "MDS", "MLE"],
                       widths=0.5, patch_artist=True)
for patch, color in zip(bp2["boxes"], ["tab:orange", "tab:green", "tab:red"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for i, d in enumerate(ds_data):
    axes[1].scatter(np.full_like(d, i + 1) + 0.05 * (np.random.rand(len(d)) - 0.5),
                    d, s=12, alpha=0.4, color="k")
axes[1].set_ylabel("distance Spearman $\\rho$")
axes[1].set_title("Structure recovery")
axes[1].grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("figures/fig_sweep_recovery.png", dpi=150, bbox_inches="tight")
print("Figure: figures/fig_sweep_recovery.png saved")
if _wandb_active:
    log_figure("figures/fig_sweep_recovery.png",
               caption="Topic recovery RMSE and distance-Spearman per method")

# ---- figure: dim selection frequency ----
fig, ax = plt.subplots(figsize=(6, 4))
dims_arr = np.arange(1, 6)
freqs = np.array([dim_counts[d] / len(records) for d in dims_arr])
ax.bar(dims_arr, freqs, width=0.5, color="tab:blue", alpha=0.7)
ax.axvline(2, color="gray", ls="--", label="true dim")
ax.set_xlabel("selected dimension")
ax.set_ylabel("frequency")
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig_sweep_dimselect.png", dpi=150, bbox_inches="tight")
print("Figure: figures/fig_sweep_dimselect.png saved")

# ── close W&B run ──
finish_run()