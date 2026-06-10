#!/usr/bin/env python3
"""Approval-rate sweep.

Vary mu_log_sigma to change the base approval rate and observe how recovery
quality and dimensionality selection change.

Usage:
    python run_approval_sweep.py [--seed 0] [--n_voters 2000]

Figures:
    figures/fig_approval_recovery.png   — RMSE and dist-Spearman vs approval rate
    figures/fig_approval_dimselect.png  — selected dimension vs approval rate
"""
import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt

from votesim import SimConfig, make_world, vote_probs, sample_votes
from votesim.metrics import aligned_rmse, distance_spearman, mean_loglik
from votesim.recovery import recover_pca, recover_mds, mle_fit, _validation_split

os.makedirs("figures", exist_ok=True)

parser = argparse.ArgumentParser(description="Sweep approval rate")
parser.add_argument("--seed", type=int, default=0, help="fixed seed")
parser.add_argument("--n_voters", type=int, default=2000, help="voters per run")
args = parser.parse_args()

# mu_log_sigma controls median voter tolerance:
#   lower mu  -> smaller sigma  -> voters need closer topics -> lower approval rate
#   higher mu -> larger sigma   -> voters approve more broadly  -> higher approval rate
mu_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

dim_candidates = [1, 2, 3, 4, 5]
records = []

print(f"{'mu_log_sigma':>12s} {'approval':>8s} {'RMSE_pca':>8s} {'RMSE_mds':>8s} "
      f"{'RMSE_mle':>8s} {'ds_mds':>8s} {'ds_mle':>8s} {'best_dim':>8s}")

for mu in mu_vals:
    cfg = SimConfig(alpha=2, n_topics=60, n_voters=args.n_voters,
                    mu_log_sigma=mu, tau=0.4,
                    kernel="logistic", eps=0.03, seed=args.seed)
    world = make_world(cfg)
    P_true = vote_probs(world["T"], world["V"], world["sigma"],
                        world["b"], world["c"], cfg.eps, cfg.kernel)
    Y = sample_votes(P_true, world["rng"])
    approval = float(Y.mean())

    # ---- recovery ----
    T_pca = recover_pca(Y, cfg.alpha)
    T_mds = recover_mds(Y, cfg.alpha)
    fit = mle_fit(Y, cfg.alpha, eps=cfg.eps, init_T=T_mds,
                  n_iter=3000, seed=args.seed)

    rmse_pca = aligned_rmse(T_pca, world["T"])
    rmse_mds = aligned_rmse(T_mds, world["T"])
    rmse_mle = aligned_rmse(fit["T"], world["T"])
    ds_mds = distance_spearman(T_mds, world["T"])
    ds_mle = distance_spearman(fit["T"], world["T"])

    # ---- dimensionality selection ----
    mask, val_mask = _validation_split(Y, holdout_frac=0.10, seed=args.seed)
    bll = {}
    for d in dim_candidates:
        init = recover_mds(Y, d)
        f = mle_fit(Y, d, mask=mask, eps=cfg.eps, n_iter=800,
                    val_mask=val_mask, init_T=init, seed=args.seed)
        Pd = np.clip(f["P"], 1e-12, 1 - 1e-12)
        bll[d] = float(
            (Y * np.log(Pd) + (1 - Y) * np.log(1 - Pd))[val_mask].mean())
    best_dim = max(bll, key=bll.get)

    print(f"{mu:12.1f} {approval:8.3f} {rmse_pca:8.4f} {rmse_mds:8.4f} "
          f"{rmse_mle:8.4f} {ds_mds:8.4f} {ds_mle:8.4f} {best_dim:8d}")

    records.append(dict(mu_log_sigma=mu, approval=approval,
                        rmse_pca=rmse_pca, rmse_mds=rmse_mds, rmse_mle=rmse_mle,
                        ds_mds=ds_mds, ds_mle=ds_mle,
                        selected_dim=best_dim,
                        heldout_ll={str(d): float(bll[d]) for d in dim_candidates}))

with open("approval_sweep_results.json", "w") as f:
    json.dump(records, f, indent=2)

# ---- figure: recovery metrics vs approval rate ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

approvals = np.array([r["approval"] for r in records])
for ax, metric, methods in [
    (axes[0], "RMSE", [("PCA", "rmse_pca", "tab:orange"),
                       ("MDS", "rmse_mds", "tab:green"),
                       ("MLE", "rmse_mle", "tab:red")]),
    (axes[1], "dist-Spearman", [("MDS", "ds_mds", "tab:green"),
                                 ("MLE", "ds_mle", "tab:red")])]:
    for label, key, color in methods:
        vals = [r[key] for r in records]
        ax.plot(approvals, vals, "o-", color=color, label=label)
    ax.set_xlabel("approval rate")
    ax.set_ylabel(metric)
    ax.legend()

fig.tight_layout()
fig.savefig("figures/fig_approval_recovery.png", dpi=150, bbox_inches="tight")
print("\nFigure: figures/fig_approval_recovery.png saved")

# ---- figure: dim selection vs approval rate ----
fig, ax = plt.subplots(figsize=(6, 4))
dims_selected = [r["selected_dim"] for r in records]
ax.plot(approvals, dims_selected, "o-", color="tab:blue")
ax.axhline(2, color="gray", ls="--", label="true dim")
ax.set_xlabel("approval rate")
ax.set_ylabel("selected dimension")
ax.set_yticks(dim_candidates)
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig_approval_dimselect.png", dpi=150, bbox_inches="tight")
print("Figure: figures/fig_approval_dimselect.png saved")