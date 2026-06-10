"""Weights & Biases logging utilities for voting-simulation experiments.

Gracefully degrades: if the API key is absent or the import fails,
all public methods become no-ops so consuming code needs no guards.
"""
from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── guard: try to import wandb ────────────────────────────────────────────────
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]


def _resolve_api_key() -> str | None:
    """Return the API key from the environment, or None.

    Checks both the canonical ``WANDB_API_KEY`` and the misspelled
    ``WAND_API_KEY`` (:file:`.env`) as a fallback.
    """
    return os.environ.get("WANDB_API_KEY") or os.environ.get("WAND_API_KEY")


def _is_enabled() -> bool:
    """Return True if wandb can be used (imported + key present)."""
    if not _WANDB_AVAILABLE:
        logger.debug("wandb not installed — skipping logging")
        return False
    if not _resolve_api_key():
        logger.info("WANDB_API_KEY not found in environment — skipping wandb logging")
        return False
    return True


# ── public API ────────────────────────────────────────────────────────────────

def init_run(
    project: str = "voting-simulation",
    config: dict[str, Any] | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
) -> tuple[bool, str | None]:
    """Initialise a wandb run.

    Parameters
    ----------
    project : str
        W&B project name.
    config : dict or None
        Hyperparameters / experiment config to save with the run.
    notes : str or None
        Free-text description of the run.
    tags : list[str] or None
        Tags for organising runs in the W&B UI.

    Returns
    -------
    (success, run_url)
        ``(True, "https://wandb.ai/...")`` on success,
        ``(False, None)`` when wandb is unavailable / skipped.
    """
    if not _is_enabled():
        return False, None

    run = wandb.init(  # type: ignore[union-attr]
        project=project,
        config=config,
        notes=notes,
        tags=tags,
    )
    if run is None:
        logger.warning("wandb.init() returned None — check your API key / network")
        return False, None

    try:
        url = wandb.run.get_url()  # type: ignore[union-attr]
    except Exception:
        url = f"https://wandb.ai/home?project={project}"
    print(f"  └─ W&B project : {project}")
    print(f"  └─ W&B run URL : {url}")
    return True, url


def log_metrics(
    data: dict[str, Any],
    step: int | None = None,
) -> None:
    """Log a dictionary of metrics to the current wandb run.

    Safe to call even when wandb is disabled (no-op).
    """
    if wandb is None or not wandb.run:
        return
    wandb.log(data, step=step)  # type: ignore[union-attr]


def log_figure(
    path: str,
    caption: str | None = None,
) -> None:
    """Upload a figure file as a wandb Artifact.

    Parameters
    ----------
    path : str
        Path to the figure file on disk.
    caption : str or None
        Optional caption stored as artifact description.
    """
    if wandb is None or not wandb.run:
        return
    if not os.path.isfile(path):
        logger.warning("Figure not found, skipping upload: %s", path)
        return

    name = os.path.splitext(os.path.basename(path))[0].replace("_", "-")
    artifact = wandb.Artifact(  # type: ignore[union-attr]
        name=name,
        type="figure",
        description=caption or "",
    )
    artifact.add_file(path)
    wandb.run.log_artifact(artifact)  # type: ignore[union-attr]


def log_summary(metrics: dict[str, Any]) -> None:
    """Set wandb run summary fields (displayed on the run overview page)."""
    if wandb is None or not wandb.run:
        return
    for key, value in metrics.items():
        wandb.run.summary[key] = value  # type: ignore[union-attr]


def finish_run(exit_code: int = 0) -> None:
    """Mark the current wandb run as finished."""
    if wandb is None or not wandb.run:
        return
    if exit_code != 0:
        wandb.run.finish(exit_code=exit_code)  # type: ignore[union-attr]
    else:
        wandb.run.finish()  # type: ignore[union-attr]