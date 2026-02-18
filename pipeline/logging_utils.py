"""
Dictionary-based logging for the Chroma + OpenFold pipeline.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

METRIC_KEYS = ["dist_diff", "plddt_mean", "pae_mean", "cmap_ent_mean", "tm_score", "mpnn_ce_mean", "mpnn_ent_mean"]
BEST_CRITERIA = ["mean_plddt", "dist_diff"]  # mpnn_ce, mpnn_ent omitted (not implemented)


def _metrics_to_scalars(metrics: dict) -> dict:
    """Convert metrics dict to JSON-serializable scalars."""
    out = {}
    for k, v in metrics.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif hasattr(v, "item"):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


def append_metrics(all_metrics: List[dict], metrics: dict) -> None:
    """Append a metrics dict to the list (in-place)."""
    all_metrics.append(_metrics_to_scalars(metrics))


def update_best(
    best: Dict[str, Optional[dict]],
    metrics: dict,
    exp_num: int,
    iter_num: int,
    pdb_path: str,
) -> None:
    """
    Update best-structure tracking by criterion.

    best[criterion] = {"exp": int, "iter": int, "path": str, "metrics": dict}
    """
    plddt = metrics.get("plddt_mean") or 0.0
    dist_diff = metrics.get("dist_diff")
    if dist_diff is None:
        dist_diff = float("inf")
    mpnn_ce = metrics.get("mpnn_ce_mean")
    mpnn_ent = metrics.get("mpnn_ent_mean")

    entry = {
        "exp": exp_num,
        "iter": iter_num,
        "path": pdb_path,
        "metrics": _metrics_to_scalars(metrics),
    }

    # mean_plddt: higher is better
    prev_plddt = best.get("mean_plddt", {}).get("metrics", {}).get("plddt_mean", 0.0) if best.get("mean_plddt") else 0.0
    if plddt > prev_plddt:
        best["mean_plddt"] = entry

    # dist_diff: lower is better
    prev_dist = best.get("dist_diff", {}).get("metrics", {}).get("dist_diff", float("inf")) if best.get("dist_diff") else float("inf")
    if dist_diff < prev_dist:
        best["dist_diff"] = entry

    # mpnn_ce, mpnn_ent: NOT IMPLEMENTED - placeholder only
    if mpnn_ce is not None:
        prev_ce = best.get("mpnn_ce", {}).get("metrics", {}).get("mpnn_ce_mean")
        if prev_ce is None or mpnn_ce < prev_ce:
            best["mpnn_ce"] = entry
    if mpnn_ent is not None:
        prev_ent = best.get("mpnn_ent", {}).get("metrics", {}).get("mpnn_ent_mean")
        if prev_ent is None or mpnn_ent < prev_ent:
            best["mpnn_ent"] = entry


def save_plots(
    save_dir: str,
    metrics_list: List[dict],
    exp_name: str,
    metric_keys: Optional[List[str]] = None,
) -> None:
    """Save metric plots (no plt.show for headless)."""
    if metric_keys is None:
        metric_keys = [k for k in METRIC_KEYS if any(m.get(k) is not None for m in metrics_list)]
    os.makedirs(save_dir, exist_ok=True)
    for key in metric_keys:
        values = [m.get(key) for m in metrics_list if m.get(key) is not None]
        if not values:
            continue
        plt.figure()
        plt.plot(range(len(values)), values)
        plt.title(f"{exp_name} {key}")
        plt.xlabel("iteration")
        out_path = os.path.join(save_dir, f"{exp_name}_{key}_fig.jpg")
        plt.savefig(out_path)
        plt.close()
        logger.info("Saved plot: %s", out_path)


def save_metrics_json(metrics_list: List[dict], path: str) -> None:
    """Save metrics list to JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    logger.info("Saved metrics to %s", path)


def log_best(best: Dict[str, Optional[dict]]) -> None:
    """Log best structures per criterion."""
    for criterion, entry in best.items():
        if entry is None:
            continue
        m = entry["metrics"]
        logger.info(
            "%s: best_exp=%s best_iter=%s path=%s dist_diff=%.3g plddt_mean=%.3g tm_score=%.3g",
            criterion,
            entry["exp"],
            entry["iter"],
            entry["path"],
            m.get("dist_diff", 0),
            m.get("plddt_mean", 0),
            m.get("tm_score", 0),
        )
