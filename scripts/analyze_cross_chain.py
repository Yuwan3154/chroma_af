#!/usr/bin/env python3
"""
Cross-chain analysis: summarize metrics across all experiments and chains.

For each chain (pdb_id):
- Find the structure with highest pLDDT across all trajectories and iterations
- Compute TM-score of that structure vs ground-truth
- Plot: highest pLDDT vs TM-score (one point per chain)
- Color by: has_high_plddt_low_tm (any example with plddt_mean > 70 and TM-score < 0.5)
- Save JSON with pLDDT, TM-score, iteration#, experiment# for each chain
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Add chroma_af to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.metrics import compute_tm_score


def load_af2rank_lookup(csv_path: str, pdb_root: str) -> dict:
    """pdb_id (lower) -> (chain, gt_path)."""
    import csv
    lookup = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            natives_rcsb = (row.get("natives_rcsb") or "").strip()
            if not natives_rcsb or "_" not in natives_rcsb:
                continue
            pdb_chain = natives_rcsb.rsplit("_", 1)
            rcsb_pdb_id = pdb_chain[0].lower()
            chain = pdb_chain[1]
            mid2 = rcsb_pdb_id[1:3] if len(rcsb_pdb_id) >= 4 else rcsb_pdb_id
            gt_path = os.path.join(pdb_root, mid2, f"{rcsb_pdb_id}.cif")
            lookup[rcsb_pdb_id] = (chain, gt_path)
            natives_frank = (row.get("natives_frank") or "").strip().lower()
            if natives_frank and natives_frank != rcsb_pdb_id:
                lookup[natives_frank] = (chain, gt_path)
    return lookup


def collect_chain_data(
    output_dir: str,
    lookup: dict,
    usalign_binary: str = "USalign",
) -> dict:
    """
    Collect per-chain data: best pLDDT structure and TM-score.

    Returns dict: pdb_id -> {
        "best_plddt": float,
        "best_tm_score": float,
        "best_iter": int,
        "best_exp": int,
        "best_path": str,
        "has_high_plddt_low_tm": bool,  # any example with plddt>70 and tm<0.5
        "all_points": [(plddt, tm_score, iter, exp), ...]
    }
    """
    results = {}
    exp_dirs = sorted(Path(output_dir).iterdir())
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        if "_exp_" not in name:
            continue
        parts = name.split("_exp_")
        if len(parts) != 2:
            continue
        pdb_id = parts[0].lower()
        exp_num = int(parts[1]) if parts[1].isdigit() else 0

        if pdb_id not in lookup:
            continue
        chain, gt_path = lookup[pdb_id]
        if not os.path.exists(gt_path):
            continue

        metrics_path = exp_dir / f"{name}_metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics_list = json.load(f)

        for m in metrics_list:
            iter_num = m.get("iter", 0)
            plddt = m.get("plddt_mean")
            tm_score = m.get("tm_score")
            if plddt is None:
                continue

            # Resolve structure path (af or ch; try both cif and pdb)
            candidates = [
                exp_dir / f"{name}_af_{iter_num:03d}.cif",
                exp_dir / f"{name}_af_{iter_num:03d}.pdb",
                exp_dir / f"{name}_ch_{iter_num:03d}.pdb",
            ]
            struct_path = next((p for p in candidates if p.exists()), None)
            if struct_path is None:
                continue

            # Compute TM-score if not in metrics
            if tm_score is None:
                tm_score = compute_tm_score(str(struct_path), gt_path, usalign_binary=usalign_binary)

            if pdb_id not in results:
                results[pdb_id] = {
                    "best_plddt": None,
                    "best_tm_score": None,
                    "best_iter": None,
                    "best_exp": None,
                    "best_path": None,
                    "has_high_plddt_low_tm": False,
                    "all_points": [],
                }

            r = results[pdb_id]
            r["all_points"].append({
                "plddt_mean": plddt,
                "tm_score": tm_score,
                "iter": iter_num,
                "exp": exp_num,
                "path": str(struct_path),
            })

            if r["best_plddt"] is None or plddt > r["best_plddt"]:
                r["best_plddt"] = plddt
                r["best_tm_score"] = tm_score
                r["best_iter"] = iter_num
                r["best_exp"] = exp_num
                r["best_path"] = str(struct_path)

            if plddt > 70 and tm_score is not None and tm_score < 0.5:
                r["has_high_plddt_low_tm"] = True

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-chain analysis: pLDDT vs TM-score")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/chroma_af/results")
    parser.add_argument("--af2rank_csv", type=str, required=True)
    parser.add_argument("--af2rank_pdb_root", type=str, required=True)
    parser.add_argument("--usalign_binary", type=str, default="USalign")
    parser.add_argument("--plot_path", type=str, default=None, help="Path to save plot")
    parser.add_argument("--json_path", type=str, default=None, help="Path to save JSON summary")
    args = parser.parse_args()

    lookup = load_af2rank_lookup(args.af2rank_csv, args.af2rank_pdb_root)
    data = collect_chain_data(args.output_dir, lookup, usalign_binary=args.usalign_binary)

    # Build summary for JSON
    summary = {}
    for pdb_id, r in data.items():
        if r["best_plddt"] is None:
            continue
        summary[pdb_id] = {
            "plddt_mean": r["best_plddt"],
            "tm_score": r["best_tm_score"],
            "iter": r["best_iter"],
            "exp": r["best_exp"],
            "path": r["best_path"],
            "has_high_plddt_low_tm": r["has_high_plddt_low_tm"],
            "all_points": r["all_points"],
        }

    json_path = args.json_path or os.path.join(args.output_dir, "cross_chain_summary.json")
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON to {json_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plddts = []
        tm_scores = []
        colors = []
        labels = []
        for pdb_id, r in data.items():
            if r["best_plddt"] is None or r["best_tm_score"] is None:
                continue
            plddts.append(r["best_plddt"])
            tm_scores.append(r["best_tm_score"])
            colors.append("red" if r["has_high_plddt_low_tm"] else "blue")
            labels.append(pdb_id)

        plt.figure(figsize=(8, 6))
        plt.scatter(plddts, tm_scores, c=colors, alpha=0.7, s=50)
        for i, lbl in enumerate(labels):
            plt.annotate(lbl, (plddts[i], tm_scores[i]), fontsize=8, alpha=0.8)
        plt.xlabel("Highest pLDDT across trajectories/iterations")
        plt.ylabel("TM-score vs ground-truth")
        plt.title("Per-chain: best pLDDT vs TM-score (red = has pLDDT>70 & TM<0.5)")
        plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        plt.axvline(x=70, color="gray", linestyle="--", alpha=0.5)
        plot_path = args.plot_path or os.path.join(args.output_dir, "cross_chain_plddt_vs_tm.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
