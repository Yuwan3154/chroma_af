#!/usr/bin/env python3
"""
Chroma + OpenFold pipeline: alternating structure refinement.

Usage:
    conda run -n openfold_chroma python run_chroma_openfold.py \\
        --pdb_list /path/to/test_decoys.txt \\
        --output_dir /home/ubuntu/chroma_af/results \\
        --num_experiments 2 \\
        --num_iterations 3
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure chroma_af is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "chroma"))

# Chroma API key
from chroma import api
api.register_key("10e48bde5ef449e3bcee003bf12d5b59")
from chroma import Chroma, Protein, conditioners

from pipeline.openfold_wrapper import load_pdb_for_openfold, run_openfold_on_pdb
from pipeline.metrics import compute_iteration_metrics, compute_af_chroma_dist_diff
from pipeline.logging_utils import (
    append_metrics,
    update_best,
    save_plots,
    save_metrics_json,
    log_best,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_pdb(pdb_id: str, work_dir: str) -> str:
    """Download PDB from RCSB if not present. Returns path to PDB file."""
    pdb_path = os.path.join(work_dir, f"{pdb_id}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path
    url = f"https://files.rcsb.org/view/{pdb_id}.pdb"
    subprocess.run(["wget", "-qnc", url, "-O", pdb_path], check=True, cwd=work_dir)
    return pdb_path


def parse_pdb_list(path: str) -> list[tuple[str, str]]:
    """
    Parse PDB list file. Each line: pdb_id or pdb_id chain.
    Returns list of (pdb_id, chain) tuples.
    """
    result = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            pdb_id = parts[0]
            chain = parts[1] if len(parts) > 1 else "A"
            result.append((pdb_id, chain))
    return result


def load_af2rank_lookup(
    csv_path: str,
    pdb_root: str,
) -> dict[str, tuple[str, str]]:
    """
    Load af2rank CSV and build lookup: pdb_id (lowercase) -> (chain, gt_path).

    Chain from natives_rcsb (format pdbid_chain). Lookup: first natives_frank,
    then natives_rcsb. Ground-truth path: pdb_root/{mid2}/{pdbid}.cif
    where mid2 = middle two chars of 4-char pdb code (e.g. 1r6j -> r6).
    """
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


def _get_resume_iter(save_dir: str, exp_name: str, num_iterations: int) -> tuple[int, str]:
    """
    Find last completed iteration for resume. Returns (start_iter, tem_fname).
    A full iteration is complete when ch_i.pdb exists.
    """
    last_completed = -1
    for i in range(num_iterations):
        ch_path = os.path.join(save_dir, f"{exp_name}_ch_{i:03d}.pdb")
        if os.path.exists(ch_path):
            last_completed = i
    if last_completed >= num_iterations - 1:
        return num_iterations, ""  # Experiment complete
    start_iter = last_completed + 1
    if last_completed >= 0:
        tem_fname = os.path.join(save_dir, f"{exp_name}_ch_{last_completed:03d}.pdb")
    else:
        tem_fname = os.path.join(save_dir, f"{exp_name}_init.pdb")
    return start_iter, tem_fname


def main():
    parser = argparse.ArgumentParser(description="Chroma + OpenFold alternating refinement pipeline")
    parser.add_argument(
        "--pdb_list",
        type=str,
        default="/home/ubuntu/ProteinEBM/protein_ebm/data/data_lists/test_decoys.txt",
        help="Path to file with PDB IDs (one per line, optional chain)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/chroma_af/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/home/ubuntu/chroma_af",
        help="Working directory for PDB downloads",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=2,
        help="Number of experiments per PDB",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=3,
        help="Iterations per experiment",
    )
    parser.add_argument(
        "--exp_start",
        type=int,
        default=0,
        help="Starting experiment index",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=2,
        help="Save plots every N iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Default: cuda if available",
    )
    parser.add_argument(
        "--limit_pdb",
        type=int,
        default=None,
        help="Limit to first N PDBs (for testing)",
    )
    parser.add_argument(
        "--jax_params",
        type=str,
        default="/home/ubuntu/params/params_model_1_ptm.npz",
        help="Path to OpenFold JAX params",
    )
    parser.add_argument(
        "--template_mode",
        type=str,
        default="full_template",
        choices=["full_template", "distogram_only"],
        help="Template mode: full_template (default) uses structure file; distogram_only uses CA distogram",
    )
    parser.add_argument(
        "--kalign_binary_path",
        type=str,
        default="/usr/bin/kalign",
        help="Path to kalign binary (required when --template_mode=full_template)",
    )
    parser.add_argument(
        "--compile_chroma",
        action="store_true",
        help="Use torch.compile on Chroma backbone (PyTorch 2.0+). May speed up sampling.",
    )
    parser.add_argument(
        "--compile_openfold",
        action="store_true",
        help="Use torch.compile on OpenFold model. May speed up inference.",
    )
    parser.add_argument(
        "--template_sequence_all_x",
        action="store_true",
        help="Mask template sequence to all X (restype 20). Use when template has placeholder sequence (e.g. all A from Chroma).",
    )
    parser.add_argument(
        "--af2rank_csv",
        type=str,
        default=None,
        help="Path to af2rank CSV (natives_frank, natives_rcsb). When set with --af2rank_pdb_root, use for chain and ground-truth lookup.",
    )
    parser.add_argument(
        "--af2rank_pdb_root",
        type=str,
        default=None,
        help="Path to af2rank pdb dir. Ground truth: {root}/{mid2}/{pdbid}.cif (e.g. r6/1r6j.cif).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint. Skips completed experiments/iterations.",
    )
    args = parser.parse_args()

    if args.template_mode == "full_template" and not args.kalign_binary_path:
        parser.error("--kalign_binary_path is required when --template_mode=full_template")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    pdb_list = parse_pdb_list(args.pdb_list)
    if args.limit_pdb:
        pdb_list = pdb_list[: args.limit_pdb]
        logger.info("Limited to first %d PDBs", args.limit_pdb)

    af2rank_lookup = None
    if args.af2rank_csv and args.af2rank_pdb_root:
        af2rank_lookup = load_af2rank_lookup(args.af2rank_csv, args.af2rank_pdb_root)
        logger.info("Loaded af2rank lookup: %d entries", len(af2rank_lookup))

    for pdb_id, chain in pdb_list:
        pdb_id_lower = pdb_id.lower()
        if af2rank_lookup is not None and pdb_id_lower in af2rank_lookup:
            chain, gt_fname = af2rank_lookup[pdb_id_lower]
            if not os.path.exists(gt_fname):
                logger.warning("Ground truth not found: %s, skipping", gt_fname)
                continue
        else:
            gt_fname = fetch_pdb(pdb_id, args.work_dir)

        logger.info("Processing %s chain %s", pdb_id, chain)

        # Get sequence length from ground truth (chain-filtered for multi-chain CIF/PDB)
        _, residue_type_gt, _, _, _, _ = load_pdb_for_openfold(
            gt_fname, chain_id=chain, device=torch.device("cpu")
        )
        num_resi = residue_type_gt.shape[1]
        gt_dist = torch.zeros(num_resi, num_resi)  # will be computed in loop

        best = {"mean_plddt": None, "dist_diff": None}

        for exp_num in range(args.exp_start, args.exp_start + args.num_experiments):
            exp_name = f"{pdb_id}_exp_{exp_num:03d}"
            save_dir = os.path.join(args.output_dir, exp_name)
            os.makedirs(save_dir, exist_ok=True)

            init_fname = os.path.join(save_dir, f"{exp_name}_init.pdb")

            # Resume: skip completed experiments, find start iteration
            if args.resume:
                start_iter, tem_fname = _get_resume_iter(
                    save_dir, exp_name, args.num_iterations
                )
                if start_iter >= args.num_iterations:
                    logger.info("Skipping %s (already complete)", exp_name)
                    # Still update best from completed experiment's metrics
                    metrics_path = os.path.join(save_dir, f"{exp_name}_metrics.json")
                    if os.path.exists(metrics_path):
                        with open(metrics_path) as f:
                            for m in json.load(f):
                                iter_num = m.get("iter", 0)
                                ch_path = os.path.join(
                                    save_dir, f"{exp_name}_ch_{iter_num:03d}.pdb"
                                )
                                update_best(best, m, exp_num, iter_num, ch_path)
                    continue
                if start_iter > 0:
                    logger.info("Resuming %s from iteration %d", exp_name, start_iter)
            else:
                start_iter = 0
                tem_fname = init_fname

            # Schedules
            plddt_cutoff_schedule = np.linspace(0.6, 0.7, args.num_iterations)
            temperature_schedule = np.linspace(8, 8, args.num_iterations)
            noise_schedule = np.linspace(8, 0, args.num_iterations)
            tspan = (0.1, 0.9)

            # Chroma model (one per experiment)
            chroma = Chroma(device=device)
            if args.compile_chroma and hasattr(torch, "compile"):
                chroma.backbone_network = torch.compile(
                    chroma.backbone_network, mode="reduce-overhead"
                )
                logger.info("Chroma backbone compiled with torch.compile")

            # Initial structure (only if not resuming from iter > 0)
            # Regenerate init if it has wrong length (stale from multi-chain bug)
            if start_iter == 0 and os.path.exists(init_fname):
                _, rt_init, _, _, _, _ = load_pdb_for_openfold(
                    init_fname, chain_id=chain, device=torch.device("cpu")
                )
                if rt_init.shape[1] != num_resi:
                    logger.warning("Regenerating init.pdb: had %d residues, need %d", rt_init.shape[1], num_resi)
                    os.remove(init_fname)
            if start_iter == 0 and not os.path.exists(init_fname):
                protein_init = chroma.sample(chain_lengths=[num_resi], design_method=None)
                protein_init.to(init_fname)

            # Load existing metrics when resuming
            metrics_path = os.path.join(save_dir, f"{exp_name}_metrics.json")
            if args.resume and start_iter > 0 and os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    all_metrics = json.load(f)
                for m in all_metrics:
                    iter_num = m.get("iter", 0)
                    ch_path = os.path.join(save_dir, f"{exp_name}_ch_{iter_num:03d}.pdb")
                    update_best(best, m, exp_num, iter_num, ch_path)
            else:
                all_metrics = []

            for i in range(start_iter, args.num_iterations):
                # --- OpenFold ---
                af_out_path = os.path.join(save_dir, f"{exp_name}_af_{i:03d}.cif")
                of_result = run_openfold_on_pdb(
                    tem_fname,
                    af_out_path,
                    chain_id=chain,
                    device=device,
                    jax_params_path=args.jax_params,
                    template_mode=args.template_mode,
                    kalign_binary_path=args.kalign_binary_path,
                    compile_model=args.compile_openfold,
                    template_sequence_all_x=args.template_sequence_all_x,
                    query_sequence_path=gt_fname,
                    skip_template_alignment=True,
                )
                tem_fname = af_out_path

                # --- Metrics ---
                iter_metrics = compute_iteration_metrics(
                    af_pdb_path=tem_fname,
                    gt_pdb_path=gt_fname,
                    num_resi=num_resi,
                    openfold_output=of_result,
                    device=device,
                )
                iter_metrics["iter"] = i
                iter_metrics["exp"] = exp_num
                iter_metrics["pdb_id"] = pdb_id

                # --- Chroma ---
                plddt = of_result.get("plddt")
                if plddt is not None:
                    plddt_t = plddt.detach().cpu()
                else:
                    plddt_t = torch.ones(num_resi) * 70.0

                plddt_cutoff = plddt_cutoff_schedule[i]
                selection_str = "all"
                protein_af = (
                    Protein.from_CIF(tem_fname)
                    if tem_fname.lower().endswith(".cif")
                    else Protein.from_PDB(tem_fname)
                )
                if plddt_t.max() > plddt_cutoff:
                    plddt_cutoff = min(
                        plddt_cutoff,
                        float(torch.quantile(plddt_t, 0.25, interpolation="linear")),
                    )
                    mask = (plddt_t > plddt_cutoff).bool()
                    high_conf_indices = mask.nonzero(as_tuple=True)[0].tolist()
                    if high_conf_indices:
                        protein_af.sys.save_selection(
                            gti=high_conf_indices, selname="plddt_mask"
                        )
                        selection_str = "namesel plddt_mask"
                substructure_conditioner = conditioners.SubstructureConditioner(
                    protein_af,
                    backbone_model=chroma.backbone_network,
                    selection=selection_str,
                    tspan=tspan,
                ).to(device)

                temperature = float(temperature_schedule[i])
                noise = float(noise_schedule[i])
                protein_ch = chroma.sample(
                    chain_lengths=[num_resi],
                    initialize_noise=True,
                    conditioner=substructure_conditioner,
                    langevin_factor=noise,
                    langevin_isothermal=True,
                    inverse_temperature=temperature,
                    sde_func="langevin",
                    steps=400,
                    design_method=None,
                )
                ch_out_path = os.path.join(save_dir, f"{exp_name}_ch_{i:03d}.pdb")
                protein_ch.to(ch_out_path)
                tem_fname = ch_out_path

                # AF-Chroma dist diff
                iter_metrics["af_chroma_dist_diff"] = compute_af_chroma_dist_diff(
                    af_out_path, ch_out_path
                )

                append_metrics(all_metrics, iter_metrics)
                update_best(best, iter_metrics, exp_num, i, tem_fname)

                logger.info(
                    "Iter %d: dist_diff=%.3g plddt=%.3g pae=%.3g cmap_ent=%.3g af_chroma_diff=%.3g",
                    i,
                    iter_metrics["dist_diff"],
                    iter_metrics["plddt_mean"],
                    iter_metrics["pae_mean"],
                    iter_metrics["cmap_ent_mean"],
                    iter_metrics["af_chroma_dist_diff"],
                )

                if (i + 1) % args.print_interval == 0:
                    save_plots(save_dir, all_metrics, exp_name)

                torch.cuda.empty_cache()

            save_metrics_json(all_metrics, os.path.join(save_dir, f"{exp_name}_metrics.json"))
            log_best(best)

        logger.info("Done %s chain %s", pdb_id, chain)


if __name__ == "__main__":
    main()
