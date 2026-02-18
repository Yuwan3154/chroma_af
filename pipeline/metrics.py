"""
Metrics computation for the Chroma + OpenFold pipeline.
"""

import re
import subprocess
import tempfile
from typing import Optional

import numpy as np
import torch


def pair_wise_distance_ca(X: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise CA distances.
    X has shape (batches, residues, atoms, coordinates); alpha-Carbon is at index 1.
    """
    assert len(X.shape) == 4
    ca = X[:, :, 1]
    dist = ((ca[:, :, None] - ca[:, None, :]).square().sum(-1) + 1e-8).sqrt()
    return dist


def entropy(C: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Entropy of probability distribution on the given dimension."""
    return -(C * C.log()).sum(dim=dim)


def compute_tm_score(
    pred_path: str,
    ref_path: str,
    usalign_binary: str = "USalign",
) -> Optional[float]:
    """
    Compute TM-score between predicted and reference structure using USalign.

    Args:
        pred_path: Path to predicted structure (PDB or CIF).
        ref_path: Path to reference/ground-truth structure (PDB or CIF).
        usalign_binary: Path to USalign executable.

    Returns:
        TM-score (0-1) normalized by reference length, or None if USalign fails.
    """
    try:
        result = subprocess.run(
            [usalign_binary, pred_path, ref_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return None
        # Parse: TM-score= 0.41854 (normalized by length of Structure_2: L=82, d0=3.24)
        # Use the one normalized by Structure_2 (reference)
        match = re.search(
            r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_2:",
            result.stdout,
        )
        if match:
            return float(match.group(1))
        # Fallback: first TM-score line
        match = re.search(r"TM-score=\s*([\d.]+)", result.stdout)
        return float(match.group(1)) if match else None
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


def compute_iteration_metrics(
    af_pdb_path: str,
    gt_pdb_path: str,
    num_resi: int,
    openfold_output: dict,
    device: torch.device,
    usalign_binary: str = "USalign",
) -> dict:
    """
    Compute metrics for one iteration: dist_diff, plddt, pae, cmap_ent.

    Args:
        af_pdb_path: Path to OpenFold-predicted PDB
        gt_pdb_path: Path to ground-truth PDB
        num_resi: Number of residues
        openfold_output: Dict from run_openfold_on_pdb (plddt, predicted_aligned_error, distogram_logits)
        device: Torch device

    Returns:
        Dict with dist_diff, plddt_mean, pae_mean, cmap_ent_mean, tm_score, mpnn_ce_mean, mpnn_ent_mean.
        mpnn_ce_mean and mpnn_ent_mean are None (not implemented).
    """
    # Lazy import Chroma (may not be installed in Proteina-only env)
    try:
        from chroma import Protein
    except ImportError:
        Protein = None

    def _load_protein(path: str):
        if Protein is None:
            raise ImportError("Chroma is required for compute_iteration_metrics (dist_diff)")
        return Protein.from_CIF(path) if path.lower().endswith(".cif") else Protein.from_PDB(path)

    # TM-score (USalign)
    tm_score = compute_tm_score(af_pdb_path, gt_pdb_path, usalign_binary=usalign_binary)

    # dist_diff (requires Chroma; None if not available)
    mean_dist_diff = None
    if Protein is not None:
        gt_protein = _load_protein(gt_pdb_path)
        af_protein = _load_protein(af_pdb_path)
        gt_X = gt_protein.to_XCS()[0][:, :num_resi]
        af_X = af_protein.to_XCS()[0]
        gt_dist = pair_wise_distance_ca(gt_X)
        af_dist = pair_wise_distance_ca(af_X)
        mean_dist_diff = (af_dist - gt_dist).abs().mean().item()

    # pLDDT from OpenFold output
    plddt = openfold_output.get("plddt")
    if plddt is not None:
        plddt_mean = float(plddt.detach().cpu().numpy().mean())
    else:
        plddt_mean = 0.0

    # PAE from OpenFold output (scale to 0-1 for consistency with ColabDesign: /32)
    pae = openfold_output.get("predicted_aligned_error")
    if pae is not None:
        pae_mean = float(pae.detach().cpu().numpy().mean()) / 32.0
    else:
        pae_mean = 0.0

    # Contact map entropy from distogram_logits
    distogram_logits = openfold_output.get("distogram_logits")
    if distogram_logits is not None:
        cmap = distogram_logits.to(device)
        cmap_probs = torch.softmax(cmap, dim=-1).clamp(min=1e-8)  # avoid log(0)
        norm = np.log(max(num_resi, 1)) * max(num_resi, 1)
        cmap_ent = entropy(cmap_probs) / (norm if norm > 0 else 1.0)
        cmap_ent_mean = float(cmap_ent.mean().item())
    else:
        cmap_ent_mean = 0.0

    # ProteinMPNN: NOT IMPLEMENTED - placeholder for future integration.
    # Masking currently uses pLDDT only; MPNN metrics can be added when available.
    mpnn_ce_mean = None
    mpnn_ent_mean = None

    return {
        "dist_diff": mean_dist_diff,
        "plddt_mean": plddt_mean,
        "pae_mean": pae_mean,
        "cmap_ent_mean": cmap_ent_mean,
        "tm_score": tm_score,
        "mpnn_ce_mean": mpnn_ce_mean,
        "mpnn_ent_mean": mpnn_ent_mean,
    }


def compute_af_chroma_dist_diff(af_pdb_path: str, chroma_pdb_path: str) -> Optional[float]:
    """Distance difference between AF and Chroma structures (debug metric). Returns None if Chroma not available."""
    try:
        from chroma import Protein
    except ImportError:
        return None

    def _load_protein(path: str):
        return Protein.from_CIF(path) if path.lower().endswith(".cif") else Protein.from_PDB(path)

    af_protein = _load_protein(af_pdb_path)
    ch_protein = _load_protein(chroma_pdb_path)

    af_dist = pair_wise_distance_ca(af_protein.to_XCS()[0])
    ch_dist = pair_wise_distance_ca(ch_protein.to_XCS()[0])

    return float((ch_dist - af_dist).abs().mean().item())
