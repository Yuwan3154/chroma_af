"""
Metrics computation for the Chroma + OpenFold pipeline.
"""

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


def compute_iteration_metrics(
    af_pdb_path: str,
    gt_pdb_path: str,
    num_resi: int,
    openfold_output: dict,
    device: torch.device,
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
        Dict with dist_diff, plddt_mean, pae_mean, cmap_ent_mean, mpnn_ce_mean, mpnn_ent_mean.
        mpnn_ce_mean and mpnn_ent_mean are None (not implemented).
    """
    from chroma import Protein

    def _load_protein(path: str):
        return Protein.from_CIF(path) if path.lower().endswith(".cif") else Protein.from_PDB(path)

    # Load structures
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
        "mpnn_ce_mean": mpnn_ce_mean,
        "mpnn_ent_mean": mpnn_ent_mean,
    }


def compute_af_chroma_dist_diff(af_pdb_path: str, chroma_pdb_path: str) -> float:
    """Distance difference between AF and Chroma structures (debug metric)."""
    from chroma import Protein

    def _load_protein(path: str):
        return Protein.from_CIF(path) if path.lower().endswith(".cif") else Protein.from_PDB(path)

    af_protein = _load_protein(af_pdb_path)
    ch_protein = _load_protein(chroma_pdb_path)

    af_dist = pair_wise_distance_ca(af_protein.to_XCS()[0])
    ch_dist = pair_wise_distance_ca(ch_protein.to_XCS()[0])

    return float((ch_dist - af_dist).abs().mean().item())
