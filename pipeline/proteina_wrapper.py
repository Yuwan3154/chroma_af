"""
Proteina motif scaffolding wrapper for the Chroma-AF pipeline.

Provides an alternative to Chroma for the motif-scaffolding step using Proteina's
RFDiffusion-style contig-based motif conditioning.

Indexing convention: Both OpenFold output and Proteina contig use 1-based sequential
residue numbering (1, 2, 3, ..., L). The contig string "10/A11-26/20" means scaffold
of 10, motif residues 11-26 from chain A, scaffold of 20. high_conf_indices are
0-based (0 = first residue), so we convert: start_1 = idx + 1.
"""

import os
import sys
import tempfile
from typing import Optional

import numpy as np
import torch


def _ensure_sequential_residue_numbering(structure_path: str, chain_id: str = "A") -> str:
    """
    Ensure structure has sequential 1-based residue numbering for Proteina contig.

    Proteina's motif_extract uses res_id from the structure. Our contig assumes
    1-based sequential (1, 2, 3, ..., L). If the structure has different numbering,
    rewrite to a temp file with sequential numbering.

    Returns path to structure file (original or temp with normalized numbering).
    """
    _PROTEINA_PATH = "/home/ubuntu/proteina"
    if _PROTEINA_PATH not in sys.path:
        sys.path.insert(0, _PROTEINA_PATH)
    import biotite.structure.io as strucio
    from biotite.structure import create_continuous_res_ids

    array = strucio.load_structure(structure_path, model=1)
    chain_mask = array.chain_id == chain_id
    if not np.any(chain_mask):
        return structure_path
    chain_atoms = array[chain_mask].copy()
    res_ids = np.unique(chain_atoms.res_id)
    expected = np.arange(1, len(res_ids) + 1, dtype=res_ids.dtype)
    if np.array_equal(np.sort(res_ids), expected):
        return structure_path  # Already sequential 1-based

    chain_atoms.res_id = create_continuous_res_ids(chain_atoms, restart_each_chain=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    strucio.save_structure(tmp_path, chain_atoms)
    return tmp_path


def build_contig_from_plddt_mask(
    high_conf_indices: list[int],
    num_resi: int,
    chain_id: str = "A",
) -> str:
    """
    Build RFDiffusion-style contig string from high-confidence residue indices.

    Args:
        high_conf_indices: Sorted list of 0-based residue indices (high pLDDT).
        num_resi: Total number of residues in the chain.
        chain_id: Chain ID for motif segments (e.g. "A").

    Returns:
        Contig string, e.g. "10/A11-26/20" for scaffold before, motif A11-26, scaffold after.

    Edge case (no high-conf): returns minimal motif "1/A1/{num_resi-1}".
    """
    if not high_conf_indices:
        # Fallback: use first residue as minimal motif
        return f"1/{chain_id}1/{num_resi - 1}"

    sorted_indices = sorted(high_conf_indices)
    # Group into contiguous runs
    runs = []
    run_start = sorted_indices[0]
    run_end = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == run_end + 1:
            run_end = idx
        else:
            runs.append((run_start, run_end))
            run_start = run_end = idx
    runs.append((run_start, run_end))

    parts = []
    # Scaffold before first motif
    before = runs[0][0]
    parts.append(str(before))

    for i, (s, e) in enumerate(runs):
        start_1 = s + 1
        end_1 = e + 1
        if start_1 == end_1:
            parts.append(f"{chain_id}{start_1}")
        else:
            parts.append(f"{chain_id}{start_1}-{end_1}")
        # Scaffold between this run and next
        if i < len(runs) - 1:
            gap = runs[i + 1][0] - (e + 1)
            parts.append(str(gap))

    # Scaffold after last motif
    after = num_resi - (runs[-1][1] + 1)
    if after > 0:
        parts.append(str(after))

    return "/".join(parts)


def _default_inference_config() -> dict:
    """Default inference config for Proteina motif scaffolding."""
    return {
        "dt": 0.0025,
        "self_cond": False,
        "fold_cond": False,
        "guidance_weight": 1.0,
        "autoguidance_ratio": 1.0,
        "autoguidance_ckpt_path": None,
        "sampling_caflow": {
            "sampling_mode": "sc",
            "sc_scale_noise": 0.4,
            "sc_scale_score": 1.0,
            "gt_mode": "1/t",
            "gt_p": 1.0,
            "gt_clamp_val": None,
        },
        "schedule": {
            "schedule_mode": "log",
            "schedule_p": 2.0,
        },
    }


def create_proteina_inference(
    ckpt_path: str,
    ckpt_name: str = "proteina_v1.7_DFS_60M_notri_motif_scaffolding.ckpt",
    device: Optional[torch.device] = None,
    config_path: Optional[str] = None,
):
    # Lazy imports to avoid loading Proteina deps when using Chroma backend
    _PROTEINA_PATH = "/home/ubuntu/proteina"
    if _PROTEINA_PATH not in sys.path:
        sys.path.insert(0, _PROTEINA_PATH)
    from omegaconf import OmegaConf
    from proteinfoundation.proteinflow.proteina import Proteina
    """
    Load Proteina motif-scaffolding model from checkpoint for reuse across runs.

    Call once per experiment (or per PDB) and pass the returned model to
    run_proteina_motif_scaffolding.

    Args:
        ckpt_path: Directory containing the checkpoint.
        ckpt_name: Checkpoint filename.
        device: Device to load model on. Default: cuda if available.
        config_path: Optional path to YAML config. If None, uses default inference config.

    Returns:
        Proteina model with inference configured.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_file = os.path.join(ckpt_path, ckpt_name)
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(
            f"Proteina checkpoint not found: {ckpt_file}. "
            "Set --proteina_ckpt_path to the directory containing the motif-scaffolding checkpoint."
        )

    model = Proteina.load_from_checkpoint(ckpt_file, map_location=device)
    model = model.to(device)
    model.eval()

    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create(_default_inference_config())

    model.configure_inference(cfg, nn_ag=None)
    return model


def run_proteina_motif_scaffolding(
    proteina_model: "Proteina",
    motif_pdb_path: str,
    contig_string: str,
    output_path: str,
    num_resi: int,
    chain_id: str = "A",
    motif_only: bool = False,
    nsamples: int = 1,
) -> str:
    """
    Run Proteina motif scaffolding with a pre-loaded model.

    Args:
        proteina_model: Pre-loaded Proteina model from create_proteina_inference.
        motif_pdb_path: Path to motif structure (OpenFold output, PDB or CIF).
        contig_string: RFDiffusion-style contig (from build_contig_from_plddt_mask).
        output_path: Path to write output PDB.
        num_resi: Total number of residues (used for min/max length bounds).
        motif_only: If True, whole file is motif. Default False (motif is subset).
        nsamples: Number of samples (default 1).

    Returns:
        Path to written output PDB.
    """
    # Lazy imports to avoid loading Proteina deps when using Chroma backend
    _PROTEINA_PATH = "/home/ubuntu/proteina"
    if _PROTEINA_PATH not in sys.path:
        sys.path.insert(0, _PROTEINA_PATH)
    from proteinfoundation.nn.motif_factory import parse_motif
    from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

    # Ensure motif structure has sequential 1-based residue numbering for contig
    motif_path_for_parse = _ensure_sequential_residue_numbering(motif_pdb_path, chain_id=chain_id)
    needs_cleanup = motif_path_for_parse != motif_pdb_path
    try:
        mask, x_motif_list, out_str_list = parse_motif(
            motif_path_for_parse,
            contig_string,
            nsamples=nsamples,
            make_tensor=False,
            motif_only=motif_only,
            min_length=num_resi,
            max_length=num_resi,
        )

        device = next(proteina_model.parameters()).device
        # Use first sample
        motif_mask = mask[0].to(device)
        x_motif_full = x_motif_list[0].to(device)
        # Proteina expects motif coords in nm (Angstrom / 10)
        x_motif_nm = x_motif_full / 10.0
        nres = motif_mask.shape[0]
        fixed_structure_mask = motif_mask[:, None] * motif_mask[None, :]

        # General mask (all positions valid)
        general_mask = torch.ones(nres, dtype=torch.bool, device=device)

        # Build batch for generate
        cath_code = [["x.x.x.x"] for _ in range(nsamples)]
        residue_type = None

        result = proteina_model.generate(
            nsamples=nsamples,
            n=nres,
            dt=float(proteina_model.inf_cfg.dt),
            self_cond=proteina_model.inf_cfg.self_cond,
            cath_code=cath_code,
            residue_type=residue_type,
            guidance_weight=proteina_model.inf_cfg.get("guidance_weight", 1.0),
            autoguidance_ratio=proteina_model.inf_cfg.get("autoguidance_ratio", 0.0),
            dtype=torch.float32,
            schedule_mode=proteina_model.inf_cfg.schedule.schedule_mode,
            schedule_p=proteina_model.inf_cfg.schedule.schedule_p,
            sampling_mode=proteina_model.inf_cfg.sampling_caflow.sampling_mode,
            sc_scale_noise=proteina_model.inf_cfg.sampling_caflow.sc_scale_noise,
            sc_scale_score=proteina_model.inf_cfg.sampling_caflow.sc_scale_score,
            gt_mode=proteina_model.inf_cfg.sampling_caflow.gt_mode,
            gt_p=proteina_model.inf_cfg.sampling_caflow.gt_p,
            gt_clamp_val=proteina_model.inf_cfg.sampling_caflow.gt_clamp_val,
            mask=general_mask.unsqueeze(0),
            x_motif=x_motif_nm.unsqueeze(0),
            fixed_sequence_mask=motif_mask.unsqueeze(0),
            fixed_structure_mask=fixed_structure_mask.unsqueeze(0),
        )

        coords = result["coords"]
        coords_atom37 = proteina_model.samples_to_atom37(coords)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        write_prot_to_pdb(
            coords_atom37[0].detach().cpu().numpy(),
            output_path,
            overwrite=True,
            no_indexing=True,
        )
        return output_path
    finally:
        if needs_cleanup and os.path.exists(motif_path_for_parse):
            try:
                os.remove(motif_path_for_parse)
            except OSError:
                pass
