"""
OpenFold inference wrapper for the Chroma pipeline.

Runs OpenFold (monomer, model_1_ptm) with either:
- distogram_only: template from CA-distance distogram (no external file)
- full_template: template from structure file (mmCIF or PDB)

For full_template with PDB input:
- If basename is a 4-char PDB ID (e.g. 1r6j), fetches mmCIF from RCSB
- Otherwise uses OpenFold's PDB->ModelCIF conversion (no gemmi needed)
"""

import logging
import os
import sys
import tempfile
import urllib.request
from typing import Optional, Tuple

import gemmi
import numpy as np
import torch

from openfold.data import mmcif_parsing
from openfold.np import protein
from openfold.np import residue_constants as rc

# Ensure proteina is on path for OpenFoldTemplateInference
_PROTEINA_PATH = "/home/ubuntu/proteina"
if _PROTEINA_PATH not in sys.path:
    sys.path.insert(0, _PROTEINA_PATH)

from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference


def create_openfold_inference(
    model_name: str = "model_1_ptm",
    jax_params_path: str = "/home/ubuntu/params/params_model_1_ptm.npz",
    device: Optional[torch.device] = None,
    max_recycling_iters: int = 3,
    rm_template_sequence: bool = False,
    skip_template_alignment: bool = True,
    compile_model: bool = False,
) -> OpenFoldTemplateInference:
    """
    Create and return an OpenFoldTemplateInference instance for reuse across runs.

    Call once per experiment (or per PDB) and pass the returned instance to
    run_openfold_on_pdb via the infer argument to avoid reloading weights each iteration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValueError("OpenFold requires CUDA")

    infer = OpenFoldTemplateInference(
        model_name=model_name,
        jax_params_path=jax_params_path,
        device=device,
        max_recycling_iters=max_recycling_iters,
        rm_template_sequence=rm_template_sequence,
        skip_template_alignment=skip_template_alignment,
    )
    if compile_model and hasattr(torch, "compile"):
        infer.model = torch.compile(infer.model, mode="reduce-overhead")
        logging.getLogger(__name__).info("OpenFold model compiled with torch.compile")
    return infer


def _fetch_mmcif_from_rcsb(pdb_id: str) -> str:
    """Fetch mmCIF from RCSB for a known PDB ID. Returns path to temp file."""
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.cif"
    fd, cif_path = tempfile.mkstemp(suffix=".cif")
    os.close(fd)
    urllib.request.urlretrieve(url, cif_path)
    return cif_path


def _pdb_to_mmcif(pdb_path: str, chain_id: str = "A") -> Optional[str]:
    """
    Get mmCIF path for full_template mode.

    - If basename (without ext) looks like a 4-char PDB ID, fetch mmCIF from RCSB.
    - Otherwise try gemmi conversion (requires gemmi package).
    - Returns path to mmCIF file, or None if conversion fails.
    """
    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    # Check if it's a standard PDB ID (4 alphanumeric)
    if len(stem) == 4 and stem.isalnum():
        try:
            return _fetch_mmcif_from_rcsb(stem)
        except Exception as e:
            logging.warning("Could not fetch mmCIF from RCSB for %s: %s", stem, e)

    # Use OpenFold's own PDB->ModelCIF conversion (gemmi output is unparseable by OpenFold)
    try:
        with open(pdb_path) as f:
            pdb_str = f.read()
        prot = protein.from_pdb_string(pdb_str, chain_id=chain_id)
        cif_str = protein.to_modelcif(prot)
        fd, cif_path = tempfile.mkstemp(suffix=".cif")
        os.close(fd)
        with open(cif_path, "w") as f:
            f.write(cif_str)
        return cif_path
    except Exception as e:
        logging.warning("Could not convert PDB to ModelCIF: %s", e)
        return None


def _distogram_probs_from_pseudo_beta(
    pseudo_beta: torch.Tensor,
    *,
    num_bins: int = 39,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
) -> torch.Tensor:
    """Build distogram probabilities [1, L, L, num_bins] from CA coordinates in Å."""
    d = torch.cdist(pseudo_beta[None, ...], pseudo_beta[None, ...], p=2.0)[0]
    boundaries = torch.linspace(
        min_bin, max_bin, num_bins - 1, device=pseudo_beta.device, dtype=pseudo_beta.dtype
    )
    b = torch.bucketize(d, boundaries)
    logits = torch.zeros(
        (d.shape[0], d.shape[1], num_bins), device=pseudo_beta.device, dtype=pseudo_beta.dtype
    )
    logits.scatter_(-1, b[..., None], 1.0)
    return logits[None, ...]


def _protein_from_atom37(
    *,
    atom37: np.ndarray,
    aatype: np.ndarray,
    residue_index: np.ndarray,
    chain_index: np.ndarray,
    remark: str,
) -> protein.Protein:
    """Convert atom37 coordinates to OpenFold Protein for PDB writing."""
    if atom37.ndim != 3 or atom37.shape[1] != rc.atom_type_num or atom37.shape[2] != 3:
        raise ValueError(f"Expected atom37 shape [L, 37, 3], got {atom37.shape}")
    aatype_clamped = np.clip(aatype.astype(np.int32), 0, rc.restype_num)
    atom_mask = rc.restype_atom37_mask[aatype_clamped].astype(np.float32)
    b_factors = np.zeros_like(atom_mask, dtype=np.float32)
    return protein.Protein(
        atom_positions=atom37.astype(np.float32),
        aatype=aatype_clamped.astype(np.int32),
        atom_mask=atom_mask,
        residue_index=residue_index.astype(np.int32),
        b_factors=b_factors,
        chain_index=chain_index.astype(np.int32),
        remark=remark,
    )


def _get_label_to_auth_chain_mapping(cif_path: str) -> dict[str, str]:
    """
    Build label_asym_id -> auth_asym_id mapping from mmCIF.

    Standard CIF chain assignment uses label_asym_id (A, B, C...).
    Gemmi and PDB use auth_asym_id for chain names.
    """
    try:
        doc = gemmi.cif.read_file(cif_path)
        block = doc.sole_block()
        cat = block.get_mmcif_category("_atom_site")
        labels = cat.get("label_asym_id", [])
        auths = cat.get("auth_asym_id", [])
        if not labels or not auths:
            # Fallback: struct_asym (pdbx_blank_PDB_chainid_flag)
            cat = block.get_mmcif_category("_struct_asym")
            ids = cat.get("id", [])
            auths = cat.get("pdbx_blank_PDB_chainid_flag", [])
            labels = ids
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(auths, str):
            auths = [auths]
        return dict(zip(labels, auths))
    except Exception:
        return {}


def load_pdb_for_openfold(
    pdb_path: str,
    chain_id: str = "A",
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PDB or CIF and prepare inputs for OpenFoldTemplateInference.

    Supports .pdb and .cif (including OpenFold ModelCIF). For CIF, uses gemmi
    to convert to PDB string for parsing.

    Returns:
        distogram_probs: [1, L, L, 39]
        residue_type: [1, L]
        mask: [1, L]
        aatype: [L] (for PDB writing)
        residue_index: [L]
        chain_index: [L]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    effective_chain_id = chain_id
    if pdb_path.lower().endswith(".cif"):
        # Resolve standard (label_asym_id) to author (auth) chain for mmCIF
        # af2rank and similar use standard chain assignment (label_asym_id)
        label_to_auth = _get_label_to_auth_chain_mapping(pdb_path)
        if chain_id is not None and chain_id in label_to_auth:
            effective_chain_id = label_to_auth[chain_id]

        s = gemmi.read_structure(pdb_path)
        if chain_id is not None:
            for model in s:
                chain_names = [c.name for c in model]
                target = effective_chain_id
                if target in chain_names:
                    for c in list(model):
                        if c.name != target:
                            model.remove_chain(c.name)
                elif chain_names:
                    logging.getLogger(__name__).warning(
                        "Chain %s (auth %s) not found in %s (has %s), using %s",
                        chain_id, effective_chain_id, pdb_path, chain_names, chain_names[0],
                    )
                    effective_chain_id = chain_names[0]
                    for c in list(model):
                        if c.name != effective_chain_id:
                            model.remove_chain(c.name)
        fd, tmp_pdb = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)
        try:
            s.write_pdb(tmp_pdb)
            with open(tmp_pdb) as f:
                pdb_str = f.read()
        finally:
            try:
                os.remove(tmp_pdb)
            except OSError:
                pass
    else:
        with open(pdb_path, "r") as f:
            pdb_str = f.read()

    prot = protein.from_pdb_string(pdb_str, chain_id=effective_chain_id)
    keep = prot.aatype != rc.restype_num
    aatype = prot.aatype[keep]
    atom_positions = prot.atom_positions[keep]

    is_gly = aatype == rc.restype_order["G"]
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = np.where(
        np.tile(is_gly[..., None], (*((1,) * len(is_gly.shape)), 3)),
        atom_positions[..., ca_idx, :],
        atom_positions[..., cb_idx, :],
    )

    pseudo_beta_t = torch.tensor(pseudo_beta, dtype=torch.float32, device=device)
    distogram_probs = _distogram_probs_from_pseudo_beta(pseudo_beta_t)

    residue_type = torch.tensor(aatype, dtype=torch.long, device=device)[None, :]
    mask = torch.ones_like(residue_type, dtype=torch.float32)

    residue_index = prot.residue_index[keep]
    chain_index = prot.chain_index[keep]

    return (
        distogram_probs,
        residue_type,
        mask,
        aatype,
        residue_index,
        chain_index,
    )


def run_openfold_on_pdb(
    pdb_path: str,
    output_path: str,
    chain_id: str = "A",
    device: Optional[torch.device] = None,
    jax_params_path: str = "/home/ubuntu/params/params_model_1_ptm.npz",
    model_name: str = "model_1_ptm",
    template_mode: str = "full_template",
    kalign_binary_path: Optional[str] = None,
    compile_model: bool = False,
    rm_template_sequence: bool = False,
    query_sequence_path: Optional[str] = None,
    skip_template_alignment: bool = True,
    infer: Optional[OpenFoldTemplateInference] = None,
) -> dict:
    """
    Run OpenFold (monomer) on a PDB and save result.

    Args:
        template_mode: "full_template" (default) uses structure file as template;
            "distogram_only" uses CA-distance distogram only.
        kalign_binary_path: Required when template_mode is "full_template".
        rm_template_sequence: If True, mask template sequence to all X (restype 20).
        query_sequence_path: Path to ground-truth PDB/CIF for query sequence. If None, uses template.
        skip_template_alignment: If True, use 1:1 mapping (template same length as query).
        infer: Optional pre-created OpenFoldTemplateInference instance. If provided, reuse it
            instead of creating a new one (avoids reloading weights each call).

    Returns:
        dict with keys: final_atom_positions, plddt, predicted_aligned_error (if PTM),
        distogram_logits, and scalar metrics (plddt_mean, pae_mean).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValueError("OpenFold requires CUDA")
    if template_mode == "full_template" and not kalign_binary_path:
        raise ValueError("kalign_binary_path is required for template_mode=full_template")

    (
        distogram_probs,
        residue_type,
        mask,
        aatype,
        residue_index,
        chain_index,
    ) = load_pdb_for_openfold(pdb_path, chain_id=chain_id, device=device)

    # Query sequence: always from ground-truth when provided
    if query_sequence_path is not None:
        _, gt_residue_type, _, gt_aatype, _, _ = load_pdb_for_openfold(
            query_sequence_path, chain_id=chain_id, device=device
        )
        if gt_residue_type.shape[1] != residue_type.shape[1]:
            if residue_type.shape[1] > gt_residue_type.shape[1]:
                L = gt_residue_type.shape[1]
                residue_type = residue_type[:, :L]
                residue_index = residue_index[:L]
                chain_index = chain_index[:L]
                aatype = aatype[:L]
                distogram_probs = distogram_probs[:, :L, :L, :]
            else:
                raise ValueError(
                    f"Query sequence length {gt_residue_type.shape[1]} != template length {residue_type.shape[1]}"
                )
        residue_type = gt_residue_type
        aatype = gt_aatype

    if infer is None:
        infer = create_openfold_inference(
            model_name=model_name,
            jax_params_path=jax_params_path,
            device=device,
            max_recycling_iters=3,
            rm_template_sequence=rm_template_sequence,
            skip_template_alignment=skip_template_alignment,
            compile_model=compile_model,
        )

    template_mmcif_path = None
    temp_cif_path = None
    effective_template_mode = template_mode
    if template_mode == "full_template":
        if pdb_path.lower().endswith(".cif"):
            template_mmcif_path = pdb_path
        elif pdb_path.lower().endswith(".pdb"):
            temp_cif_path = _pdb_to_mmcif(pdb_path, chain_id=chain_id)
            if temp_cif_path:
                # Verify OpenFold can parse it
                with open(temp_cif_path) as f:
                    parse_result = mmcif_parsing.parse(
                        file_id="template", mmcif_string=f.read()
                    )
                if parse_result.mmcif_object is None:
                    logging.warning(
                        "PDB->mmCIF conversion produced unparseable file; "
                        "falling back to distogram_only"
                    )
                    if temp_cif_path and os.path.exists(temp_cif_path):
                        try:
                            os.remove(temp_cif_path)
                        except OSError:
                            pass
                    temp_cif_path = None
                    effective_template_mode = "distogram_only"
                else:
                    template_mmcif_path = temp_cif_path
            else:
                logging.warning(
                    "Could not convert PDB to mmCIF; falling back to distogram_only"
                )
                effective_template_mode = "distogram_only"
        else:
            template_mmcif_path = pdb_path

        if effective_template_mode == "full_template":
            # Dummy distogram (not used in full_template)
            l = residue_type.shape[1]
            distogram_probs = torch.full(
                (1, l, l, 39), 1.0 / 39.0, device=device, dtype=torch.float32
            )
        else:
            # Use existing distogram_probs from load_pdb_for_openfold above
            pass
    else:
        effective_template_mode = "distogram_only"

    try:
        out = infer(
            distogram_probs,
            residue_type,
            mask,
            template_mode=effective_template_mode,
            template_mmcif_path=template_mmcif_path if effective_template_mode == "full_template" else None,
            template_chain_id=chain_id,
            kalign_binary_path=kalign_binary_path if effective_template_mode == "full_template" else None,
        )
    finally:
        if temp_cif_path and os.path.exists(temp_cif_path):
            try:
                os.remove(temp_cif_path)
            except OSError:
                pass

    atom_pos = out["final_atom_positions"]
    if atom_pos.dim() == 3:
        atom37 = atom_pos.unsqueeze(0)
    else:
        atom37 = atom_pos

    # Write output (CIF preferred for full_template compatibility; PDB fallback)
    remark = f"Predicted by OpenFold (template_mode={effective_template_mode})"
    prot_pred = _protein_from_atom37(
        atom37=atom37[0].detach().cpu().numpy(),
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        remark=remark,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".cif"):
        cif_str = protein.to_modelcif(prot_pred)
        with open(output_path, "w") as f:
            f.write(cif_str)
    else:
        pdb_str = protein.to_pdb(prot_pred)
        with open(output_path, "w") as f:
            f.write(pdb_str)

    # Extract scalar metrics
    result = {"output_path": output_path}
    result["plddt"] = out["plddt"]
    if out["plddt"] is not None:
        plddt_np = out["plddt"].detach().cpu().numpy()
        result["plddt_mean"] = float(np.mean(plddt_np))

    if "predicted_aligned_error" in out:
        pae = out["predicted_aligned_error"]
        pae_np = pae.detach().cpu().numpy()
        result["predicted_aligned_error"] = pae
        result["pae_mean"] = float(np.mean(pae_np))

    if "distogram_logits" in out:
        result["distogram_logits"] = out["distogram_logits"]

    result["final_atom_positions"] = atom37

    return result
