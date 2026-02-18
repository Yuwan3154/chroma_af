"""
Structure conversion utilities. Uses gemmi (no Chroma dependency).
"""

import os

import gemmi


def cif_to_pdb(cif_path: str, pdb_path: str, chain_id: str = "A") -> None:
    """
    Convert CIF to PDB, optionally filtering to a single chain.

    Uses gemmi for conversion. Does not require Chroma.
    """
    s = gemmi.read_structure(cif_path)
    if chain_id is not None:
        for model in s:
            for c in list(model):
                if c.name != chain_id:
                    model.remove_chain(c.name)
    os.makedirs(os.path.dirname(pdb_path) or ".", exist_ok=True)
    s.write_pdb(pdb_path)
