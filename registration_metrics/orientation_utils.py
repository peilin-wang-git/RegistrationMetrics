"""NIfTI loading and physical orientation utilities."""
from __future__ import annotations

import logging
from pathlib import Path
import nibabel as nib
import numpy as np

LOGGER = logging.getLogger("registration_metrics")


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """Load a NIfTI image and log path, shape, dtype, affine, spacing, and orientation."""
    p = Path(path)
    LOGGER.info("[LOAD] path=%s exists=%s", p, p.exists())
    img = nib.load(str(p))
    LOGGER.info("[LOAD] shape=%s dtype=%s", img.shape, img.get_data_dtype())
    LOGGER.info("[ORIENTATION] affine=\n%s", img.affine)
    LOGGER.info("[ORIENTATION] spacing=%s axis_codes=%s", get_spacing_from_affine(img), nib.aff2axcodes(img.affine))
    return img


def get_spacing_from_affine(img: nib.Nifti1Image) -> tuple[float, float, float]:
    """Return physical spacing from affine column norms for the first three array axes."""
    aff = img.affine[:3, :3]
    return tuple(float(np.linalg.norm(aff[:, i])) for i in range(3))


def get_axis_code_mapping(img: nib.Nifti1Image) -> dict[int, str]:
    """Map array axes to anatomical code families R/L, A/P, and S/I using NIfTI affine."""
    codes = nib.aff2axcodes(img.affine)
    if len(codes) < 3 or any(c is None for c in codes[:3]):
        raise ValueError("[ORIENTATION] affine cannot reliably determine axis codes")
    fam = {"R": "RL", "L": "RL", "A": "AP", "P": "AP", "S": "SI", "I": "SI"}
    mapping = {i: fam[codes[i]] for i in range(3)}
    if set(mapping.values()) != {"AP", "RL", "SI"}:
        raise ValueError(f"[ORIENTATION] ambiguous axis mapping: {mapping}")
    LOGGER.info("[ORIENTATION] axis mapping=%s axis_codes=%s", mapping, codes)
    return mapping


def convert_voxel_shift_to_physical_ap_rl_si(shift_voxel: np.ndarray | list[float], affine: np.ndarray) -> dict[str, float]:
    """Convert voxel displacement to AP/RL/SI millimeters using the affine linear part."""
    vec = np.asarray(shift_voxel, dtype=float)[:3]
    phys = affine[:3, :3] @ vec
    return {"RL": float(phys[0]), "AP": float(phys[1]), "SI": float(phys[2])}


def standardize_direction_names(name: str) -> str:
    """Normalize direction names to AP, RL, or SI."""
    n = str(name).upper().replace("-", "").replace("_", "")
    aliases = {"ANTERIORPOSTERIOR": "AP", "POSTERIORANTERIOR": "AP", "AP": "AP", "PA": "AP", "RIGHTLEFT": "RL", "LEFTRIGHT": "RL", "RL": "RL", "LR": "RL", "SUPERIORINFERIOR": "SI", "INFERIORSUPERIOR": "SI", "SI": "SI", "IS": "SI"}
    if n not in aliases:
        raise ValueError(f"Unknown direction name: {name}")
    return aliases[n]
