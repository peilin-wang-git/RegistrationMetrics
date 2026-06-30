"""Dense displacement field Jacobian determinant metrics."""
from __future__ import annotations

import logging
from pathlib import Path
import nibabel as nib
import numpy as np
from .orientation_utils import get_spacing_from_affine, load_nifti

LOGGER = logging.getLogger("registration_metrics")


def infer_dvf_array(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Infer supported DVF vector axis and return array shaped (X,Y,Z,3)."""
    LOGGER.info("[DVF] raw shape=%s", data.shape)
    if data.ndim == 4 and data.shape[-1] == 3:
        return data.astype(float), "last"
    if data.ndim == 5 and data.shape[-2:] == (1, 3):
        return data[..., 0, :].astype(float), "last_after_singleton"
    if data.ndim == 4 and data.shape[0] == 3:
        return np.moveaxis(data, 0, -1).astype(float), "first"
    raise ValueError(f"Unsupported DVF shape {data.shape}; expected X,Y,Z,3 or X,Y,Z,1,3 or 3,X,Y,Z")


def jacobian_determinant(dvf: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Compute det(I + grad(u)) using physical spacing."""
    grads = [np.gradient(dvf[..., c], *spacing, edge_order=1) for c in range(3)]
    j = np.zeros(dvf.shape[:3] + (3, 3), dtype=float)
    for c in range(3):
        for ax in range(3):
            j[..., c, ax] = grads[c][ax]
    eye = np.eye(3)
    return np.linalg.det(j + eye)


def compute_dvf_metrics(transform_path: str | Path, case_id: str, save_jacobian_path: str | Path | None = None) -> dict[str, float]:
    """Load a dense DVF NIfTI and compute Jacobian/folding metrics."""
    LOGGER.info("[DVF] case=%s", case_id); LOGGER.info("[DVF] path=%s", transform_path)
    img = load_nifti(transform_path); raw = np.asanyarray(img.dataobj)
    dvf, axis = infer_dvf_array(raw); spacing = get_spacing_from_affine(img)
    LOGGER.info("[DVF] inferred vector axis=%s", axis); LOGGER.info("[DVF] spacing=%s", spacing)
    for c in range(3): LOGGER.info("[DVF] displacement component=%s min=%s max=%s mean=%s", c, float(np.nanmin(dvf[..., c])), float(np.nanmax(dvf[..., c])), float(np.nanmean(dvf[..., c])))
    det = jacobian_determinant(dvf, spacing); valid = np.isfinite(det); fold = valid & (det <= 0)
    out = {"jacobian_min": float(np.nanmin(det)), "jacobian_max": float(np.nanmax(det)), "jacobian_mean": float(np.nanmean(det)), "jacobian_std": float(np.nanstd(det)), "num_folding_voxels": int(fold.sum()), "folding_ratio": float(fold.sum() / max(int(valid.sum()), 1))}
    LOGGER.info("[DVF] jacobian min/max/mean/std=%s/%s/%s/%s", out["jacobian_min"], out["jacobian_max"], out["jacobian_mean"], out["jacobian_std"])
    LOGGER.info("[DVF] folding voxels=%s, folding_ratio=%s", out["num_folding_voxels"], out["folding_ratio"])
    if save_jacobian_path:
        nib.save(nib.Nifti1Image(det.astype(np.float32), img.affine), str(save_jacobian_path)); LOGGER.info("[SAVE] jacobian determinant map=%s", save_jacobian_path)
    return out
