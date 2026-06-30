"""Segmentation Dice, IoU, HD95, and ASSD metrics."""
from __future__ import annotations

import logging
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

LOGGER = logging.getLogger("registration_metrics")


def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Return Dice; both empty masks score 1."""
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    sa = int(a.sum()); sb = int(b.sum())
    if sa == 0 and sb == 0: return 1.0
    if sa == 0 or sb == 0: return 0.0
    return float(2 * np.logical_and(a, b).sum() / (sa + sb))


def iou_score(a: np.ndarray, b: np.ndarray) -> float:
    """Return IoU; both empty masks score 1."""
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool) ^ binary_erosion(mask.astype(bool))


def surface_distances(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Return symmetric surface distances in physical units."""
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    if not a.any() and not b.any(): return np.array([0.0])
    if not a.any() or not b.any(): return np.array([np.nan])
    sa = _surface(a); sb = _surface(b)
    da = distance_transform_edt(~sa, sampling=spacing); db = distance_transform_edt(~sb, sampling=spacing)
    return np.concatenate([db[sa], da[sb]]).astype(float)


def hd95(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Compute physical HD95; one-empty returns NaN and both-empty returns 0."""
    d = surface_distances(a, b, spacing)
    return float(np.nanpercentile(d, 95)) if np.isfinite(d).any() else float("nan")


def assd(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Compute physical ASSD; one-empty returns NaN and both-empty returns 0."""
    d = surface_distances(a, b, spacing)
    return float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")


def pair_metrics(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    """Compute Dice/IoU/HD95/ASSD for one mask pair."""
    return {"dice": dice_coefficient(a, b), "iou": iou_score(a, b), "hd95": hd95(a, b, spacing), "assd": assd(a, b, spacing)}


def compute_segmentation_metrics(fixed_seg: np.ndarray, moving_seg: np.ndarray, warped_seg: np.ndarray, label_map: dict[int, str], spacing: tuple[float, float, float], case_id: str, frame: int) -> dict[str, float]:
    """Compute foreground and per-label segmentation metrics with verbose logs."""
    out: dict[str, float] = {}
    items = [(-1, "foreground")]+[(k, v) for k, v in label_map.items() if k != 0]
    for label, organ in items:
        fm = fixed_seg > 0 if label == -1 else fixed_seg == label
        mm = moving_seg > 0 if label == -1 else moving_seg == label
        wm = warped_seg > 0 if label == -1 else warped_seg == label
        LOGGER.info("[SEG] case=%s, frame=%s, organ=%s, label=%s", case_id, frame, organ, label)
        LOGGER.info("[SEG] fixed voxels=%s, moving voxels=%s, warped voxels=%s", int(fm.sum()), int(mm.sum()), int(wm.sum()))
        for suffix, mask in [("moving_fixed", mm), ("warped_fixed", wm)]:
            res = pair_metrics(fm, mask, spacing)
            for metric, val in res.items(): out[f"{metric}_{organ}_{suffix}"] = val
        LOGGER.info("[SEG] moving-fixed dice=%s, iou=%s, hd95=%s, assd=%s", out[f"dice_{organ}_moving_fixed"], out[f"iou_{organ}_moving_fixed"], out[f"hd95_{organ}_moving_fixed"], out[f"assd_{organ}_moving_fixed"])
        LOGGER.info("[SEG] warped-fixed dice=%s, iou=%s, hd95=%s, assd=%s", out[f"dice_{organ}_warped_fixed"], out[f"iou_{organ}_warped_fixed"], out[f"hd95_{organ}_warped_fixed"], out[f"assd_{organ}_warped_fixed"])
    return out
