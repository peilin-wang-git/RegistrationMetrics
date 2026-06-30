"""Segmentation Dice, IoU, HD95, and ASSD metrics."""
from __future__ import annotations

import logging
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from .gpu_utils import torch, to_tensor_np, device_name
from .config import SEG_METRIC_ORGANS, SEG_MEAN_ORGANS

LOGGER = logging.getLogger("registration_metrics")


def dice_coefficient(a: np.ndarray, b: np.ndarray, device="cpu") -> float:
    """Return Dice; both empty masks score 1."""
    if torch is not None and str(device) != "cpu":
        try:
            ta = to_tensor_np(a, device, dtype=torch.bool); tb = to_tensor_np(b, device, dtype=torch.bool)
            sa = int(torch.sum(ta).detach().cpu()); sb = int(torch.sum(tb).detach().cpu())
            if sa == 0 and sb == 0: return 1.0
            if sa == 0 or sb == 0: return 0.0
            return float((2 * torch.sum(ta & tb).float() / (sa + sb)).detach().cpu())
        except RuntimeError as e:
            LOGGER.warning("[GPU FALLBACK] metric=Dice reason=%s", e)
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    sa = int(a.sum()); sb = int(b.sum())
    if sa == 0 and sb == 0: return 1.0
    if sa == 0 or sb == 0: return 0.0
    return float(2 * np.logical_and(a, b).sum() / (sa + sb))


def iou_score(a: np.ndarray, b: np.ndarray, device="cpu") -> float:
    """Return IoU; both empty masks score 1."""
    if torch is not None and str(device) != "cpu":
        try:
            ta = to_tensor_np(a, device, dtype=torch.bool); tb = to_tensor_np(b, device, dtype=torch.bool)
            union = int(torch.sum(ta | tb).detach().cpu())
            return 1.0 if union == 0 else float((torch.sum(ta & tb).float() / union).detach().cpu())
        except RuntimeError as e:
            LOGGER.warning("[GPU FALLBACK] metric=IoU reason=%s", e)
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


def pair_metrics(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float], device="cpu") -> dict[str, float]:
    """Compute Dice/IoU/HD95/ASSD for one mask pair."""
    return {"dice": dice_coefficient(a, b, device=device), "iou": iou_score(a, b, device=device), "hd95": hd95(a, b, spacing), "assd": assd(a, b, spacing)}


def compute_segmentation_metrics(fixed_seg: np.ndarray, moving_seg: np.ndarray, warped_seg: np.ndarray, label_map: dict[int, str], spacing: tuple[float, float, float], case_id: str, frame: int, row_index=None, device="cpu", seg_metric_organs: list[str] | None = None, seg_mean_organs: list[str] | None = None, verbose_seg_mean: bool = False, min_mask_volume_voxels: int = 20) -> dict[str, float]:
    """Compute selected per-organ segmentation metrics plus all-foreground-label mean metrics."""
    out: dict[str, float] = {}
    selected = set(seg_metric_organs or SEG_METRIC_ORGANS)
    mean_set = set(seg_mean_organs or SEG_MEAN_ORGANS)
    selected_items = [(k, v) for k, v in label_map.items() if k != 0 and v in selected]
    mean_items = [(k, v) for k, v in label_map.items() if k != 0 and v in mean_set]
    all_items = [(k, v) for k, v in label_map.items() if k != 0]
    mean_values: dict[tuple[str, str], list[float]] = {(metric, suffix): [] for metric in ["dice", "iou", "hd95", "assd"] for suffix in ["moving_fixed", "warped_fixed"]}
    LOGGER.info("[SEG METRIC] selected individual organ metrics organs=%s", [organ for _, organ in selected_items])
    LOGGER.info("[SEG METRIC] mean organ set organs=%s", [organ for _, organ in mean_items])
    LOGGER.info("[SEG METRIC] note: mean_*_all_organs_* uses SEG_MEAN_ORGANS heart-to-kidney organ set, not all foreground labels")
    for label, organ in all_items:
        output_individual = organ in selected
        use_for_mean = organ in mean_set
        if not output_individual and not use_for_mean:
            if verbose_seg_mean:
                LOGGER.debug("[SEG METRIC] organ=%s label=%s output_individual=False use_for_mean=False reason=outside heart-to-kidney organ set", organ, label)
            continue
        fm = fixed_seg == label
        mm = moving_seg == label
        wm = warped_seg == label
        LOGGER.info("[SEG] case=%s, frame=%s, organ=%s, label=%s", case_id, frame, organ, label)
        LOGGER.info("[METRIC MODE] metric=Dice mode=3D organ=%s label=%s", organ, label)
        LOGGER.info("[METRIC MODE] metric=IoU mode=3D organ=%s label=%s", organ, label)
        LOGGER.info("[METRIC MODE] metric=HD95 mode=3D spacing=%s", spacing)
        LOGGER.info("[METRIC MODE] metric=ASSD mode=3D spacing=%s", spacing)
        LOGGER.info("[SEG] fixed voxels=%s, moving voxels=%s, warped voxels=%s", int(fm.sum()), int(mm.sum()), int(wm.sum()))
        for suffix, mask in [("moving_fixed", mm), ("warped_fixed", wm)]:
            pair_label = suffix.replace("_", "-")
            if str(device) != "cpu" and output_individual:
                LOGGER.info("[GPU] metric=Dice organ=%s device=%s voxels_fixed=%s voxels_warped=%s", organ, device_name(device), int(fm.sum()), int(mask.sum()))
                LOGGER.info("[GPU] metric=HD95 organ=%s device=cpu reason=scipy_distance_transform", organ)
            fixed_volume = int(fm.sum()); target_volume = int(mask.sum())
            if fixed_volume < min_mask_volume_voxels or target_volume < min_mask_volume_voxels:
                target_name = pair_label.split("-")[0]
                LOGGER.info("[SEG SKIP] case=%s row=%s frame=%s organ=%s label=%s pair=%s reason=fixed_or_%s_mask_empty_or_too_small fixed_volume=%s %s_volume=%s metrics_set_to_nan=True", case_id, row_index, frame, organ, label, pair_label, target_name, fixed_volume, target_name, target_volume)
                res = {"dice": float("nan"), "iou": float("nan"), "hd95": float("nan"), "assd": float("nan")}
            else:
                res = pair_metrics(fm, mask, spacing, device=device)
            for metric, val in res.items():
                if use_for_mean:
                    mean_values[(metric, suffix)].append(val)
                metric_name = metric.upper() if metric in ["hd95", "assd"] else metric.capitalize()
                if output_individual:
                    LOGGER.info("[SEG METRIC] case_id=%s row=%s frame=%s organ=%s label=%s output_individual=True use_for_mean=%s metric=%s pair=%s value=%s", case_id, row_index, frame, organ, label, use_for_mean, metric_name, pair_label, val)
                    LOGGER.info("[ORGAN METRIC] case_id=%s row=%s frame=%s organ=%s label=%s metric=%s pair=%s device=%s fixed_voxels=%s target_voxels=%s spacing=%s", case_id, row_index, frame, organ, label, metric_name, pair_label, device_name(device) if metric in ["dice", "iou"] else "cpu", int(fm.sum()), int(mask.sum()), spacing)
                    out[f"{metric}_{organ}_{suffix}"] = val
                elif verbose_seg_mean:
                    LOGGER.debug("[SEG METRIC] case_id=%s row=%s frame=%s organ=%s label=%s output_individual=False use_for_mean=%s metric=%s pair=%s value=%s", case_id, row_index, frame, organ, label, use_for_mean, metric_name, pair_label, val)
                if metric in ["hd95", "assd"] and (int(fm.sum()) == 0 or int(mask.sum()) == 0) and not (int(fm.sum()) == 0 and int(mask.sum()) == 0):
                    LOGGER.info("[SKIP] case_id=%s row=%s frame=%s organ=%s metric=%s reason=%s mask empty", case_id, row_index, frame, organ, metric_name, "fixed" if int(fm.sum()) == 0 else "target")
        if output_individual:
            LOGGER.info("[SEG] moving-fixed dice=%s, iou=%s, hd95=%s, assd=%s", out[f"dice_{organ}_moving_fixed"], out[f"iou_{organ}_moving_fixed"], out[f"hd95_{organ}_moving_fixed"], out[f"assd_{organ}_moving_fixed"])
            LOGGER.info("[SEG] warped-fixed dice=%s, iou=%s, hd95=%s, assd=%s", out[f"dice_{organ}_warped_fixed"], out[f"iou_{organ}_warped_fixed"], out[f"hd95_{organ}_warped_fixed"], out[f"assd_{organ}_warped_fixed"])
    for (metric, suffix), values in mean_values.items():
        mean_value = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        column = f"mean_{metric}_all_organs_{suffix}"
        out[column] = mean_value
        based_on = int(np.isfinite(values).sum())
        LOGGER.info("[SEG METRIC] case_id=%s frame=%s %s=%s based_on_n_organs=%s", case_id, frame, column, mean_value, based_on)
    return out
