"""Organ ROI bounding boxes, iterative NCC matching, and motion metrics."""
from __future__ import annotations

import logging
from copy import deepcopy
import numpy as np
import pandas as pd
from skimage.measure import label as cc_label
from .config import DIRECTIONS
from .image_metrics import normalized_cross_correlation_similarity
from .orientation_utils import convert_voxel_shift_to_physical_ap_rl_si
from .gpu_utils import torch, to_tensor_np, device_name

LOGGER = logging.getLogger("registration_metrics")
BBox = tuple[int, int, int, int, int, int]


def bbox_center(b: BBox) -> np.ndarray:
    """Return voxel center of bbox."""
    return np.array([(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2], dtype=float)


def largest_component_bbox(mask: np.ndarray, case_id: str, frame: int, organ: str) -> BBox | None:
    """Keep largest 6-connected component and return bbox as min/max per array axis."""
    LOGGER.info("[BOUND] case=%s, frame=%s, organ=%s", case_id, frame, organ)
    nz = int(np.count_nonzero(mask)); LOGGER.info("[BOUND] mask shape=%s, nonzero voxels=%s", mask.shape, nz)
    if nz == 0: LOGGER.info("[BOUND] organ_mask_empty"); return None
    lab, n = cc_label(mask.astype(bool), connectivity=1, return_num=True); LOGGER.info("[BOUND] number of connected components=%s", n)
    counts = np.bincount(lab.ravel()); counts[0] = 0; idx = int(np.argmax(counts)); comp = lab == idx
    pts = np.argwhere(comp); mins = pts.min(axis=0); maxs = pts.max(axis=0) + 1
    bbox = (int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1]), int(mins[2]), int(maxs[2]))
    LOGGER.info("[BOUND] largest component size=%s", int(counts[idx]))
    LOGGER.info("[BOUND] bbox voxel axis0=(%s,%s), axis1=(%s,%s), axis2=(%s,%s)", *bbox)
    return bbox


def _axis_range(b: BBox, ax: int) -> tuple[int, int]:
    return (b[2*ax], b[2*ax+1])


def _set_axis(b: list[int], ax: int, lo: int, hi: int) -> None:
    b[2*ax] = lo; b[2*ax+1] = hi


def equal_range(moving_bound: BBox, target_bound: BBox, image_shape: tuple[int, int, int]) -> tuple[BBox, BBox]:
    """Expand moving/target bbox per axis to equal ranges while clamping to image boundaries."""
    LOGGER.info("[EQUAL_RANGE] before moving bbox=%s", moving_bound); LOGGER.info("[EQUAL_RANGE] before target bbox=%s", target_bound); LOGGER.info("[EQUAL_RANGE] image_shape=%s", image_shape)
    mb = list(moving_bound); tb = list(target_bound)
    for ax in range(3):
        mlo, mhi = _axis_range(tuple(mb), ax); tlo, thi = _axis_range(tuple(tb), ax)
        mr, tr = mhi - mlo, thi - tlo; mc, tc = (mlo + mhi) / 2, (tlo + thi) / 2
        desired = max(mr, tr)
        def expand(center: float) -> tuple[int, int]:
            lo = int(round(center - desired / 2)); hi = lo + desired
            if lo < 0: hi -= lo; lo = 0
            if hi > image_shape[ax]: lo -= hi - image_shape[ax]; hi = image_shape[ax]
            return max(0, lo), min(image_shape[ax], hi)
        if mr < tr:
            lo, hi = expand(mc); _set_axis(mb, ax, lo, hi)
        elif tr < mr:
            lo, hi = expand(tc); _set_axis(tb, ax, lo, hi)
    out = tuple(mb), tuple(tb)
    if any(out[0][2*i] >= out[0][2*i+1] or out[1][2*i] >= out[1][2*i+1] for i in range(3)):
        raise ValueError(f"[EQUAL_RANGE] invalid bbox after clamp: {out}")
    LOGGER.info("[EQUAL_RANGE] after moving bbox=%s", out[0]); LOGGER.info("[EQUAL_RANGE] after target bbox=%s", out[1])
    return out


def _crop(img: np.ndarray, b: BBox, block: int = 0) -> np.ndarray | None:
    lo = [b[0]-block, b[2]-block, b[4]-block]; hi = [b[1]+block, b[3]+block, b[5]+block]
    if any(lo[i] < 0 or hi[i] > img.shape[i] or lo[i] >= hi[i] for i in range(3)): return None
    return img[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def _shift(b: BBox, ax: int, step: int) -> BBox:
    x = list(b); x[2*ax] += step; x[2*ax+1] += step; return tuple(x)


def match_ncc(reference_image: np.ndarray, reference_bound: BBox, target_image: np.ndarray, target_bound: BBox, affine: np.ndarray, case_id: str, frame: int, organ: str, mode: str, block: int = 5, length: int = 24, max_iter: int = 20, row_index=None, device="cpu", ncc_batch_size: int = 64, invalid_candidate_policy: str = "nan") -> BBox | None:
    """Iteratively match target ROI to reference ROI by positive NCC, selecting argmax candidates."""
    half = length // 2
    LOGGER.info("[MATCH_NCC START] case=%s, frame=%s, organ=%s, mode=%s", case_id, frame, organ, mode)
    LOGGER.info("[MATCH_NCC PARAM] block=%s, length=%s, half_length=%s, max_iter=%s, ncc_batch_size=%s, device=%s invalid_candidate_policy=%s", block, length, half, max_iter, ncc_batch_size, device_name(device), invalid_candidate_policy)
    LOGGER.info("[MATCH_NCC INPUT] ref image shape=%s, target image shape=%s", reference_image.shape, target_image.shape)
    LOGGER.info("[MATCH_NCC INPUT] ref bbox=%s, target bbox=%s", reference_bound, target_bound)
    rb, tb = equal_range(reference_bound, target_bound, reference_image.shape[:3])
    adjusted = min(block, *(rb[2*i] for i in range(3)), *(reference_image.shape[i]-rb[2*i+1] for i in range(3)))
    adjusted = max(0, int(adjusted)); ref_roi = _crop(reference_image, rb, adjusted)
    LOGGER.info("[MATCH_NCC ROI] reference roi shape=%s, adjusted block=%s", None if ref_roi is None else ref_roi.shape, adjusted)
    if ref_roi is None or np.nanstd(ref_roi) < 1e-12: return None
    tmp = tb; collect = [tmp]; collect_ncc: list[list[float]] = []
    for it in range(max_iter):
        LOGGER.info("[MATCH_NCC ITER] iter=%s", it); LOGGER.info("[MATCH_NCC ITER] current target bbox=%s", tmp)
        scores = []
        candidates = []
        for ax in range(3):
            for step in range(-half, half + 1):
                cand = _shift(tmp, ax, step); roi = _crop(target_image, cand, adjusted); valid = roi is not None and roi.shape == ref_roi.shape
                if not valid and invalid_candidate_policy == "zero_roi":
                    roi = np.zeros_like(ref_roi); valid = True
                candidates.append((ax, step, valid, roi))
        for start in range(0, len(candidates), ncc_batch_size):
            batch = candidates[start:start+ncc_batch_size]
            LOGGER.info("[GPU] metric=NCCMove organ=%s candidate_batch=%s batch_size=%s device=%s", organ, start // ncc_batch_size, len(batch), device_name(device))
            valid_rois = [x[3] for x in batch if x[2]]
            batch_scores = []
            if valid_rois and torch is not None and str(device) != "cpu":
                try:
                    ref_t = to_tensor_np(np.stack([ref_roi] * len(valid_rois)), device)
                    tgt_t = to_tensor_np(np.stack(valid_rois), device)
                    rf = ref_t.reshape(len(valid_rois), -1); tf = tgt_t.reshape(len(valid_rois), -1)
                    rf = rf - rf.mean(dim=1, keepdim=True); tf = tf - tf.mean(dim=1, keepdim=True)
                    denom = torch.std(rf, dim=1, unbiased=False) * torch.std(tf, dim=1, unbiased=False)
                    vals = torch.mean(rf * tf, dim=1) / denom
                    batch_scores = [float(v.detach().cpu()) if bool(torch.isfinite(v)) and float(d.detach().cpu()) >= 1e-12 else float("nan") for v, d in zip(vals, denom)]
                except RuntimeError as e:
                    LOGGER.warning("[GPU FALLBACK] metric=NCCMove case_id=%s reason=%s", case_id, e)
                    batch_scores = [normalized_cross_correlation_similarity(ref_roi, roi, metric_name="NCCMove", pair=mode, case_id=case_id) for roi in valid_rois]
            else:
                batch_scores = [normalized_cross_correlation_similarity(ref_roi, roi, metric_name="NCCMove", pair=mode, case_id=case_id) for roi in valid_rois]
            vi = iter(batch_scores)
            for ax, step, valid, roi in batch:
                ncc = next(vi) if valid else float("nan")
                scores.append(ncc); LOGGER.debug("[MATCH_NCC CAND] axis=axis%s, step=%s, valid=%s, ncc=%s", ax, step, valid, ncc)
        n_finite = int(np.isfinite(scores).sum()); n_nan = int(len(scores) - n_finite)
        LOGGER.info("[MATCH_NCC SUMMARY] case=%s row=%s frame=%s organ=%s mode=%s iter=%s n_candidates=%s n_finite=%s n_nan=%s", case_id, row_index, frame, organ, mode, it, len(scores), n_finite, n_nan)
        if not np.isfinite(scores).any():
            LOGGER.info("[MATCH_NCC SKIP] case=%s row=%s frame=%s organ=%s mode=%s reason=all_candidate_scores_nan metric_columns_set_to_nan=True", case_id, row_index, frame, organ, mode)
            return None
        collect_ncc.append(scores); arr = np.asarray(scores, dtype=float)
        best_steps = []
        for ax in range(3):
            sl = arr[ax*(length+1):(ax+1)*(length+1)]
            finite = np.isfinite(sl)
            if finite.any():
                best_idx = int(np.nanargmax(sl)); best_step = best_idx - half; axis_status = "valid"; best_score_axis = float(sl[best_idx])
            else:
                # Old code avoided all-NaN axis segments by substituting invalid candidates with zero ROIs.
                # New strict NaN policy keeps invalid candidates as NaN, so an all-NaN axis uses zero step.
                best_step = 0; axis_status = "all_nan_use_zero_step"; best_score_axis = float("nan")
            best_steps.append(best_step)
            LOGGER.info("[MATCH_NCC AXIS] case=%s row=%s frame=%s organ=%s mode=%s iter=%s axis=%s n_candidates=%s n_finite=%s n_nan=%s best_step=%s status=%s best_score=%s", case_id, row_index, frame, organ, mode, it, ax, len(sl), int(finite.sum()), int(len(sl)-finite.sum()), best_step, axis_status, best_score_axis)
        center_ncc = arr[half] if len(arr) > half else float("nan")
        if not np.isfinite(center_ncc): LOGGER.info("[MATCH_NCC CENTER] case=%s row=%s frame=%s organ=%s mode=%s iter=%s center_score=nan", case_id, row_index, frame, organ, mode, it)
        best_ncc = float(np.nanmax(arr)); proposed = tmp
        for ax, st in enumerate(best_steps): proposed = _shift(proposed, ax, st)
        flag = sum(abs(s) for s in best_steps); cycle = proposed in collect; one_side = False
        LOGGER.info("[MATCH_NCC SUMMARY] iter=%s, center_ncc=%s, best_axis0_step=%s, best_axis1_step=%s, best_axis2_step=%s, best_ncc=%s", it, center_ncc, *best_steps, best_ncc)
        LOGGER.info("[MATCH_NCC UPDATE] flag_converge=%s, proposed bbox=%s", flag, proposed)
        if flag == 0: break
        if cycle or (len(collect_ncc) >= 2 and np.nanmax(collect_ncc[-2]) > center_ncc):
            one_side = True; prev = np.asarray(collect_ncc[-2] if len(collect_ncc) >= 2 else scores); idx = int(np.nanargmax(prev)); proposed = _shift(tmp, idx // (length+1), idx % (length+1) - half)
        LOGGER.info("[MATCH_NCC CYCLE] cycle detected=%s, one-side update=%s", cycle, one_side)
        tmp = proposed; collect.append(tmp)
    center = bbox_center(tmp); phys = convert_voxel_shift_to_physical_ap_rl_si(center, affine)
    LOGGER.info("[MATCH_NCC END] converged=%s, iter=%s, final bbox=%s, final center voxel=%s, final center AP/RL/SI mm=%s", flag == 0, it, tmp, center, phys)
    return tmp



def _clamp_bbox_to_shape(bbox: BBox | None, shape: tuple[int, ...]) -> BBox | None:
    """Return a copied bbox clamped to image shape, or None if invalid after clamp."""
    if bbox is None:
        return None
    b = list(deepcopy(bbox))
    for ax in range(3):
        b[2 * ax] = max(0, min(int(b[2 * ax]), int(shape[ax])))
        b[2 * ax + 1] = max(0, min(int(b[2 * ax + 1]), int(shape[ax])))
        if b[2 * ax] >= b[2 * ax + 1]:
            return None
    return tuple(b)


def choose_bboxes_with_volume_fallback(reference_mask: np.ndarray, target_mask: np.ndarray, reference_bbox: BBox | None, target_bbox: BBox | None, reference_name: str, target_name: str, organ: str, case_id: str, frame: int, row_index, reference_image_shape: tuple[int, ...], target_image_shape: tuple[int, ...], reference_affine: np.ndarray, target_affine: np.ndarray, mode: str, min_mask_volume_voxels: int = 20, severe_volume_ratio_threshold: float = 0.20) -> dict:
    """Choose reference/target bboxes with volume-based fallback for unreliable masks."""
    rv = int(np.count_nonzero(reference_mask)); tv = int(np.count_nonzero(target_mask))
    larger = max(rv, tv); smaller = min(rv, tv)
    ratio = float(smaller / larger) if larger > 0 else float("nan")
    pair = f"{reference_name}-{target_name}"
    ref_invalid = rv < min_mask_volume_voxels
    tgt_invalid = tv < min_mask_volume_voxels
    if larger < min_mask_volume_voxels:
        status = "both_invalid"
    elif ref_invalid or (rv < tv and ratio < severe_volume_ratio_threshold):
        status = f"{reference_name}_unreliable"
    elif tgt_invalid or (tv < rv and ratio < severe_volume_ratio_threshold):
        status = f"{target_name}_unreliable"
    else:
        status = "ok"
    LOGGER.info("[MOTION MASK CHECK] case=%s row=%s frame=%s organ=%s mode=%s pair=%s %s_volume=%s %s_volume=%s ratio=%s threshold=%s status=%s", case_id, row_index, frame, organ, mode, pair, reference_name, rv, target_name, tv, ratio, severe_volume_ratio_threshold, status)
    result = {"reference_bbox": _clamp_bbox_to_shape(reference_bbox, reference_image_shape), "target_bbox": _clamp_bbox_to_shape(target_bbox, target_image_shape), "reference_volume": rv, "target_volume": tv, "volume_ratio": ratio, "fallback_used": False, "fallback_side": None, "fallback_reason": "", "valid": True, "skip_reason": ""}
    if status == "both_invalid":
        result.update(valid=False, skip_reason="both masks missing_or_too_small")
        LOGGER.warning("[MOTION SKIP] case=%s row=%s frame=%s organ=%s mode=%s reason=both masks missing_or_too_small %s_volume=%s %s_volume=%s metric_columns_set_to_nan=True", case_id, row_index, frame, organ, mode, reference_name, rv, target_name, tv)
        return result
    shape_compatible = tuple(reference_image_shape[:3]) == tuple(target_image_shape[:3])
    affine_compatible = np.allclose(reference_affine, target_affine, atol=1e-3, rtol=1e-3)
    if status != "ok" and (not shape_compatible or not affine_compatible):
        result.update(valid=False, skip_reason="shape_or_affine_incompatible_for_bbox_fallback")
        LOGGER.warning("[MOTION SKIP] case=%s row=%s frame=%s organ=%s mode=%s reason=shape_or_affine_incompatible_for_bbox_fallback %s_shape=%s %s_shape=%s", case_id, row_index, frame, organ, mode, reference_name, reference_image_shape, target_name, target_image_shape)
        return result
    if status == f"{reference_name}_unreliable":
        fallback_bbox = _clamp_bbox_to_shape(target_bbox, reference_image_shape)
        if fallback_bbox is None or result["target_bbox"] is None:
            result.update(valid=False, skip_reason="fallback_bbox_invalid")
            return result
        reason = f"{reference_name} mask invalid or much smaller; using {target_name} bbox as {reference_name} initial bbox"
        result.update(reference_bbox=fallback_bbox, fallback_used=True, fallback_side="reference", fallback_reason=reason)
        LOGGER.warning("[MOTION BBOX FALLBACK] case=%s row=%s frame=%s organ=%s mode=%s unreliable_side=%s reliable_side=%s reason=%r %s_volume=%s %s_volume=%s ratio=%s using_bbox_from=%s_seg applying_to=%s_image initial_reference_bbox=%s", case_id, row_index, frame, organ, mode, reference_name, target_name, f"{reference_name} mask invalid or much smaller", reference_name, rv, target_name, tv, ratio, target_name, reference_name, fallback_bbox)
    elif status == f"{target_name}_unreliable":
        fallback_bbox = _clamp_bbox_to_shape(reference_bbox, target_image_shape)
        if result["reference_bbox"] is None or fallback_bbox is None:
            result.update(valid=False, skip_reason="fallback_bbox_invalid")
            return result
        reason = f"{target_name} mask invalid or much smaller; using {reference_name} bbox as {target_name} initial bbox"
        result.update(target_bbox=fallback_bbox, fallback_used=True, fallback_side="target", fallback_reason=reason)
        LOGGER.warning("[MOTION BBOX FALLBACK] case=%s row=%s frame=%s organ=%s mode=%s unreliable_side=%s reliable_side=%s reason=%r %s_volume=%s %s_volume=%s ratio=%s using_bbox_from=%s_seg applying_to=%s_image initial_target_bbox=%s", case_id, row_index, frame, organ, mode, target_name, reference_name, f"{target_name} mask invalid or much smaller", reference_name, rv, target_name, tv, ratio, reference_name, target_name, fallback_bbox)
    if result["reference_bbox"] is None or result["target_bbox"] is None:
        result.update(valid=False, skip_reason="bbox_invalid_after_clamp")
    return result

def pearson_safe(pred, gt) -> float:
    """Pearson correlation with finite filtering and constant-vector handling."""
    p = np.asarray(pred, dtype=float); g = np.asarray(gt, dtype=float); m = np.isfinite(p) & np.isfinite(g); p = p[m]; g = g[m]
    if len(p) < 2: return float("nan")
    if np.std(p) < 1e-12 or np.std(g) < 1e-12: return 1.0 if np.allclose(p, g, equal_nan=False) else float("nan")
    return float(np.corrcoef(p, g)[0, 1])


def motion_summary(pred, gt) -> dict[str, float]:
    """Return AMD/RMSE/MAPE/PCC over finite pred/gt pairs."""
    p = np.asarray(pred, dtype=float); g = np.asarray(gt, dtype=float); m = np.isfinite(p) & np.isfinite(g)
    if not np.any(m): return {"amd": float("nan"), "rmse": float("nan"), "mape": float("nan"), "pcc": float("nan")}
    e = p[m] - g[m]
    return {"amd": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e**2))), "mape": float(np.mean(np.abs(e) / np.maximum(np.abs(g[m]), 1e-6)) * 100), "pcc": pearson_safe(p, g)}


def _find_col(row: pd.Series, organ: str, direction: str, gt: bool) -> str | None:
    cand = ([f"{organ}NCCMove_{direction}_GT", f"{organ}NCCMoveGT_{direction}", f"{organ}NCCMoveGT{direction}", f"{organ}NCCMove_{direction}GT"] if gt else [f"{organ}NCCMove_{direction}", f"{organ}NCCMove{direction}", f"{organ}NCCMove_{direction}_Warped", f"{organ}NCCMoveWarped{direction}"])
    return next((c for c in cand if c in row.index), None)


def compute_frame_motion_metrics(df: pd.DataFrame, relative_to_first_frame: bool = False) -> pd.DataFrame:
    """Compute row/frame-level motion metrics from NCCMove columns."""
    rows = []
    for _, row in df.iterrows():
        organ = row.get("Organ", row.get("organ", "")); case = row.get("CaseID", row.get("case_id", "")); frame = row.get("Frame", 0)
        out = row.to_dict(); pred=[]; gt=[]
        for d in DIRECTIONS:
            pc = _find_col(row, organ, d, False); gc = _find_col(row, organ, d, True)
            pv = float(row[pc]) if pc and pd.notna(row[pc]) else np.nan; gv = float(row[gc]) if gc and pd.notna(row[gc]) else np.nan
            pred.append(pv); gt.append(gv); out[f"{organ}RelativeMove_{d}"] = pv; out[f"{organ}RelativeMoveGT_{d}"] = gv; out[f"{organ}RelativeError_{d}"] = pv-gv; out[f"{organ}RelativeAbsError_{d}"] = abs(pv-gv); out[f"MovementError_{d}"] = abs(pv-gv)
        summ = motion_summary(pred, gt); out["MovementError"] = summ["amd"]; out["MotionAMD_AllDirections"] = summ["amd"]; out["MotionRMSE_AllDirections"] = summ["rmse"]; out["MotionMAPE_percent_AllDirections"] = summ["mape"]; out["MotionPCC_AllDirections"] = summ["pcc"]
        pa = float(np.linalg.norm(np.nan_to_num(pred, nan=0.0))); ga = float(np.linalg.norm(np.nan_to_num(gt, nan=0.0))); out[f"{organ}RelativeMove_Amplitude"] = pa; out[f"{organ}RelativeMoveGT_Amplitude"] = ga; out[f"{organ}AmplitudeAbsError"] = abs(pa-ga); out["AmplitudeAMD"] = abs(pa-ga)
        LOGGER.info("[MOTION FRAME] case=%s, frame=%s, organ=%s", case, frame, organ); LOGGER.info("[MOTION FRAME] pred AP/RL/SI=%s", pred); LOGGER.info("[MOTION FRAME] gt AP/RL/SI=%s", gt); LOGGER.info("[MOTION FRAME] abs error AP/RL/SI=%s", [out[f"MovementError_{d}"] for d in DIRECTIONS]); LOGGER.info("[MOTION FRAME] MovementError=%s, RMSE=%s, MAPE=%s, PCC=%s, AmplitudeAMD=%s", out["MovementError"], out["MotionRMSE_AllDirections"], out["MotionMAPE_percent_AllDirections"], out["MotionPCC_AllDirections"], out["AmplitudeAMD"])
        rows.append(out)
    return pd.DataFrame(rows)


def compute_case_motion_metrics(frame_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frame motion metrics by case and output per-direction/all-direction summaries."""
    group_cols = [c for c in ["Method", "Center", "Organ", "Task", "AnalysisGroup", "CaseID", "MovingImagePath", "FixedImagePath", "moving_img_path", "fixed_img_path"] if c in frame_df.columns]
    outs=[]
    for keys, g in frame_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple): keys=(keys,)
        out=dict(zip(group_cols, keys)); organ=out.get("Organ", g.iloc[0].get("Organ", "")); pred=[]; gt=[]; valid=[]
        for d in DIRECTIONS:
            pc=f"{organ}RelativeMove_{d}"; gc=f"{organ}RelativeMoveGT_{d}"
            if pc in g and gc in g and (g[pc].notna() & g[gc].notna()).any():
                valid.append(d); pred.extend(g[pc].to_numpy(float)); gt.extend(g[gc].to_numpy(float)); s=motion_summary(g[pc], g[gc]); out[f"MotionPCC_{d}"]=s["pcc"]; out[f"MotionAMD_{d}"]=s["amd"]; out[f"MotionRMSE_{d}"]=s["rmse"]; out[f"MotionMAPE_percent_{d}"]=s["mape"]; out[f"MovementError_{d}"]=s["amd"]
        s=motion_summary(pred, gt); out.update({"MovementError":s["amd"], "MotionPCC_AllDirections":s["pcc"], "MotionAMD_AllDirections":s["amd"], "MotionRMSE_AllDirections":s["rmse"], "MotionMAPE_percent_AllDirections":s["mape"], "AmplitudeAMD":float(g.get("AmplitudeAMD", pd.Series([np.nan])).mean())})
        LOGGER.info("[MOTION CASE] case=%s, organ=%s, n_frames=%s, valid_directions=%s", out.get("CaseID"), organ, len(g), valid); LOGGER.info("[MOTION CASE] MovementError=%s, AP=%s, RL=%s, SI=%s", out["MovementError"], out.get("MovementError_AP"), out.get("MovementError_RL"), out.get("MovementError_SI")); LOGGER.info("[MOTION CASE] PCC=%s, AMD=%s, RMSE=%s, MAPE=%s, AmplitudeAMD=%s", out["MotionPCC_AllDirections"], out["MotionAMD_AllDirections"], out["MotionRMSE_AllDirections"], out["MotionMAPE_percent_AllDirections"], out["AmplitudeAMD"])
        outs.append(out)
    return pd.DataFrame(outs)

def compute_organ_ncc_moves(fixed, moving, warped, fixed_seg, moving_seg, warped_seg, label_map: dict[int, str], affine: np.ndarray, case_id: str, frame: int, organs: list[str] | None = None, row_index=None, device="cpu", ncc_batch_size: int = 64, min_mask_volume_voxels: int = 20, severe_volume_ratio_threshold: float = 0.20) -> dict[str, float]:
    """Compute NCCMove=moving->warped and NCCMoveGT=moving->fixed for requested organs."""
    out: dict[str, float] = {}; organs = organs or ["liver", "spleen", "pancreas", "kidney_left", "kidney_right"]
    inv = {v: k for k, v in label_map.items()}
    for organ in organs:
        labels = [inv.get("kidney_left"), inv.get("kidney_right")] if organ == "kidney" else [inv.get(organ)]
        labels = [int(x) for x in labels if x is not None]
        LOGGER.info("[BOUND] case=%s, frame=%s, organ=%s labels=%s", case_id, frame, organ, labels)
        if not labels: continue
        moving_mask = np.isin(moving_seg, labels); fixed_mask = np.isin(fixed_seg, labels); warped_mask = np.isin(warped_seg, labels)
        mb = largest_component_bbox(moving_mask, case_id, frame, organ)
        fb = largest_component_bbox(fixed_mask, case_id, frame, organ)
        wb = largest_component_bbox(warped_mask, case_id, frame, organ)
        pred_choice = choose_bboxes_with_volume_fallback(moving_mask, warped_mask, mb, wb, "moving", "warped", organ, case_id, frame, row_index, moving.shape, warped.shape, affine, affine, "pred", min_mask_volume_voxels, severe_volume_ratio_threshold)
        out[f"{organ}NCCMoveFallbackUsed"] = pred_choice["fallback_used"]; out[f"{organ}NCCMoveFallbackReason"] = pred_choice["fallback_reason"] or pred_choice["skip_reason"]; out[f"{organ}NCCMoveReferenceMaskVolume"] = pred_choice["reference_volume"]; out[f"{organ}NCCMoveTargetMaskVolume"] = pred_choice["target_volume"]; out[f"{organ}NCCMoveMaskVolumeRatio"] = pred_choice["volume_ratio"]
        pred = None
        if pred_choice["valid"]:
            LOGGER.info("[ORGAN METRIC] case_id=%s row=%s frame=%s organ=%s label=%s metric=NCCMove mode=moving-to-warped device=%s bbox_moving=%s bbox_warped=%s", case_id, row_index, frame, organ, labels, device_name(device), pred_choice["reference_bbox"], pred_choice["target_bbox"])
            pred = match_ncc(moving, pred_choice["reference_bbox"], warped, pred_choice["target_bbox"], affine, case_id, frame, organ, "pred", row_index=row_index, device=device, ncc_batch_size=ncc_batch_size)
        else:
            for d in DIRECTIONS: out[f"{organ}NCCMove_{d}"] = float("nan")
        gt_choice = choose_bboxes_with_volume_fallback(moving_mask, fixed_mask, mb, fb, "moving", "fixed", organ, case_id, frame, row_index, moving.shape, fixed.shape, affine, affine, "gt", min_mask_volume_voxels, severe_volume_ratio_threshold)
        out[f"{organ}NCCMoveGTFallbackUsed"] = gt_choice["fallback_used"]; out[f"{organ}NCCMoveGTFallbackReason"] = gt_choice["fallback_reason"] or gt_choice["skip_reason"]; out[f"{organ}NCCMoveGTReferenceMaskVolume"] = gt_choice["reference_volume"]; out[f"{organ}NCCMoveGTTargetMaskVolume"] = gt_choice["target_volume"]; out[f"{organ}NCCMoveGTMaskVolumeRatio"] = gt_choice["volume_ratio"]
        gt = None
        if gt_choice["valid"]:
            LOGGER.info("[ORGAN METRIC] case_id=%s row=%s frame=%s organ=%s label=%s metric=NCCMoveGT mode=moving-to-fixed device=%s bbox_moving=%s bbox_fixed=%s", case_id, row_index, frame, organ, labels, device_name(device), gt_choice["reference_bbox"], gt_choice["target_bbox"])
            gt = match_ncc(moving, gt_choice["reference_bbox"], fixed, gt_choice["target_bbox"], affine, case_id, frame, organ, "gt", row_index=row_index, device=device, ncc_batch_size=ncc_batch_size)
        else:
            for d in DIRECTIONS: out[f"{organ}NCCMoveGT_{d}"] = float("nan")
        mc = bbox_center(pred_choice["reference_bbox"] or mb) if (pred_choice["reference_bbox"] or mb) is not None else None
        if pred is not None and mc is not None:
            disp = convert_voxel_shift_to_physical_ap_rl_si(bbox_center(pred) - mc, affine)
            for d in DIRECTIONS: out[f"{organ}NCCMove_{d}"] = disp[d]
        elif pred is None:
            for d in DIRECTIONS: out.setdefault(f"{organ}NCCMove_{d}", float("nan"))
        gt_mc = bbox_center(gt_choice["reference_bbox"] or mb) if (gt_choice["reference_bbox"] or mb) is not None else None
        if gt is not None and gt_mc is not None:
            disp = convert_voxel_shift_to_physical_ap_rl_si(bbox_center(gt) - gt_mc, affine)
            for d in DIRECTIONS: out[f"{organ}NCCMoveGT_{d}"] = disp[d]
        elif gt is None:
            for d in DIRECTIONS: out.setdefault(f"{organ}NCCMoveGT_{d}", float("nan"))
    return out
