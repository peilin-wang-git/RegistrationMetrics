"""Organ ROI bounding boxes, iterative NCC matching, and motion metrics."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from skimage.measure import label as cc_label
from .config import DIRECTIONS
from .image_metrics import normalized_cross_correlation_similarity
from .orientation_utils import convert_voxel_shift_to_physical_ap_rl_si

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


def match_ncc(reference_image: np.ndarray, reference_bound: BBox, target_image: np.ndarray, target_bound: BBox, affine: np.ndarray, case_id: str, frame: int, organ: str, mode: str, block: int = 5, length: int = 24, max_iter: int = 20) -> BBox | None:
    """Iteratively match target ROI to reference ROI by positive NCC, selecting argmax candidates."""
    half = length // 2
    LOGGER.info("[MATCH_NCC START] case=%s, frame=%s, organ=%s, mode=%s", case_id, frame, organ, mode)
    LOGGER.info("[MATCH_NCC PARAM] block=%s, length=%s, half_length=%s, max_iter=%s", block, length, half, max_iter)
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
        for ax in range(3):
            for step in range(-half, half + 1):
                cand = _shift(tmp, ax, step); roi = _crop(target_image, cand, adjusted); valid = roi is not None and roi.shape == ref_roi.shape
                ncc = normalized_cross_correlation_similarity(ref_roi, roi) if valid else float("nan")
                scores.append(ncc); LOGGER.debug("[MATCH_NCC CAND] axis=axis%s, step=%s, valid=%s, ncc=%s", ax, step, valid, ncc)
        if not np.isfinite(scores).any(): LOGGER.info("[MATCH_NCC END] all candidate NCC are nan"); return None
        collect_ncc.append(scores); arr = np.asarray(scores, dtype=float)
        best_steps = []
        for ax in range(3):
            sl = arr[ax*(length+1):(ax+1)*(length+1)]; best_steps.append(int(np.nanargmax(sl) - half))
        center_ncc = arr[half]
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

def compute_organ_ncc_moves(fixed, moving, warped, fixed_seg, moving_seg, warped_seg, label_map: dict[int, str], affine: np.ndarray, case_id: str, frame: int, organs: list[str] | None = None) -> dict[str, float]:
    """Compute NCCMove=moving->warped and NCCMoveGT=moving->fixed for requested organs."""
    out: dict[str, float] = {}; organs = organs or ["liver", "spleen", "pancreas", "kidney_left", "kidney_right"]
    inv = {v: k for k, v in label_map.items()}
    for organ in organs:
        labels = [inv.get("kidney_left"), inv.get("kidney_right")] if organ == "kidney" else [inv.get(organ)]
        labels = [int(x) for x in labels if x is not None]
        LOGGER.info("[BOUND] case=%s, frame=%s, organ=%s labels=%s", case_id, frame, organ, labels)
        if not labels: continue
        mb = largest_component_bbox(np.isin(moving_seg, labels), case_id, frame, organ)
        fb = largest_component_bbox(np.isin(fixed_seg, labels), case_id, frame, organ)
        wb = largest_component_bbox(np.isin(warped_seg, labels), case_id, frame, organ)
        if mb is None or fb is None or wb is None:
            out[f"{organ}NCCMove_status"] = "organ_mask_empty"; continue
        pred = match_ncc(moving, mb, warped, wb, affine, case_id, frame, organ, "pred")
        gt = match_ncc(moving, mb, fixed, fb, affine, case_id, frame, organ, "gt")
        mc = bbox_center(mb)
        if pred is not None:
            disp = convert_voxel_shift_to_physical_ap_rl_si(bbox_center(pred) - mc, affine)
            for d in DIRECTIONS: out[f"{organ}NCCMove_{d}"] = disp[d]
        if gt is not None:
            disp = convert_voxel_shift_to_physical_ap_rl_si(bbox_center(gt) - mc, affine)
            for d in DIRECTIONS: out[f"{organ}NCCMoveGT_{d}"] = disp[d]
    return out
