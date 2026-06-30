"""Vertebra NCC metrics from multi-label segmentations."""
from __future__ import annotations
import logging
import numpy as np
from .image_metrics import normalized_cross_correlation_similarity
from .motion_metrics import largest_component_bbox, _crop
from .gpu_utils import device_name
LOGGER = logging.getLogger("registration_metrics")
VERTEBRA_LABELS = [40,41,42,43,44,45,46,47,48,57,64]

def compute_vertebra_ncc(fixed, moving, warped, fixed_seg, moving_seg, warped_seg, case_id: str, frame: int, roi_mode: str = "fixed", row_index=None, device="cpu") -> dict[str, float]:
    """Compute moving-fixed and warped-fixed NCC inside fixed or fixed/warped union vertebra ROI."""
    LOGGER.info("[VERTEBRA] case=%s, frame=%s", case_id, frame); LOGGER.info("[VERTEBRA] labels used=%s", VERTEBRA_LABELS)
    fm=np.isin(fixed_seg, VERTEBRA_LABELS); mm=np.isin(moving_seg, VERTEBRA_LABELS); wm=np.isin(warped_seg, VERTEBRA_LABELS)
    LOGGER.info("[VERTEBRA] mask nonzero fixed/moving/warped=%s/%s/%s", int(fm.sum()), int(mm.sum()), int(wm.sum()))
    roi = fm | wm if roi_mode == "union" else fm
    bbox = largest_component_bbox(roi, case_id, frame, "vertebra")
    if bbox is None: return {"VertebraNCC_moving_fixed": float("nan"), "VertebraNCC_warped_fixed": float("nan")}
    fr=_crop(fixed,bbox); mr=_crop(moving,bbox); wr=_crop(warped,bbox)
    LOGGER.info("[VERTEBRA] bbox=%s", bbox); LOGGER.info("[VERTEBRA] roi shape fixed=%s, moving=%s, warped=%s", fr.shape, mr.shape, wr.shape)
    LOGGER.info("[ORGAN METRIC] case_id=%s row=%s frame=%s organ=vertebrae labels=%s metric=VertebraNCC pair=moving-fixed device=%s bbox=%s", case_id, row_index, frame, VERTEBRA_LABELS, device_name(device), bbox)
    LOGGER.info("[ORGAN METRIC] case_id=%s row=%s frame=%s organ=vertebrae labels=%s metric=VertebraNCC pair=warped-fixed device=%s bbox=%s", case_id, row_index, frame, VERTEBRA_LABELS, device_name(device), bbox)
    out={"VertebraNCC_moving_fixed": normalized_cross_correlation_similarity(mr, fr, device=device, metric_name="VertebraNCC", pair="moving-fixed", case_id=case_id, row_index=row_index, frame=frame), "VertebraNCC_warped_fixed": normalized_cross_correlation_similarity(wr, fr, device=device, metric_name="VertebraNCC", pair="warped-fixed", case_id=case_id, row_index=row_index, frame=frame)}
    LOGGER.info("[VERTEBRA] NCC moving-fixed=%s, warped-fixed=%s", out["VertebraNCC_moving_fixed"], out["VertebraNCC_warped_fixed"])
    return out
