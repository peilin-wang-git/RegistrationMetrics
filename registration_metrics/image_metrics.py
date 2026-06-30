"""Global intensity metrics: NCC/LCC, NMI, SSIM, and MSE."""
from __future__ import annotations

import importlib.util
import logging
import numpy as np
from skimage.metrics import structural_similarity
from .gpu_utils import torch, to_tensor_np, device_name

LOGGER = logging.getLogger("registration_metrics")


def normalized_cross_correlation_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12, device="cpu", metric_name: str = "LCC", pair: str = "", case_id: str = "", row_index=None, frame=None) -> float:
    """Compute positive NCC similarity; larger is better and constants return NaN."""
    if torch is not None and str(device) != "cpu":
        try:
            LOGGER.info("[GPU] metric=%s pair=%s device=%s tensor_shape=%s", metric_name, pair, device_name(device), np.shape(a))
            x = to_tensor_np(a, device).flatten(); y = to_tensor_np(b, device).flatten()
            m = torch.isfinite(x) & torch.isfinite(y); x = x[m]; y = y[m]
            if x.numel() == 0 or torch.std(x) < eps or torch.std(y) < eps:
                return float("nan")
            return float(torch.mean((x - torch.mean(x)) * (y - torch.mean(y))) / (torch.std(x, unbiased=False) * torch.std(y, unbiased=False))).__float__()
        except RuntimeError as e:
            LOGGER.warning("[GPU FALLBACK] metric=%s case_id=%s reason=%s", metric_name, case_id, e)
    x = np.asarray(a, dtype=float).ravel(); y = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y); x = x[m]; y = y[m]
    if x.size == 0 or y.size == 0 or np.std(x) < eps or np.std(y) < eps:
        LOGGER.info("[GLOBAL] NCC constant/empty input; returning nan")
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std()))


def normalized_mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64, pair: str = "") -> float:
    """Compute sklearn-compatible normalized mutual information after shared binning."""
    x = np.asarray(a, dtype=float).ravel(); y = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y); x = x[m]; y = y[m]
    if x.size < 2:
        LOGGER.info("[NMI] pair=%s valid_voxels=%s min=nan max=nan value=nan", pair, x.size)
        return float("nan")
    combined = np.concatenate([x, y])
    mn = float(np.min(combined)); mx = float(np.max(combined))
    if mn == mx:
        LOGGER.info("[NMI] pair=%s valid_voxels=%s min=%s max=%s value=nan", pair, x.size, mn, mx)
        return float("nan")
    edges = np.histogram_bin_edges(combined, bins=bins)
    x_bins = np.digitize(x, edges[1:-1]); y_bins = np.digitize(y, edges[1:-1])
    if importlib.util.find_spec("sklearn") is None:
        raise ImportError("scikit-learn is required for NMI because NMI must match sklearn.metrics.normalized_mutual_info_score")
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(x_bins, y_bins, average_method="arithmetic")
    nmi = float(np.clip(nmi, 0.0, 1.0))
    LOGGER.info("[NMI] pair=%s valid_voxels=%s min=%s max=%s value=%s", pair, x.size, mn, mx, nmi)
    return nmi


def ssim_3d_volume(a: np.ndarray, b: np.ndarray) -> float:
    """Compute true 3D SSIM on a full volume with automatic odd win_size."""
    x = np.asarray(a, dtype=float); y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 3:
        LOGGER.info("[SSIM SKIP] reason=expected matching 3D volumes shapes=%s/%s", x.shape, y.shape)
        return float("nan")
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        LOGGER.info("[SSIM SKIP] reason=no finite voxel pairs")
        return float("nan")
    xmin = float(np.nanmin([np.nanmin(x[finite]), np.nanmin(y[finite])]))
    xmax = float(np.nanmax([np.nanmax(x[finite]), np.nanmax(y[finite])]))
    dr = xmax - xmin
    min_dim = int(min(x.shape))
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if dr <= 0 or win_size < 3:
        LOGGER.info("[SSIM SKIP] reason=3D volume too small or zero data_range data_range=%s win_size=%s shape=%s", dr, win_size, x.shape)
        return float("nan")
    try:
        return float(structural_similarity(x, y, data_range=dr, win_size=win_size, channel_axis=None))
    except ValueError as e:
        LOGGER.info("[SSIM SKIP] reason=3D SSIM failed error=%s", e)
        return float("nan")


def ssim_3d_slice_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Backward-compatible wrapper now computing true 3D volume SSIM, not slice means."""
    return ssim_3d_volume(a, b)


def mse(a: np.ndarray, b: np.ndarray, device="cpu", case_id: str = "", pair: str = "") -> float:
    """Compute mean squared error over finite pairs."""
    if torch is not None and str(device) != "cpu":
        try:
            LOGGER.info("[GPU] metric=MSE pair=%s device=%s tensor_shape=%s", pair, device_name(device), np.shape(a))
            x = to_tensor_np(a, device); y = to_tensor_np(b, device); m = torch.isfinite(x) & torch.isfinite(y)
            return float(torch.mean((x[m] - y[m]) ** 2).detach().cpu()) if bool(torch.any(m)) else float("nan")
        except RuntimeError as e:
            LOGGER.warning("[GPU FALLBACK] metric=MSE case_id=%s reason=%s", case_id, e)
    x = np.asarray(a, dtype=float); y = np.asarray(b, dtype=float); m = np.isfinite(x) & np.isfinite(y)
    return float(np.mean((x[m] - y[m]) ** 2)) if np.any(m) else float("nan")


def compute_global_metrics(fixed: np.ndarray, moving: np.ndarray, warped: np.ndarray, case_id: str, frame: int, bins: int = 64, row_index=None, device="cpu") -> dict[str, float]:
    """Compute fixed-vs-moving and fixed-vs-warped global metrics with verbose logs."""
    LOGGER.info("[GLOBAL] case=%s, frame=%s", case_id, frame)
    LOGGER.info("[GLOBAL] fixed shape=%s, moving shape=%s, warped shape=%s", fixed.shape, moving.shape, warped.shape)
    for metric in ["LCC", "NCC", "MSE", "NMI", "SSIM"]:
        LOGGER.info("[METRIC MODE] metric=%s mode=3D frame=%s", metric, frame)
    for metric in ["NMI", "SSIM", "LCC", "MSE"]:
        for pair in ["moving-fixed", "warped-fixed"]:
            LOGGER.info("[GLOBAL METRIC] case_id=%s row=%s frame=%s metric=%s pair=%s device=%s", case_id, row_index, frame, metric, pair, device_name(device))
    if str(device) != "cpu":
        LOGGER.info("[GPU] metric=NMI device=cpu reason=torch histogram not enabled")
        LOGGER.info("[GPU] metric=SSIM device=cpu reason=skimage SSIM CPU backend")
    LOGGER.info("[NMI] definition=sklearn.normalized_mutual_info_score average_method=arithmetic bins=%s mode=3D", bins)
    out = {
        "nmi_moving_fixed": normalized_mutual_information(moving, fixed, bins, pair="moving-fixed"), "nmi_warped_fixed": normalized_mutual_information(warped, fixed, bins, pair="warped-fixed"),
        "ssim_moving_fixed": ssim_3d_volume(moving, fixed), "ssim_warped_fixed": ssim_3d_volume(warped, fixed),
        "lcc_moving_fixed": normalized_cross_correlation_similarity(moving, fixed, device=device, metric_name="LCC", pair="moving-fixed", case_id=case_id, row_index=row_index, frame=frame), "lcc_warped_fixed": normalized_cross_correlation_similarity(warped, fixed, device=device, metric_name="LCC", pair="warped-fixed", case_id=case_id, row_index=row_index, frame=frame),
        "mse_moving_fixed": mse(moving, fixed, device=device, case_id=case_id, pair="moving-fixed"), "mse_warped_fixed": mse(warped, fixed, device=device, case_id=case_id, pair="warped-fixed"),
    }
    LOGGER.info("[GLOBAL] NMI moving-fixed=%s, warped-fixed=%s", out["nmi_moving_fixed"], out["nmi_warped_fixed"])
    LOGGER.info("[GLOBAL] SSIM moving-fixed=%s, warped-fixed=%s", out["ssim_moving_fixed"], out["ssim_warped_fixed"])
    LOGGER.info("[GLOBAL] LCC moving-fixed=%s, warped-fixed=%s", out["lcc_moving_fixed"], out["lcc_warped_fixed"])
    return out
