"""Global intensity metrics: NCC/LCC, NMI, SSIM, and MSE."""
from __future__ import annotations

import logging
import numpy as np
from skimage.metrics import structural_similarity

LOGGER = logging.getLogger("registration_metrics")


def normalized_cross_correlation_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute positive NCC similarity; larger is better and constants return NaN."""
    x = np.asarray(a, dtype=float).ravel(); y = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y); x = x[m]; y = y[m]
    if x.size == 0 or y.size == 0 or np.std(x) < eps or np.std(y) < eps:
        LOGGER.info("[GLOBAL] NCC constant/empty input; returning nan")
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std()))


def normalized_mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Compute histogram normalized mutual information after removing non-finite pairs."""
    x = np.asarray(a, dtype=float).ravel(); y = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y); x = x[m]; y = y[m]
    if x.size == 0:
        return float("nan")
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist / np.sum(hist); px = pxy.sum(axis=1); py = pxy.sum(axis=0)
    hx = -np.sum(px[px > 0] * np.log(px[px > 0])); hy = -np.sum(py[py > 0] * np.log(py[py > 0])); hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))
    return float((hx + hy) / hxy) if hxy > 0 else float("nan")


def ssim_3d_slice_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Compute 3D SSIM by averaging valid axial slice SSIM values."""
    x = np.asarray(a, dtype=float); y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        return float("nan")
    vals = []
    for i in range(x.shape[2] if x.ndim == 3 else 1):
        xs = x[:, :, i] if x.ndim == 3 else x; ys = y[:, :, i] if y.ndim == 3 else y
        dr = float(np.nanmax([xs.max(), ys.max()]) - np.nanmin([xs.min(), ys.min()])) if xs.size else 0.0
        if dr > 0 and min(xs.shape) >= 7:
            vals.append(structural_similarity(xs, ys, data_range=dr))
    return float(np.mean(vals)) if vals else float("nan")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean squared error over finite pairs."""
    x = np.asarray(a, dtype=float); y = np.asarray(b, dtype=float); m = np.isfinite(x) & np.isfinite(y)
    return float(np.mean((x[m] - y[m]) ** 2)) if np.any(m) else float("nan")


def compute_global_metrics(fixed: np.ndarray, moving: np.ndarray, warped: np.ndarray, case_id: str, frame: int, bins: int = 64) -> dict[str, float]:
    """Compute fixed-vs-moving and fixed-vs-warped global metrics with verbose logs."""
    LOGGER.info("[GLOBAL] case=%s, frame=%s", case_id, frame)
    LOGGER.info("[GLOBAL] fixed shape=%s, moving shape=%s, warped shape=%s", fixed.shape, moving.shape, warped.shape)
    out = {
        "nmi_moving_fixed": normalized_mutual_information(moving, fixed, bins), "nmi_warped_fixed": normalized_mutual_information(warped, fixed, bins),
        "ssim_moving_fixed": ssim_3d_slice_mean(moving, fixed), "ssim_warped_fixed": ssim_3d_slice_mean(warped, fixed),
        "lcc_moving_fixed": normalized_cross_correlation_similarity(moving, fixed), "lcc_warped_fixed": normalized_cross_correlation_similarity(warped, fixed),
        "mse_moving_fixed": mse(moving, fixed), "mse_warped_fixed": mse(warped, fixed),
    }
    LOGGER.info("[GLOBAL] NMI moving-fixed=%s, warped-fixed=%s", out["nmi_moving_fixed"], out["nmi_warped_fixed"])
    LOGGER.info("[GLOBAL] SSIM moving-fixed=%s, warped-fixed=%s", out["ssim_moving_fixed"], out["ssim_warped_fixed"])
    LOGGER.info("[GLOBAL] LCC moving-fixed=%s, warped-fixed=%s", out["lcc_moving_fixed"], out["lcc_warped_fixed"])
    return out
