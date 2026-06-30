"""Configuration loading, label maps, and logging helpers."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LABEL_MAP: dict[int, str] = {
    0: "background", 1: "adrenal_gland_left", 2: "adrenal_gland_right", 3: "aorta", 4: "brain",
    5: "colon", 6: "duodenum", 7: "esophagus", 8: "femur_left", 9: "femur_right",
    10: "gallbladder", 11: "gluteus_maximus_left", 12: "gluteus_maximus_right", 13: "gluteus_medius_left",
    14: "gluteus_medius_right", 15: "gluteus_minimus_left", 16: "gluteus_minimus_right", 17: "heart",
    18: "hip_left", 19: "hip_right", 20: "iliac_artery_left", 21: "iliac_artery_right", 22: "iliac_vena_left",
    23: "iliac_vena_right", 24: "inferior_vena_cava", 25: "kidney_left", 26: "kidney_right", 27: "liver",
    28: "lung_lower_lobe_left", 29: "lung_lower_lobe_right", 30: "lung_middle_lobe_right", 31: "lung_upper_lobe_left",
    32: "lung_upper_lobe_right", 33: "pancreas", 34: "portal_vein_and_splenic_vein", 35: "prostate",
    36: "small_bowel", 37: "spleen", 38: "stomach", 39: "urinary_bladder", 40: "vertebrae_L1",
    41: "vertebrae_L2", 42: "vertebrae_L3", 43: "vertebrae_L4", 44: "vertebrae_L5", 45: "vertebrae_T10",
    46: "vertebrae_T11", 47: "vertebrae_T12", 48: "vertebrae_T9", 49: "autochthon_left", 50: "autochthon_right",
    51: "autochthon_right", 52: "clavicula_left", 53: "humerus_left", 54: "humerus_right", 55: "iliopsoas_left",
    56: "iliopsoas_right", 57: "intervertebral_discs", 58: "lung_left", 59: "lung_right", 60: "sacrum",
    61: "scapula_left", 62: "scapula_right", 63: "spinal_cord", 64: "vertebrae",
}
MOTION_ORGANS = ["liver", "spleen", "pancreas", "kidney_left", "kidney_right", "kidney"]
SEG_METRIC_ORGANS = ["heart", "liver", "spleen", "pancreas", "kidney_left", "kidney_right", "stomach", "aorta", "inferior_vena_cava"]
SEG_MEAN_ORGANS = ["heart", "liver", "spleen", "pancreas", "kidney_left", "kidney_right", "stomach", "gallbladder", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein", "duodenum", "small_bowel", "colon"]
DIRECTIONS = ["AP", "RL", "SI"]
DEFAULT_MIN_MASK_VOLUME_VOXELS = 20
DEFAULT_SEVERE_VOLUME_RATIO_THRESHOLD = 0.20
REQUIRED_OUTPUT_COLUMNS = ["case_id", "fixed_img_path", "moving_img_path", "warped_img_path", "fixed_seg_path", "moving_seg_path", "warped_seg_path", "transform_path", "Method", "Center", "Modality", "Task", "Organ", "AnalysisGroup", "Frame", "status", "error_message", "skip_reason", "runtime_seconds"]


def setup_logging(output_dir: str | Path) -> logging.Logger:
    """Configure console INFO logging and DEBUG file logging."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("registration_metrics")
    logger.setLevel(logging.DEBUG); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    fh = logging.FileHandler(out / f"metrics_run_{datetime.now():%Y%m%d_%H%M%S}.log"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def load_config(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load YAML dictionary config and validate top-level shape."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("[CONFIG] config must be a dictionary of methods and groups")
    if "CONFIG" in cfg and isinstance(cfg["CONFIG"], dict):
        loaded = dict(cfg["CONFIG"])
        for special_key in ["label_map", "seg_metric_organs", "seg_mean_organs", "min_mask_volume_voxels", "severe_volume_ratio_threshold", "nmi_bins"]:
            if special_key in cfg:
                loaded[special_key] = cfg[special_key]
        return loaded
    return cfg


def merged_label_map(config: dict[str, Any] | None = None) -> dict[int, str]:
    """Return built-in label map plus optional YAML override under label_map."""
    label_map = dict(DEFAULT_LABEL_MAP)
    override = (config or {}).get("label_map", {}) if isinstance(config, dict) else {}
    for key, value in override.items():
        label_map[int(key)] = str(value)
    return label_map


def resolve_seg_metric_organs(config: dict[str, Any] | None = None, cli_organs: str | list[str] | None = None) -> list[str]:
    """Resolve selected segmentation organs from CLI, config, or defaults."""
    if cli_organs:
        if isinstance(cli_organs, str):
            return [x.strip() for x in cli_organs.split(",") if x.strip()]
        return [str(x).strip() for x in cli_organs if str(x).strip()]
    cfg_organs = (config or {}).get("seg_metric_organs") if isinstance(config, dict) else None
    if cfg_organs:
        if isinstance(cfg_organs, str):
            return [x.strip() for x in cfg_organs.split(",") if x.strip()]
        return [str(x).strip() for x in cfg_organs if str(x).strip()]
    return list(SEG_METRIC_ORGANS)


def resolve_seg_mean_organs(config: dict[str, Any] | None = None, cli_organs: str | list[str] | None = None) -> list[str]:
    """Resolve segmentation mean organs from CLI, config, or defaults."""
    if cli_organs:
        if isinstance(cli_organs, str):
            return [x.strip() for x in cli_organs.split(",") if x.strip()]
        return [str(x).strip() for x in cli_organs if str(x).strip()]
    cfg_organs = (config or {}).get("seg_mean_organs") if isinstance(config, dict) else None
    if cfg_organs:
        if isinstance(cfg_organs, str):
            return [x.strip() for x in cfg_organs.split(",") if x.strip()]
        return [str(x).strip() for x in cfg_organs if str(x).strip()]
    return list(SEG_MEAN_ORGANS)


def resolve_mask_quality_thresholds(config: dict[str, Any] | None = None, min_mask_volume_voxels: int | None = None, severe_volume_ratio_threshold: float | None = None) -> tuple[int, float]:
    """Resolve mask quality thresholds from CLI, config, or defaults."""
    cfg = config or {}
    min_volume = min_mask_volume_voxels if min_mask_volume_voxels is not None else cfg.get("min_mask_volume_voxels", DEFAULT_MIN_MASK_VOLUME_VOXELS)
    ratio = severe_volume_ratio_threshold if severe_volume_ratio_threshold is not None else cfg.get("severe_volume_ratio_threshold", DEFAULT_SEVERE_VOLUME_RATIO_THRESHOLD)
    return int(min_volume), float(ratio)
