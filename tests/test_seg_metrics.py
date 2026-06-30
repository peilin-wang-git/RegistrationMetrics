import numpy as np
from registration_metrics.seg_metrics import dice_coefficient, iou_score, hd95, assd, compute_segmentation_metrics
from registration_metrics.config import SEG_METRIC_ORGANS, SEG_MEAN_ORGANS, resolve_seg_metric_organs, resolve_seg_mean_organs


def test_dice_iou_simple_mask():
    a=np.array([1,1,0,0], dtype=bool); b=np.array([1,0,1,0], dtype=bool)
    assert dice_coefficient(a,b) == 0.5
    assert iou_score(a,b) == 1/3


def test_empty_mask_behavior():
    a=np.zeros((3,3,3), dtype=bool); b=np.zeros_like(a); c=a.copy(); c[1,1,1]=1
    assert dice_coefficient(a,b) == 1.0
    assert iou_score(a,b) == 1.0
    assert hd95(a,b,(1,1,1)) == 0.0
    assert assd(a,b,(1,1,1)) == 0.0
    assert dice_coefficient(a,c) == 0.0
    assert np.isnan(hd95(a,c,(1,1,1)))


def test_default_seg_metric_and_mean_organs():
    assert SEG_METRIC_ORGANS == ["heart", "liver", "spleen", "pancreas", "kidney_left", "kidney_right", "stomach", "aorta", "inferior_vena_cava"]
    for organ in ["gallbladder", "duodenum", "small_bowel", "colon", "urinary_bladder"]:
        assert organ not in SEG_METRIC_ORGANS
    for organ in ["heart", "gallbladder", "duodenum", "small_bowel", "colon"]:
        assert organ in SEG_MEAN_ORGANS
    for organ in ["urinary_bladder", "prostate", "femur_left", "hip_left", "sacrum", "vertebrae"]:
        assert organ not in SEG_MEAN_ORGANS


def test_selected_seg_organs_columns_and_heart_to_kidney_mean():
    fixed = np.zeros((3, 3, 3), dtype=int)
    moving = np.zeros_like(fixed)
    warped = np.zeros_like(fixed)
    fixed[0, 0, 0] = 17  # selected heart and mean organ
    moving[0, 0, 0] = 17
    warped[0, 0, 0] = 17
    fixed[1, 1, 1] = 10  # mean-only gallbladder
    warped[1, 1, 1] = 10
    fixed[2, 2, 2] = 39  # urinary bladder: outside mean set
    warped[2, 2, 2] = 39
    label_map = {0: "background", 10: "gallbladder", 17: "heart", 39: "urinary_bladder"}
    out = compute_segmentation_metrics(fixed, moving, warped, label_map, (1, 1, 1), "case", 0)
    assert "dice_heart_moving_fixed" in out
    assert "iou_heart_warped_fixed" in out
    assert "hd95_heart_warped_fixed" in out
    assert "assd_heart_warped_fixed" in out
    assert "dice_gallbladder_moving_fixed" not in out
    assert "dice_urinary_bladder_warped_fixed" not in out
    assert np.isclose(out["mean_dice_all_organs_moving_fixed"], 0.5)
    assert np.isclose(out["mean_dice_all_organs_warped_fixed"], 1.0)


def test_resolve_seg_metric_and_mean_organs_independent_overrides():
    assert resolve_seg_metric_organs({}, "liver,spleen") == ["liver", "spleen"]
    assert resolve_seg_metric_organs({"seg_metric_organs": ["pancreas"]}, None) == ["pancreas"]
    assert resolve_seg_mean_organs({}, "heart,liver") == ["heart", "liver"]
    assert resolve_seg_mean_organs({"seg_mean_organs": ["duodenum"]}, None) == ["duodenum"]
