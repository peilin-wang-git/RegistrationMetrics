import numpy as np
from registration_metrics.seg_metrics import dice_coefficient, iou_score, hd95, assd

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


def test_selected_seg_organs_columns_and_all_organ_mean():
    from registration_metrics.seg_metrics import compute_segmentation_metrics
    fixed = np.zeros((3, 3, 3), dtype=int)
    moving = np.zeros_like(fixed)
    warped = np.zeros_like(fixed)
    fixed[0, 0, 0] = 27  # selected liver
    moving[0, 0, 0] = 27
    warped[0, 0, 0] = 27
    fixed[1, 1, 1] = 3  # non-selected aorta still contributes to all-organ mean
    warped[1, 1, 1] = 3
    label_map = {0: "background", 3: "aorta", 27: "liver"}
    out = compute_segmentation_metrics(fixed, moving, warped, label_map, (1, 1, 1), "case", 0, seg_metric_organs=["liver"])
    assert "dice_liver_moving_fixed" in out
    assert "dice_aorta_moving_fixed" not in out
    assert np.isclose(out["mean_dice_all_organs_moving_fixed"], 0.5)
    assert np.isclose(out["mean_dice_all_organs_warped_fixed"], 1.0)


def test_resolve_seg_metric_organs_cli_and_config_override():
    from registration_metrics.config import resolve_seg_metric_organs
    assert resolve_seg_metric_organs({}, "liver,spleen") == ["liver", "spleen"]
    assert resolve_seg_metric_organs({"seg_metric_organs": ["pancreas"]}, None) == ["pancreas"]
