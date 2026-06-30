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
