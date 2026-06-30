import numpy as np
from registration_metrics.motion_metrics import motion_summary, pearson_safe

def test_motion_metric_formula():
    pred=np.array([1.0,2.0,3.0]); gt=np.array([1.0,1.0,5.0])
    s=motion_summary(pred, gt)
    assert np.isclose(s["amd"], 1.0)
    assert np.isclose(s["rmse"], np.sqrt(5/3))
    assert np.isclose(s["mape"], np.mean([0,1/1,2/5])*100)
    assert np.isclose(s["pcc"], pearson_safe(pred, gt))


def _cube(seg, label, start, stop):
    seg[start:stop, start:stop, start:stop] = label


def test_motion_bbox_fallback_target_mask_missing(monkeypatch):
    import registration_metrics.motion_metrics as mm
    captured = {}
    def fake_match(ref_img, ref_bbox, tgt_img, tgt_bbox, *args, **kwargs):
        captured.setdefault("ref_bbox", ref_bbox); captured.setdefault("target_bbox", tgt_bbox)
        return tgt_bbox
    monkeypatch.setattr(mm, "match_ncc", fake_match)
    moving_seg = np.zeros((8, 8, 8), dtype=int); warped_seg = np.zeros_like(moving_seg); fixed_seg = np.zeros_like(moving_seg)
    _cube(moving_seg, 27, 1, 5); _cube(fixed_seg, 27, 1, 5)
    out = mm.compute_organ_ncc_moves(np.ones((8,8,8)), np.ones((8,8,8)), np.ones((8,8,8)), fixed_seg, moving_seg, warped_seg, {27:"liver"}, np.eye(4), "case", 0, organs=["liver"])
    assert out["liverNCCMoveFallbackUsed"] is True
    assert "warped mask invalid" in out["liverNCCMoveFallbackReason"]
    assert captured["target_bbox"] == captured["ref_bbox"]


def test_motion_bbox_fallback_reference_mask_missing(monkeypatch):
    import registration_metrics.motion_metrics as mm
    captured = {}
    def fake_match(ref_img, ref_bbox, tgt_img, tgt_bbox, *args, **kwargs):
        captured.setdefault("ref_bbox", ref_bbox); captured.setdefault("target_bbox", tgt_bbox)
        return tgt_bbox
    monkeypatch.setattr(mm, "match_ncc", fake_match)
    moving_seg = np.zeros((8, 8, 8), dtype=int); warped_seg = np.zeros_like(moving_seg); fixed_seg = np.zeros_like(moving_seg)
    _cube(warped_seg, 27, 2, 6); _cube(fixed_seg, 27, 2, 6)
    out = mm.compute_organ_ncc_moves(np.ones((8,8,8)), np.ones((8,8,8)), np.ones((8,8,8)), fixed_seg, moving_seg, warped_seg, {27:"liver"}, np.eye(4), "case", 0, organs=["liver"])
    assert out["liverNCCMoveFallbackUsed"] is True
    assert "moving mask invalid" in out["liverNCCMoveFallbackReason"]
    assert captured["ref_bbox"] == captured["target_bbox"]


def test_motion_skip_both_masks_missing():
    import registration_metrics.motion_metrics as mm
    seg = np.zeros((8, 8, 8), dtype=int)
    out = mm.compute_organ_ncc_moves(np.ones((8,8,8)), np.ones((8,8,8)), np.ones((8,8,8)), seg, seg, seg, {27:"liver"}, np.eye(4), "case", 0, organs=["liver"])
    assert out["liverNCCMoveFallbackUsed"] is False
    assert np.isnan(out["liverNCCMove_AP"])
    assert "both masks" in out["liverNCCMoveFallbackReason"]


def test_motion_volume_ratio_mismatch(monkeypatch):
    import registration_metrics.motion_metrics as mm
    monkeypatch.setattr(mm, "match_ncc", lambda ref_img, ref_bbox, tgt_img, tgt_bbox, *args, **kwargs: tgt_bbox)
    moving_seg = np.zeros((10, 10, 10), dtype=int); warped_seg = np.zeros_like(moving_seg); fixed_seg = np.zeros_like(moving_seg)
    _cube(moving_seg, 27, 1, 4)  # 27 voxels
    _cube(warped_seg, 27, 1, 9)  # 512 voxels, ratio < 0.20
    _cube(fixed_seg, 27, 1, 9)
    out = mm.compute_organ_ncc_moves(np.ones((10,10,10)), np.ones((10,10,10)), np.ones((10,10,10)), fixed_seg, moving_seg, warped_seg, {27:"liver"}, np.eye(4), "case", 0, organs=["liver"])
    assert out["liverNCCMoveFallbackUsed"] is True
    assert out["liverNCCMoveMaskVolumeRatio"] < 0.20
