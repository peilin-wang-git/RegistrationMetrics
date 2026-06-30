import logging
import numpy as np
import pandas as pd
import nibabel as nib
from registration_metrics.io_utils import compute_from_config


def _save(path, data):
    nib.save(nib.Nifti1Image(np.asarray(data), np.eye(4)), path)
    return str(path)


def _config(tmp_path, rows):
    csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return {"M": {"G": {"csv_path": str(csv), "center": "C", "modality": "CT", "task": "T", "organ": "liver"}}}


def _row(tmp_path, name, shape=(4, 4, 4), frame=None):
    img = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    seg = np.zeros(shape, dtype=np.int16)
    if len(shape) == 3:
        seg[1:3, 1:3, 1:3] = 27
    else:
        seg[1:3, 1:3, 1:3, :] = 27
    row = {
        "case_id": name,
        "fixed_img_path": _save(tmp_path / f"{name}_fixed.nii.gz", img),
        "moving_img_path": _save(tmp_path / f"{name}_moving.nii.gz", img),
        "warped_img_path": _save(tmp_path / f"{name}_warped.nii.gz", img),
        "fixed_seg_path": _save(tmp_path / f"{name}_fixed_seg.nii.gz", seg),
        "moving_seg_path": _save(tmp_path / f"{name}_moving_seg.nii.gz", seg),
        "warped_seg_path": _save(tmp_path / f"{name}_warped_seg.nii.gz", seg),
    }
    if frame is not None:
        row["Frame"] = frame
    return row


def test_3d_only_rejects_4d_input(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="registration_metrics")
    cfg = _config(tmp_path, [_row(tmp_path, "case4d", shape=(4, 4, 4, 2))])
    out, _, _ = compute_from_config(cfg, tmp_path / "out", enable_global=True, enable_seg=False, enable_dvf=False, enable_motion=False, enable_vertebra=False)
    assert len(out) == 1
    assert out.loc[0, "status"] == "skipped"
    assert "current pipeline expects pre-split 3D cases" in out.loc[0, "skip_reason"]
    assert "[SKIP NON-3D]" in caplog.text
    assert "total_frames" not in caplog.text


def test_single_3d_case_no_frame_loop_and_preserves_frame(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="registration_metrics")
    cfg = _config(tmp_path, [_row(tmp_path, "case3d", frame=7)])
    out, _, _ = compute_from_config(cfg, tmp_path / "out", enable_global=True, enable_seg=False, enable_dvf=False, enable_motion=False, enable_vertebra=False)
    assert len(out) == 1
    assert int(out.loc[0, "Frame"]) == 7
    assert out.loc[0, "status"] == "ok"
    assert "[LOAD 3D] fixed shape=" in caplog.text
    assert "total_frames" not in caplog.text


def test_multiprocessing_outputs_same_as_single_process(tmp_path):
    rows = [_row(tmp_path, "case1"), _row(tmp_path, "case2")]
    cfg = _config(tmp_path, rows)
    one, _, _ = compute_from_config(cfg, tmp_path / "out1", enable_global=True, enable_seg=False, enable_dvf=False, enable_motion=False, enable_vertebra=False, num_workers=1)
    two, _, _ = compute_from_config(cfg, tmp_path / "out2", enable_global=True, enable_seg=False, enable_dvf=False, enable_motion=False, enable_vertebra=False, num_workers=2)
    one = one.sort_values("case_id").reset_index(drop=True)
    two = two.sort_values("case_id").reset_index(drop=True)
    for col in ["nmi_moving_fixed", "nmi_warped_fixed", "mse_moving_fixed", "mse_warped_fixed"]:
        assert np.allclose(one[col], two[col], equal_nan=True)


def test_multiprocessing_progress_written_by_main_process(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="registration_metrics")
    cfg = _config(tmp_path, [_row(tmp_path, "case1"), _row(tmp_path, "case2")])
    compute_from_config(cfg, tmp_path / "out", enable_global=True, enable_seg=False, enable_dvf=False, enable_motion=False, enable_vertebra=False, num_workers=2)
    progress = tmp_path / "out" / "detailed_progress.csv"
    assert progress.exists()
    assert len(pd.read_csv(progress)) == 2
    assert "[MP DONE]" in caplog.text
    assert "[SAVE PROGRESS] appended_rows=" in caplog.text


def test_intensity_normalization_range():
    from registration_metrics.io_utils import normalize_intensity_0_1
    img = np.array([-100.0, 100.0, 300.0], dtype=np.float32).reshape(1, 1, 3)
    out = normalize_intensity_0_1(img, "fixed", "case", 0)
    assert out.dtype == np.float32
    assert np.nanmin(out) == 0.0
    assert np.nanmax(out) == 1.0


def test_intensity_normalization_constant_image():
    from registration_metrics.io_utils import normalize_intensity_0_1
    img = np.full((2, 2, 2), 5.0, dtype=np.float32)
    out = normalize_intensity_0_1(img, "fixed", "case", 0)
    assert out.dtype == np.float32
    assert np.all(out == 0.0)


def test_intensity_normalization_nan_inf():
    from registration_metrics.io_utils import normalize_intensity_0_1
    img = np.array([np.nan, -100.0, 100.0, np.inf], dtype=np.float32).reshape(1, 2, 2)
    out = normalize_intensity_0_1(img, "fixed", "case", 0)
    finite = np.isfinite(out)
    assert np.nanmin(out) == 0.0
    assert np.nanmax(out) == 1.0
    assert finite.sum() == 2
    assert np.isnan(out[0, 0, 0])
    assert np.isnan(out[0, 1, 1])


def test_seg_not_normalized(monkeypatch, tmp_path):
    import registration_metrics.io_utils as io
    row = _row(tmp_path, "case_seg")
    cfg = _config(tmp_path, [row])
    seen = {}
    def fake_seg(fixed_seg, moving_seg, warped_seg, *args, **kwargs):
        seen["fixed_max"] = int(np.max(fixed_seg))
        seen["moving_max"] = int(np.max(moving_seg))
        seen["warped_max"] = int(np.max(warped_seg))
        return {"seg_seen": 1.0}
    monkeypatch.setattr(io, "compute_segmentation_metrics", fake_seg)
    compute_from_config(cfg, tmp_path / "out_seg", enable_global=False, enable_seg=True, enable_dvf=False, enable_motion=False, enable_vertebra=False)
    assert seen == {"fixed_max": 27, "moving_max": 27, "warped_max": 27}


def test_dvf_not_normalized(monkeypatch, tmp_path):
    import registration_metrics.io_utils as io
    dvf = np.zeros((2, 2, 2, 3), dtype=np.float32)
    dvf[..., 0] = 25.0
    dvf_path = tmp_path / "case_dvf_transform.nii.gz"
    nib.save(nib.Nifti1Image(dvf, np.eye(4)), dvf_path)
    row = _row(tmp_path, "case_dvf")
    row["transform_path"] = str(dvf_path)
    cfg = _config(tmp_path, [row])
    seen = {}
    def fake_dvf(transform_path, *args, **kwargs):
        seen["max"] = float(np.asanyarray(nib.load(transform_path).dataobj).max())
        return {"total_voxels": 8, "num_folding_voxels": 0, "folding_ratio": 0.0}
    monkeypatch.setattr(io, "compute_dvf_metrics", fake_dvf)
    compute_from_config(cfg, tmp_path / "out_dvf", enable_global=False, enable_seg=False, enable_dvf=True, enable_motion=False, enable_vertebra=False)
    assert seen["max"] == 25.0


def test_metrics_receive_normalized_images(monkeypatch, tmp_path):
    import registration_metrics.io_utils as io
    row = _row(tmp_path, "case_norm")
    cfg = _config(tmp_path, [row])
    seen = {"global": False, "motion": False}
    def assert_norm(arr):
        assert arr.dtype == np.float32
        assert np.nanmin(arr) >= 0.0
        assert np.nanmax(arr) <= 1.0
    def fake_global(fixed, moving, warped, *args, **kwargs):
        assert_norm(fixed); assert_norm(moving); assert_norm(warped)
        seen["global"] = True
        return {"mse_moving_fixed": 0.0, "mse_warped_fixed": 0.0}
    def fake_motion(fixed, moving, warped, *args, **kwargs):
        assert_norm(fixed); assert_norm(moving); assert_norm(warped)
        seen["motion"] = True
        return {"motion_seen": 1.0}
    monkeypatch.setattr(io, "compute_global_metrics", fake_global)
    monkeypatch.setattr(io, "compute_organ_ncc_moves", fake_motion)
    compute_from_config(cfg, tmp_path / "out_norm", enable_global=True, enable_seg=False, enable_dvf=False, enable_motion=True, enable_vertebra=False)
    assert seen == {"global": True, "motion": True}
