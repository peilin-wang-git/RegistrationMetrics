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
