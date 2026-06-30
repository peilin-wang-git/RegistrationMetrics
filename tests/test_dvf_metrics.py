import numpy as np
from registration_metrics.dvf_metrics import jacobian_determinant

def test_identity_dvf_jacobian():
    dvf=np.zeros((4,5,6,3), dtype=float)
    det=jacobian_determinant(dvf, (1.0,1.0,1.0))
    assert np.allclose(det, 1.0)
    assert np.mean(det <= 0) == 0.0

import nibabel as nib
from registration_metrics.dvf_metrics import compute_dvf_metrics


def test_dvf_total_voxels_identity(tmp_path):
    dvf = np.zeros((4, 5, 6, 3), dtype=np.float32)
    path = tmp_path / "dvf.nii.gz"
    nib.save(nib.Nifti1Image(dvf, np.eye(4)), path)
    out = compute_dvf_metrics(path, "case")
    assert out["total_voxels"] == 4 * 5 * 6
    assert out["num_folding_voxels"] == 0
    assert out["folding_ratio"] == 0.0


def test_dvf_total_voxels_uses_finite_detj(monkeypatch, tmp_path):
    import registration_metrics.dvf_metrics as dm
    det = np.ones((2, 2, 2), dtype=float)
    det[0, 0, 0] = np.nan
    det[0, 0, 1] = -1.0
    monkeypatch.setattr(dm, "jacobian_determinant", lambda dvf, spacing, device="cpu": det)
    path = tmp_path / "dvf.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 3), dtype=np.float32), np.eye(4)), path)
    out = dm.compute_dvf_metrics(path, "case")
    assert out["total_voxels"] == 7
    assert out["num_folding_voxels"] == 1
    assert out["folding_ratio"] == 1 / 7
