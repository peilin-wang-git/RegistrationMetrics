import numpy as np
from registration_metrics.dvf_metrics import jacobian_determinant

def test_identity_dvf_jacobian():
    dvf=np.zeros((4,5,6,3), dtype=float)
    det=jacobian_determinant(dvf, (1.0,1.0,1.0))
    assert np.allclose(det, 1.0)
    assert np.mean(det <= 0) == 0.0
