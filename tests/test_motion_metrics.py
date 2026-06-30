import numpy as np
from registration_metrics.motion_metrics import motion_summary, pearson_safe

def test_motion_metric_formula():
    pred=np.array([1.0,2.0,3.0]); gt=np.array([1.0,1.0,5.0])
    s=motion_summary(pred, gt)
    assert np.isclose(s["amd"], 1.0)
    assert np.isclose(s["rmse"], np.sqrt(5/3))
    assert np.isclose(s["mape"], np.mean([0,1/1,2/5])*100)
    assert np.isclose(s["pcc"], pearson_safe(pred, gt))
