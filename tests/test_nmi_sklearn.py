import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from registration_metrics.image_metrics import normalized_mutual_information


def _direct_sklearn_nmi(x, y, bins=64):
    x = np.asarray(x, dtype=float).ravel(); y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y); x = x[m]; y = y[m]
    edges = np.histogram_bin_edges(np.concatenate([x, y]), bins=bins)
    return float(np.clip(normalized_mutual_info_score(np.digitize(x, edges[1:-1]), np.digitize(y, edges[1:-1]), average_method="arithmetic"), 0.0, 1.0))


def test_nmi_sklearn_range():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 5, 5))
    b = rng.normal(size=(5, 5, 5))
    nmi = normalized_mutual_information(a, b, bins=16)
    assert 0.0 <= nmi <= 1.0


def test_nmi_identical_images_close_to_one():
    a = np.arange(125, dtype=float).reshape(5, 5, 5)
    assert np.isclose(normalized_mutual_information(a, a, bins=16), 1.0)


def test_nmi_uses_sklearn_definition():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float).reshape(2, 2, 2)
    b = np.array([0, 1, 1, 3, 4, 4, 6, 7], dtype=float).reshape(2, 2, 2)
    assert np.isclose(normalized_mutual_information(a, b, bins=4), _direct_sklearn_nmi(a, b, bins=4))
