"""
GeneralizedPatchedDataset: shapes, NaN handling and dimensionality reduction.
"""

import numpy as np
import torch

from inspectorch import GeneralizedPatchedDataset


def _cube(seed=0, shape=(10, 6, 12)):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def test_pca_keeps_nan_rows_as_nan():
    """Masked pixels stay NaN through PCA; the other rows are projected."""
    data = _cube()
    data[0, 1, 0] = np.nan  # pixel (y=0, x=0) loses one wavelength
    ds = GeneralizedPatchedDataset(
        data, "y wav x", ["wav"], dim_reduction={"method": "pca", "n_components": 2}
    )

    assert ds.patches.shape == (10 * 12, 2)
    nan_rows = torch.isnan(ds.patches).any(dim=1)
    assert nan_rows.tolist().index(True) == 0 and int(nan_rows.sum()) == 1
    assert torch.isfinite(ds.y_std).all()


def test_pca_on_finite_data_matches_sklearn():
    """Without NaN, the reducer is a plain sklearn PCA."""
    from sklearn.decomposition import PCA

    data = _cube()
    ds = GeneralizedPatchedDataset(
        data, "y wav x", ["wav"], dim_reduction={"method": "pca", "n_components": 3}
    )
    rows = data.transpose(0, 2, 1).reshape(-1, data.shape[1])
    expected = PCA(n_components=3).fit_transform(rows).astype(np.float32)

    np.testing.assert_allclose(ds.patches.numpy(), expected, rtol=0, atol=1e-5)
