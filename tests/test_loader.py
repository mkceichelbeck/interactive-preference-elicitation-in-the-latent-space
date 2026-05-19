from __future__ import annotations

import torch

from pairwise_bo.data import CsvDatasetLoader


def test_loader_outputs_valid_tensor_bounds(
    synthetic_csv,
    tiny_feature_mapping,
) -> None:
    loader = CsvDatasetLoader(
        data_path=synthetic_csv,
        feature_mapping=tiny_feature_mapping,
        scaler_type="standard",
        handle_outliers=True,
    )

    assert loader.dim == 4
    assert loader.data.shape[1] == 4
    assert not torch.isnan(loader.data).any()
    assert not torch.isinf(loader.data).any()

    bounds = loader.get_bounds()
    assert bounds.shape == (2, 4)
    assert torch.all(bounds[1] >= bounds[0])
