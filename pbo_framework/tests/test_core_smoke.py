from __future__ import annotations

import torch

from pairwise_bo.core import PreferenceElicitator
from pairwise_bo.data import CsvDatasetLoader
from pairwise_bo.factory import get_elicitator


def test_preference_elicitation_single_iteration(
    synthetic_csv,
    tiny_feature_mapping,
) -> None:
    loader = CsvDatasetLoader(
        data_path=synthetic_csv,
        feature_mapping=tiny_feature_mapping,
        scaler_type="robust",
    )

    weights = torch.tensor([-0.4, 0.2, 0.3, -0.1], dtype=torch.float32)
    elicitator = PreferenceElicitator(data_loader=loader, user_weights=weights)

    candidate_pair = elicitator.select_next_candidate_pair()
    assert len(candidate_pair.listing_a.values) == loader.dim
    assert len(candidate_pair.listing_b.values) == loader.dim

    response = elicitator.predict_choice(candidate_pair.to_tensor())
    elicitator.handle_user_response(candidate_pair, response)

    ranked_idx, ranked_scores = elicitator.rank_listings(
        loader.data[:10],
        return_scores=True,
    )

    assert ranked_idx.shape[0] == 10
    assert ranked_scores.shape[0] == 10
    assert elicitator.total_compare_count >= 2


def test_factory_supports_explicit_schema(
    synthetic_csv,
    tiny_feature_mapping,
) -> None:
    weights = torch.tensor([-0.4, 0.2, 0.3, -0.1], dtype=torch.float32)

    elicitator = get_elicitator(
        data_path=synthetic_csv,
        user_feature_weights=weights,
        dataset_preset=None,
        feature_mapping=tiny_feature_mapping,
        scaler_type="standard",
    )

    assert isinstance(elicitator, PreferenceElicitator)
    assert elicitator.dim == 4
