from __future__ import annotations

import pytest
import torch
from botorch.models.pairwise_gp import PairwiseGP

from pairwise_bo.autoencoder import Autoencoder
from pairwise_bo.core import AutoencoderPreferenceElicitator, PreferenceElicitator
from pairwise_bo.data import CsvDatasetLoader


def test_persistence_roundtrip(tmp_path, synthetic_csv, tiny_feature_mapping):
    loader = CsvDatasetLoader(
        data_path=synthetic_csv,
        feature_mapping=tiny_feature_mapping,
        scaler_type="robust",
    )
    weights = torch.tensor([-0.4, 0.2, 0.3, -0.1], dtype=torch.float32)
    elicitator = PreferenceElicitator(data_loader=loader, user_weights=weights)

    # Do one choice to populate train_comps
    candidate_pair = elicitator.select_next_candidate_pair()
    response = elicitator.predict_choice(candidate_pair.to_tensor())
    elicitator.handle_user_response(candidate_pair, response)

    original_ranking = elicitator.rank_listings(loader.data[:10])

    # Save model
    model_path = tmp_path / "model.pt"
    elicitator.save_model(model_path)

    # Load into a new elicitator
    new_elicitator = PreferenceElicitator(
        data_loader=loader,
        user_weights=weights,
        saved_model_path=model_path,
    )

    new_ranking = new_elicitator.rank_listings(loader.data[:10])

    assert torch.equal(original_ranking, new_ranking)
    assert new_elicitator.total_compare_count == elicitator.total_compare_count
    assert isinstance(new_elicitator.model, PairwiseGP)


def test_ae_integration_loop(synthetic_csv, tiny_feature_mapping):
    loader = CsvDatasetLoader(
        data_path=synthetic_csv,
        feature_mapping=tiny_feature_mapping,
        scaler_type="robust",
    )
    weights = torch.tensor([-0.4, 0.2, 0.3, -0.1], dtype=torch.float32)

    autoencoder = Autoencoder(
        input_dim=loader.dim,
        latent_dim=2,
        hidden_dim_1=4,
        hidden_dim_2=3,
    )

    elicitator = AutoencoderPreferenceElicitator(
        autoencoder_model=autoencoder,
        latent_dim=2,
        data_loader=loader,
        user_weights=weights,
    )

    candidate_pair = elicitator.select_next_candidate_pair()
    assert len(candidate_pair.listing_a.values) == loader.dim
    assert len(candidate_pair.listing_b.values) == loader.dim

    response = elicitator.predict_choice(candidate_pair.to_tensor())
    elicitator.handle_user_response(candidate_pair, response)

    ranked_idx = elicitator.rank_listings(loader.data[:10])
    assert ranked_idx.shape[0] == 10
    assert elicitator.total_compare_count >= 2
