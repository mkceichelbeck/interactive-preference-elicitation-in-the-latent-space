"""Factory and artifact-loading helpers for BO workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from pairwise_bo.autoencoder import Autoencoder
from pairwise_bo.core import AutoencoderPreferenceElicitator, PreferenceElicitator
from pairwise_bo.data import CsvDatasetLoader
from pairwise_bo.types import FeatureMapping


def build_loader(
    data_path: Path,
    *,
    dataset_preset: Optional[str] = "idealista",
    feature_mapping: Optional[FeatureMapping] = None,
    rename_map: Optional[dict[str, str]] = None,
    scaler_type: str = "robust",
    drop_columns: tuple[str, ...] = ("listing_id",),
    handle_outliers: bool = False,
) -> CsvDatasetLoader:
    """Create a dataset loader from either a preset or an explicit schema."""
    if feature_mapping is not None:
        return CsvDatasetLoader(
            data_path=data_path,
            feature_mapping=feature_mapping,
            rename_map=rename_map,
            scaler_type=scaler_type,
            drop_columns=drop_columns,
            handle_outliers=handle_outliers,
        )

    if dataset_preset is None:
        raise ValueError(
            "Either feature_mapping or dataset_preset must be provided to build a loader."
        )

    return CsvDatasetLoader.from_preset(
        data_path=data_path,
        preset=dataset_preset,
        scaler_type=scaler_type,
    )


def load_autoencoder_checkpoint(
    checkpoint_path: Path,
    *,
    input_dim: int,
) -> tuple[Autoencoder, int]:
    """Load a vanilla autoencoder checkpoint and rebuild its architecture."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint
    config: dict[str, int | float] = {}
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        raw_config = checkpoint.get("config", {})
        if isinstance(raw_config, dict):
            config = raw_config

    resolved_input_dim = int(config.get("input_dim", input_dim))
    hidden_dim_1 = int(config.get("hidden_dim_1", 64))
    hidden_dim_2 = int(config.get("hidden_dim_2", 32))
    latent_dim = int(config.get("latent_dim", 8))
    dropout_rate = float(config.get("dropout_rate", 0.1))

    autoencoder = Autoencoder(
        input_dim=resolved_input_dim,
        latent_dim=latent_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout_rate=dropout_rate,
    )
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    return autoencoder, latent_dim


def get_elicitator(
    data_path: Path,
    *,
    bo_model_path: Optional[Path] = None,
    autoencoder_model_path: Optional[Path] = None,
    user_feature_weights: Optional[torch.Tensor] = None,
    user_bounds: Optional[torch.Tensor] = None,
    dataset_preset: Optional[str] = "idealista",
    feature_mapping: Optional[FeatureMapping] = None,
    rename_map: Optional[dict[str, str]] = None,
    scaler_type: str = "robust",
    drop_columns: tuple[str, ...] = ("listing_id",),
    handle_outliers: bool = False,
) -> PreferenceElicitator:
    """Create a plain or AE-backed elicitator."""
    resolved_bo_model_path = bo_model_path
    if resolved_bo_model_path is not None and not resolved_bo_model_path.exists():
        resolved_bo_model_path = None

    loader = build_loader(
        data_path=data_path,
        dataset_preset=dataset_preset,
        feature_mapping=feature_mapping,
        rename_map=rename_map,
        scaler_type=scaler_type,
        drop_columns=drop_columns,
        handle_outliers=handle_outliers,
    )

    if autoencoder_model_path is not None and autoencoder_model_path.exists():
        autoencoder_model, latent_dim = load_autoencoder_checkpoint(
            autoencoder_model_path,
            input_dim=loader.dim,
        )
        return AutoencoderPreferenceElicitator(
            autoencoder_model=autoencoder_model,
            latent_dim=latent_dim,
            data_loader=loader,
            user_weights=user_feature_weights,
            bounds=user_bounds,
            saved_model_path=resolved_bo_model_path,
        )

    return PreferenceElicitator(
        data_loader=loader,
        user_weights=user_feature_weights,
        bounds=user_bounds,
        saved_model_path=resolved_bo_model_path,
    )
