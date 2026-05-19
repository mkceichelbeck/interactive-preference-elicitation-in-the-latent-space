"""Schema-driven CSV data loader for BO workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from pairwise_bo.presets import (
    IDEALISTA_FEATURE_MAPPING,
    IDEALISTA_RENAME_MAP,
    MUNICH_FEATURE_MAPPING,
)
from pairwise_bo.types import FeatureMapping, ordered_feature_keys


ScalerType = MinMaxScaler | StandardScaler | RobustScaler


class CsvDatasetLoader:
    """Load and preprocess tabular listing features into tensors for BO."""

    def __init__(
        self,
        data_path: Path,
        *,
        feature_mapping: FeatureMapping,
        rename_map: Optional[dict[str, str]] = None,
        scaler_type: str = "robust",
        drop_columns: tuple[str, ...] = ("listing_id",),
        handle_outliers: bool = False,
    ):
        self.data_path = Path(data_path)
        self.feature_mapping = feature_mapping
        self.feature_keys = ordered_feature_keys(feature_mapping)
        self.rename_map = rename_map or {}
        self.scaler_type = scaler_type
        self.drop_columns = drop_columns
        self.handle_outliers = handle_outliers

        self.scaler: Optional[ScalerType] = None
        self.data = self._load_data()
        self.dim = int(self.data.shape[1])

    @classmethod
    def from_preset(
        cls,
        data_path: Path,
        *,
        preset: str,
        scaler_type: str = "robust",
    ) -> "CsvDatasetLoader":
        """Create a loader for one of the built-in dataset schemas."""
        lowered = preset.lower()
        if lowered == "idealista-madrid" or lowered == "idealista":
            return cls(
                data_path,
                feature_mapping=IDEALISTA_FEATURE_MAPPING,
                rename_map=IDEALISTA_RENAME_MAP,
                scaler_type=scaler_type,
                handle_outliers=False,
            )
        if lowered == "munich":
            return cls(
                data_path,
                feature_mapping=MUNICH_FEATURE_MAPPING,
                rename_map=None,
                scaler_type=scaler_type,
                handle_outliers=True,
            )
        raise ValueError(f"Unknown preset '{preset}'. Use 'idealista' or 'munich'.")

    def has_scaler(self) -> bool:
        return self.scaler is not None

    def scale(self, data: torch.Tensor) -> torch.Tensor:
        """Scale values with the fitted scaler using dataset preprocessing stats."""
        if self.scaler is None:
            raise ValueError("No scaler available. Use scaler_type other than 'none'.")
        arr = data.detach().cpu().numpy()
        return torch.tensor(self.scaler.transform(arr), dtype=torch.float32)

    def reverse_scaling(self, data: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled values into raw feature units."""
        if self.scaler is None:
            raise ValueError("No scaler available. Use scaler_type other than 'none'.")
        return self.scaler.inverse_transform(data)

    def get_bounds(self) -> torch.Tensor:
        """Return optimization bounds aligned with current feature scaling."""
        if self.scaler_type == "minmax":
            return torch.tensor(
                [[0.0] * self.dim, [1.0] * self.dim],
                dtype=torch.float32,
            )

        min_x, _ = self.data.min(dim=0)
        max_x, _ = self.data.max(dim=0)
        return torch.stack([min_x, max_x], dim=0)

    def _build_scaler(self) -> Optional[ScalerType]:
        if self.scaler_type == "none":
            return None
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        raise ValueError(
            "Unknown scaler_type. Use one of: none, minmax, standard, robust."
        )

    def _load_data(self) -> torch.Tensor:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")

        df = pd.read_csv(self.data_path)
        if self.rename_map:
            df = df.rename(columns=self.rename_map)

        existing_drop_cols = [column for column in self.drop_columns if column in df]
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)

        missing = [column for column in self.feature_keys if column not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset is missing required columns for configured schema: {missing}"
            )

        df = df[self.feature_keys]

        for column in df.select_dtypes(include=["bool"]).columns:
            df[column] = df[column].astype(int)

        df = df.replace([float("inf"), -float("inf")], np.nan)

        if df.isna().to_numpy().any():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(
                df[numeric_cols].median(numeric_only=True)
            )
            # Fill any remaining NAs with 0 just in case
            df = df.fillna(0.0)

        # Ensure all columns are numeric before to_numpy conversion
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        if self.handle_outliers:
            for column in df.select_dtypes(include=[np.number]).columns:
                q_low, q_high = df[column].quantile([0.01, 0.99])
                df[column] = df[column].clip(lower=q_low, upper=q_high)

        scaler = self._build_scaler()
        np_data = df.to_numpy(dtype=np.float32)
        if scaler is None:
            return torch.tensor(np_data, dtype=torch.float32)

        self.scaler = scaler.fit(np_data)
        scaled = self.scaler.transform(np_data).astype(np.float32)
        return torch.tensor(scaled, dtype=torch.float32)
