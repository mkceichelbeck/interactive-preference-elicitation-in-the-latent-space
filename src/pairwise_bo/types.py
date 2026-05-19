"""Core type contracts for BO components."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

import numpy as np
import torch
from pydantic import BaseModel


class FeatureMetadata(TypedDict):
    """Human-readable metadata for one feature dimension."""

    index: int
    display_name: str
    unit: str


FeatureMapping = dict[str, FeatureMetadata]


def ordered_feature_keys(feature_mapping: FeatureMapping) -> list[str]:
    """Return feature keys sorted by declared feature index."""
    return [
        feature_key
        for feature_key, _ in sorted(
            feature_mapping.items(), key=lambda item: item[1]["index"]
        )
    ]


class GenericCandidate(BaseModel):
    """Schema-agnostic candidate bound to an ordered feature key list."""

    feature_keys: list[str]
    values: list[float]
    preference_mean: Optional[float] = None
    preference_std: Optional[float] = None

    @classmethod
    def from_numpy(cls, data: np.ndarray, feature_keys: list[str]) -> "GenericCandidate":
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}.")
        if len(feature_keys) != data.shape[0]:
            raise ValueError(
                "feature_keys length must match candidate dimension. "
                f"Got {len(feature_keys)} vs {data.shape[0]}."
            )

        return cls(
            feature_keys=list(feature_keys),
            values=[float(x) for x in data.tolist()],
        )

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in zip(self.feature_keys, self.values)}


class CandidatePair(BaseModel):
    """Pair of candidates used in one preference query."""

    listing_a: GenericCandidate
    listing_b: GenericCandidate

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            np.vstack((self.listing_a.to_numpy(), self.listing_b.to_numpy())),
            dtype=torch.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_a": self.listing_a.to_dict(),
            "listing_b": self.listing_b.to_dict(),
        }
