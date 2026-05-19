from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("SMOKE_TEST", "1")

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def tiny_feature_mapping() -> dict[str, dict[str, int | str]]:
    return {
        "feature_a": {"index": 0, "display_name": "Feature A", "unit": "u"},
        "feature_b": {"index": 1, "display_name": "Feature B", "unit": "u"},
        "feature_c": {"index": 2, "display_name": "Feature C", "unit": "u"},
        "feature_d": {"index": 3, "display_name": "Feature D", "unit": "u"},
    }


@pytest.fixture
def synthetic_csv(tmp_path: Path, tiny_feature_mapping: dict[str, dict[str, int | str]]) -> Path:
    rng = np.random.default_rng(7)
    row_count = 64

    ordered_cols = [
        key for key, _ in sorted(tiny_feature_mapping.items(), key=lambda item: item[1]["index"])
    ]

    df = pd.DataFrame(
        {
            ordered_cols[0]: rng.normal(loc=1400.0, scale=250.0, size=row_count),
            ordered_cols[1]: rng.normal(loc=4.0, scale=2.0, size=row_count),
            ordered_cols[2]: rng.normal(loc=80.0, scale=20.0, size=row_count),
            ordered_cols[3]: rng.normal(loc=15.0, scale=6.0, size=row_count),
        }
    )

    df.loc[0, ordered_cols[2]] = np.inf
    df.loc[1, ordered_cols[3]] = np.nan

    csv_path = tmp_path / "synthetic_features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
