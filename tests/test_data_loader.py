"""Tests for the dataset loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pharma_side_effects.data.loader import load_dataset


def test_load_dataset_csv(synthetic_csv: Path) -> None:
    df = load_dataset(synthetic_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in ("Medicine Name", "Composition", "Uses", "Side_effects", "Manufacturer"):
        assert col in df.columns


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")


def test_load_dataset_unsupported_suffix(tmp_path: Path) -> None:
    p = tmp_path / "bad.parquet"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_dataset(p)


def test_load_dataset_missing_columns(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(p)
