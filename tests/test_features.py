"""Tests for the feature builders."""

from __future__ import annotations

import pandas as pd
from scipy.sparse import issparse

from pharma_side_effects.config import TfidfConfig
from pharma_side_effects.features.numeric import build_numeric_pipeline
from pharma_side_effects.features.text import build_text_feature_union


def test_text_feature_union_fit_transform(synthetic_df: pd.DataFrame) -> None:
    cfg = TfidfConfig(max_features=50, ngram_range=(1, 2), stop_words="english", lowercase=True)
    transformer = build_text_feature_union(
        ["Medicine Name", "Composition", "Uses", "Side_effects"], cfg
    )
    matrix = transformer.fit_transform(synthetic_df)
    assert matrix.shape[0] == len(synthetic_df)
    assert matrix.shape[1] > 0
    assert issparse(matrix)


def test_numeric_pipeline_fit_transform(synthetic_df: pd.DataFrame) -> None:
    pipeline = build_numeric_pipeline(
        ["Excellent Review %", "Average Review %", "Poor Review %"],
        ["Manufacturer"],
    )
    matrix = pipeline.fit_transform(synthetic_df)
    assert matrix.shape[0] == len(synthetic_df)
    assert matrix.shape[1] == 4
