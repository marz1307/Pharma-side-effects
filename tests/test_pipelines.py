"""End-to-end pipeline tests on the synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from pharma_side_effects.pipelines.preprocess import (
    apply_category_mapping,
    group_rare_classes,
    prepare_features_target,
)
from pharma_side_effects.pipelines.train import build_pipeline


def test_pipeline_train_and_predict(synthetic_df: pd.DataFrame, config, tmp_path: Path) -> None:
    df = apply_category_mapping(synthetic_df, config.data)
    df = group_rare_classes(df, config.data)
    features, target = prepare_features_target(df, config.data)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42, stratify=target
    )

    pipeline = build_pipeline(config, model_name="logistic_regression")
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)
    assert len(preds) == len(x_test)
    assert set(preds).issubset(set(target.unique()))

    artifact = tmp_path / "model.joblib"
    joblib.dump(pipeline, artifact)
    reloaded = joblib.load(artifact)
    assert (reloaded.predict(x_test) == preds).all()


def test_apply_category_mapping(synthetic_df: pd.DataFrame, config) -> None:
    out = apply_category_mapping(synthetic_df, config.data)
    assert config.data.target_column in out.columns
    assert out[config.data.target_column].notna().all()


def test_group_rare_classes(synthetic_df: pd.DataFrame, config) -> None:
    df = apply_category_mapping(synthetic_df, config.data)
    grouped = group_rare_classes(df, config.data)
    grouped_col = grouped[config.data.grouped_target_column]
    for rare in config.data.rare_classes_to_group:
        assert rare not in grouped_col.values
