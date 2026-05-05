"""Pre-modelling transforms: labelling, grouping, feature/target split."""

from __future__ import annotations

import pandas as pd

from pharma_side_effects.config import DataConfig
from pharma_side_effects.data.categories import map_side_effect


def apply_category_mapping(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Add the standardized side-effect class column to a copy of ``df``.

    Args:
        df: Input dataframe containing the raw side-effects column.
        cfg: Data configuration.

    Returns:
        New dataframe with the target column populated.
    """
    out = df.copy()
    out[cfg.target_column] = out[cfg.side_effects_column].apply(map_side_effect)
    return out


def group_rare_classes(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Collapse rare classes into a single ``Other`` bucket.

    Args:
        df: Dataframe containing the raw target column.
        cfg: Data configuration.

    Returns:
        New dataframe with the grouped target column added.
    """
    out = df.copy()
    out[cfg.grouped_target_column] = out[cfg.target_column].replace(
        {cls: "Other" for cls in cfg.rare_classes_to_group}
    )
    return out


def prepare_features_target(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into a feature matrix and target series.

    Args:
        df: Dataframe with both predictors and target columns present.
        cfg: Data configuration.

    Returns:
        Tuple of (features dataframe, target series).
    """
    target_col = cfg.grouped_target_column if cfg.group_rare_classes else cfg.target_column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from dataframe")
    feature_cols = (
        list(cfg.text_columns) + list(cfg.numeric_columns) + list(cfg.categorical_columns)
    )
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from dataframe: {missing}")
    return df[feature_cols].copy(), df[target_col].copy()
