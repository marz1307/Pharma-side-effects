"""Numeric and categorical preprocessing for review and manufacturer fields."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def build_numeric_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build a numeric and categorical preprocessor.

    Numeric review percentages are standardized; categorical columns
    (e.g., ``Manufacturer``) are ordinal-encoded so they remain
    compatible with tree ensembles.

    Args:
        numeric_columns: Numeric columns to scale.
        categorical_columns: Categorical columns to ordinal-encode.

    Returns:
        Configured :class:`ColumnTransformer`.
    """
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline([("scaler", StandardScaler(with_mean=False))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")
