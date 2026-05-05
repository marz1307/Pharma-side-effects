"""Inference helpers for a persisted pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from pharma_side_effects.utils.logging import get_logger

logger = get_logger(__name__)


def load_pipeline(model_path: str | Path) -> Pipeline:
    """Load a fitted pipeline from disk.

    Args:
        model_path: Path to a ``.joblib`` artifact.

    Returns:
        Loaded sklearn :class:`Pipeline`.

    Raises:
        FileNotFoundError: When the artifact is missing.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model artifact not found: {p}")
    return joblib.load(p)


def predict(model_path: str | Path, data: pd.DataFrame) -> pd.Series:
    """Run inference for a dataframe of medicines.

    Args:
        model_path: Path to a serialized pipeline.
        data: Input dataframe with the same predictor columns used at fit.

    Returns:
        Series of predicted side-effect categories.
    """
    pipeline = load_pipeline(model_path)
    preds = pipeline.predict(data)
    return pd.Series(preds, index=data.index, name="prediction")


def predict_from_csv(
    model_path: str | Path,
    csv_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Predict for an input CSV, optionally writing the augmented file.

    Args:
        model_path: Path to a serialized pipeline.
        csv_path: Input CSV path.
        output_path: Optional path to write the predictions back to disk.

    Returns:
        Dataframe with an added ``prediction`` column.
    """
    df = pd.read_csv(csv_path)
    df["prediction"] = predict(model_path, df).values
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Wrote predictions to %s", output_path)
    return df
