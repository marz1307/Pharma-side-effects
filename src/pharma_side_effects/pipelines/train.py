"""Training pipeline: build, fit, evaluate, persist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from pharma_side_effects.config import Config, load_config
from pharma_side_effects.data.loader import load_dataset
from pharma_side_effects.evaluation.metrics import (
    classification_summary,
    compute_confusion_matrix,
)
from pharma_side_effects.evaluation.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
)
from pharma_side_effects.features.numeric import build_numeric_pipeline
from pharma_side_effects.features.text import build_text_feature_union
from pharma_side_effects.models.registry import build_model
from pharma_side_effects.pipelines.preprocess import (
    apply_category_mapping,
    group_rare_classes,
    prepare_features_target,
)
from pharma_side_effects.utils.logging import get_logger

logger = get_logger(__name__)


def build_pipeline(cfg: Config, model_name: str | None = None) -> Pipeline:
    """Compose the end-to-end sklearn pipeline.

    Args:
        cfg: Loaded configuration.
        model_name: Optional override of ``cfg.models.default``.

    Returns:
        Unfitted :class:`Pipeline` of (features, classifier).
    """
    text = build_text_feature_union(list(cfg.data.text_columns), cfg.features.tfidf)
    numeric = build_numeric_pipeline(
        list(cfg.data.numeric_columns), list(cfg.data.categorical_columns)
    )
    features: ColumnTransformer | FeatureUnion = FeatureUnion(
        transformer_list=[("text", text), ("numeric", numeric)]
    )
    estimator = build_model(model_name or cfg.models.default, cfg.models)
    return Pipeline([("features", features), ("classifier", estimator)])


def train(
    cfg: Config | None = None,
    model_name: str | None = None,
    config_path: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Train, evaluate, and persist a classifier.

    Args:
        cfg: Optional pre-loaded configuration.
        model_name: Optional model override.
        config_path: Path used when ``cfg`` is not supplied.
        persist: When True, save the fitted pipeline and figures.

    Returns:
        Dictionary containing the fitted pipeline, metrics, label list,
        and (when persisted) artifact paths.
    """
    cfg = cfg or load_config(config_path)
    model_name = model_name or cfg.models.default

    df = load_dataset(cfg.paths.raw_data)
    df = apply_category_mapping(df, cfg.data)
    if cfg.data.group_rare_classes:
        df = group_rare_classes(df, cfg.data)

    features, target = prepare_features_target(df, cfg.data)

    stratify = target if cfg.split.stratify else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        stratify=stratify,
    )

    pipeline = build_pipeline(cfg, model_name=model_name)
    logger.info("Fitting %s on %d training rows", model_name, len(x_train))
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    labels = sorted(target.unique().tolist())
    metrics = classification_summary(
        y_test.to_numpy(),
        y_pred,
        labels=labels,
        zero_division=cfg.evaluation.zero_division,
    )
    matrix = compute_confusion_matrix(y_test.to_numpy(), y_pred, labels)
    logger.info(
        "Model=%s accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
        model_name,
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
    )

    artifacts: dict[str, Any] = {
        "pipeline": pipeline,
        "metrics": metrics,
        "labels": labels,
        "confusion_matrix": matrix,
        "model_name": model_name,
    }

    if persist:
        cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = cfg.paths.models_dir / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        figures_dir = cfg.paths.reports_dir / "figures"
        plot_class_distribution(
            target,
            title="Side Effect Class Distribution",
            output_path=figures_dir / "class_distribution.png",
        )
        plot_confusion_matrix(
            matrix,
            labels=labels,
            title=f"Confusion Matrix: {model_name}",
            output_path=figures_dir / f"confusion_matrix_{model_name}.png",
            figsize=cfg.evaluation.figure_size,
        )
        artifacts["model_path"] = model_path
        logger.info("Saved model to %s", model_path)

    return artifacts


def _is_dataframe(x: Any) -> bool:
    return isinstance(x, pd.DataFrame)
