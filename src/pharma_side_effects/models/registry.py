"""Central registry mapping model names to factory callables."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import ClassifierMixin

from pharma_side_effects.config import ModelsConfig
from pharma_side_effects.models.logistic_regression import build_logistic_regression
from pharma_side_effects.models.random_forest import build_random_forest

ModelFactory = Callable[[ModelsConfig], ClassifierMixin]


def _rf_factory(cfg: ModelsConfig) -> ClassifierMixin:
    return build_random_forest(cfg.random_forest)


def _lr_factory(cfg: ModelsConfig) -> ClassifierMixin:
    return build_logistic_regression(cfg.logistic_regression)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "random_forest": _rf_factory,
    "logistic_regression": _lr_factory,
}


def build_model(name: str, cfg: ModelsConfig) -> ClassifierMixin:
    """Instantiate a registered model by name.

    Args:
        name: Key in :data:`MODEL_REGISTRY`.
        cfg: Models configuration block.

    Returns:
        Unfitted estimator.

    Raises:
        KeyError: If the model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](cfg)
