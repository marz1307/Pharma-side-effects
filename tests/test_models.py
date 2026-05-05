"""Tests for the model registry and factories."""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pharma_side_effects.models.registry import MODEL_REGISTRY, build_model


def test_registry_keys() -> None:
    assert set(MODEL_REGISTRY) == {"random_forest", "logistic_regression"}


def test_build_random_forest(config) -> None:
    model = build_model("random_forest", config.models)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == config.models.random_forest.n_estimators


def test_build_logistic_regression(config) -> None:
    model = build_model("logistic_regression", config.models)
    assert isinstance(model, LogisticRegression)
    assert model.max_iter == config.models.logistic_regression.max_iter


def test_unknown_model_raises(config) -> None:
    with pytest.raises(KeyError):
        build_model("xgboost", config.models)
