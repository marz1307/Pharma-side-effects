"""Logistic Regression classifier factory."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from pharma_side_effects.config import LogisticRegressionConfig


def build_logistic_regression(cfg: LogisticRegressionConfig) -> LogisticRegression:
    """Construct a multinomial :class:`LogisticRegression` from config.

    Multinomial behaviour is auto-selected by sklearn when the target has more
    than two classes and the lbfgs solver is used. The ``multi_class`` kwarg
    is intentionally omitted as it is deprecated in scikit-learn 1.5+.

    Args:
        cfg: Logistic regression hyperparameters.

    Returns:
        Unfitted :class:`LogisticRegression`.
    """
    return LogisticRegression(
        max_iter=cfg.max_iter,
        C=cfg.C,
        solver=cfg.solver,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
    )
