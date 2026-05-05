"""Random Forest classifier factory."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from pharma_side_effects.config import RandomForestConfig


def build_random_forest(cfg: RandomForestConfig) -> RandomForestClassifier:
    """Construct a :class:`RandomForestClassifier` from config.

    Args:
        cfg: Random Forest hyperparameters.

    Returns:
        Unfitted :class:`RandomForestClassifier`.
    """
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )
