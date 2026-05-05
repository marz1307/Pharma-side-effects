"""Typed configuration objects loaded from YAML.

The package centralizes every hyperparameter, path, and category setting
in ``configs/default.yaml``. This module loads, validates, and exposes
that configuration as immutable dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class PathsConfig:
    raw_data: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class DataConfig:
    text_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    side_effects_column: str
    target_column: str
    grouped_target_column: str
    rare_classes_to_group: list[str]
    group_rare_classes: bool


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    random_state: int
    stratify: bool


@dataclass(frozen=True)
class TfidfConfig:
    max_features: int
    ngram_range: tuple[int, int]
    stop_words: str | None
    lowercase: bool


@dataclass(frozen=True)
class FeaturesConfig:
    tfidf: TfidfConfig


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    n_jobs: int
    random_state: int


@dataclass(frozen=True)
class LogisticRegressionConfig:
    max_iter: int
    C: float
    solver: str
    class_weight: str | None
    random_state: int


@dataclass(frozen=True)
class ModelsConfig:
    default: str
    random_forest: RandomForestConfig
    logistic_regression: LogisticRegressionConfig


@dataclass(frozen=True)
class EvaluationConfig:
    zero_division: int
    figure_size: tuple[int, int]


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    features: FeaturesConfig
    models: ModelsConfig
    evaluation: EvaluationConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _resolve_path(value: str, base: Path) -> Path:
    """Resolve a config path relative to the project root if not absolute."""
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a YAML configuration file.

    Args:
        path: Optional path to a YAML file. Falls back to the packaged default.

    Returns:
        Parsed :class:`Config` instance.

    Raises:
        FileNotFoundError: If the requested file does not exist.
        ValueError: If required keys are missing.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    project_root = cfg_path.resolve().parent.parent

    try:
        paths = PathsConfig(
            raw_data=_resolve_path(raw["paths"]["raw_data"], project_root),
            processed_dir=_resolve_path(raw["paths"]["processed_dir"], project_root),
            models_dir=_resolve_path(raw["paths"]["models_dir"], project_root),
            reports_dir=_resolve_path(raw["paths"]["reports_dir"], project_root),
        )
        data = DataConfig(**raw["data"])
        split = SplitConfig(**raw["split"])
        tfidf_raw = raw["features"]["tfidf"]
        features = FeaturesConfig(
            tfidf=TfidfConfig(
                max_features=int(tfidf_raw["max_features"]),
                ngram_range=tuple(tfidf_raw["ngram_range"]),  # type: ignore[arg-type]
                stop_words=tfidf_raw.get("stop_words"),
                lowercase=bool(tfidf_raw.get("lowercase", True)),
            )
        )
        models = ModelsConfig(
            default=raw["models"]["default"],
            random_forest=RandomForestConfig(**raw["models"]["random_forest"]),
            logistic_regression=LogisticRegressionConfig(**raw["models"]["logistic_regression"]),
        )
        evaluation = EvaluationConfig(
            zero_division=int(raw["evaluation"]["zero_division"]),
            figure_size=tuple(raw["evaluation"]["figure_size"]),  # type: ignore[arg-type]
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc

    return Config(
        paths=paths,
        data=data,
        split=split,
        features=features,
        models=models,
        evaluation=evaluation,
        raw=raw,
    )
