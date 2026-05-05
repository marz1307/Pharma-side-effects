"""Data loading and category mapping."""

from pharma_side_effects.data.categories import (
    CATEGORY_KEYWORDS,
    SIDE_EFFECT_CATEGORIES,
    map_side_effect,
)
from pharma_side_effects.data.loader import load_dataset

__all__ = [
    "CATEGORY_KEYWORDS",
    "SIDE_EFFECT_CATEGORIES",
    "load_dataset",
    "map_side_effect",
]
