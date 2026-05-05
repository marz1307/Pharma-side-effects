"""TF-IDF text feature builders.

The original analysis used parallel ``ColumnSelector`` transformers feeding
per-column ``TfidfVectorizer`` instances combined via ``FeatureUnion``.
This module exposes a single helper returning a ``ColumnTransformer``
that achieves the same result while remaining a clean sklearn primitive.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from pharma_side_effects.config import TfidfConfig


def build_text_feature_union(text_columns: list[str], cfg: TfidfConfig) -> ColumnTransformer:
    """Build a :class:`ColumnTransformer` of TF-IDF vectorizers.

    Each text column gets its own vectorizer so vocabularies do not
    contaminate one another. Outputs are horizontally stacked into a
    single sparse matrix.

    Args:
        text_columns: Columns to vectorize.
        cfg: TF-IDF hyperparameters.

    Returns:
        Configured :class:`ColumnTransformer`.
    """
    transformers = [
        (
            f"tfidf__{col}",
            TfidfVectorizer(
                max_features=cfg.max_features,
                ngram_range=cfg.ngram_range,
                stop_words=cfg.stop_words,
                lowercase=cfg.lowercase,
            ),
            col,
        )
        for col in text_columns
    ]
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )
