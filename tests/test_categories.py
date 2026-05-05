"""Tests for the side-effect category mapping."""

from __future__ import annotations

import pytest

from pharma_side_effects.data.categories import (
    CATEGORY_KEYWORDS,
    OTHER_CATEGORY,
    SIDE_EFFECT_CATEGORIES,
    map_side_effect,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Patient reports nausea after dose", "Gastrointestinal"),
        ("Severe headache and dizziness", "Neurological"),
        ("Skin rash and itching", "Dermatological"),
        ("Tachycardia observed", "Cardiovascular"),
        ("Persistent cough", "Respiratory"),
        ("Joint pain and muscle pain", "Musculoskeletal"),
        ("Reports of anxiety", "Psychiatric"),
        ("Anaphylaxis risk", "Allergic/Immunologic Reactions"),
        ("Fatigue and fever", "General Disorders & Administration Site Conditions"),
        ("Anemia detected", "Hematologic & Laboratory Abnormalities"),
        ("Completely unrelated text", OTHER_CATEGORY),
        (None, OTHER_CATEGORY),
    ],
)
def test_map_side_effect(text: str | None, expected: str) -> None:
    assert map_side_effect(text) == expected


def test_categories_match_keywords() -> None:
    for cat in CATEGORY_KEYWORDS:
        assert cat in SIDE_EFFECT_CATEGORIES
    assert OTHER_CATEGORY in SIDE_EFFECT_CATEGORIES


def test_keyword_lists_non_empty() -> None:
    for cat, keywords in CATEGORY_KEYWORDS.items():
        assert keywords, f"{cat} has no keywords"
        assert all(isinstance(k, str) and k for k in keywords)
