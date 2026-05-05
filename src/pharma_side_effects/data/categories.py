"""Standardized side-effect taxonomy and keyword mapping.

Categories mirror MedDRA System Organ Classes (SOC) so the labels feed
naturally into downstream pharmacovigilance workflows.
"""

from __future__ import annotations

from typing import Final

OTHER_CATEGORY: Final[str] = "Other"

# Ordered keyword map: first match wins, so more specific keywords
# (longer phrases) come earlier where ambiguity is possible.
CATEGORY_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "Gastrointestinal": (
        "nausea",
        "vomiting",
        "diarrhea",
        "constipation",
        "abdominal pain",
    ),
    "Neurological": ("headache", "dizziness", "seizure", "tremor"),
    "Dermatological": ("rash", "itch", "skin", "swelling", "photosensitivity"),
    "Cardiovascular": (
        "hypertension",
        "arrhythmia",
        "tachycardia",
        "palpitations",
    ),
    "Respiratory": ("cough", "bronchospasm", "shortness of breath"),
    "Musculoskeletal": ("muscle pain", "joint pain", "arthritis"),
    "Psychiatric": ("anxiety", "depression", "mood change"),
    "Allergic/Immunologic Reactions": (
        "allergy",
        "anaphylaxis",
        "hypersensitivity",
    ),
    "General Disorders & Administration Site Conditions": (
        "fatigue",
        "fever",
        "injection site reaction",
    ),
    "Hematologic & Laboratory Abnormalities": (
        "anemia",
        "leukopenia",
        "thrombocytopenia",
    ),
}

SIDE_EFFECT_CATEGORIES: Final[tuple[str, ...]] = tuple(CATEGORY_KEYWORDS.keys()) + (OTHER_CATEGORY,)


def map_side_effect(text: str | None) -> str:
    """Map a free-text side-effect description to a clinical category.

    Args:
        text: Raw side-effect description. ``None`` and non-string inputs
            are coerced to a string and lowercased.

    Returns:
        One of the categories in :data:`SIDE_EFFECT_CATEGORIES`. Returns
        :data:`OTHER_CATEGORY` when no keyword matches.
    """
    if text is None:
        return OTHER_CATEGORY
    haystack = str(text).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return category
    return OTHER_CATEGORY
