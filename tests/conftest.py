"""Shared pytest fixtures.

Tests rely on a deterministic synthetic dataset rather than the 4MB
production xls so the suite stays fast and hermetic.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from pharma_side_effects.config import load_config

SYNTHETIC_ROWS = 60

_SAMPLES = [
    ("Para Tabs", "paracetamol", "fever", "nausea and vomiting", "PharmaCo"),
    ("Ibu Caps", "ibuprofen", "pain", "headache and dizziness", "MediCorp"),
    ("Skin Cream", "hydrocortisone", "rash", "itch and skin irritation", "DermaLab"),
    ("Heart Tabs", "atenolol", "hypertension", "palpitations and tachycardia", "CardioInc"),
    ("Cough Syrup", "dextromethorphan", "cough", "cough and shortness of breath", "RespCo"),
    ("Joint Relief", "naproxen", "arthritis", "joint pain and muscle pain", "OrthoMed"),
    ("Mood Tabs", "sertraline", "depression", "anxiety and mood change", "MentalCare"),
    ("Allergy Tabs", "cetirizine", "allergy", "hypersensitivity reactions", "AllergyCo"),
    ("Vit Boost", "iron", "anemia", "fatigue and fever", "GenLab"),
    ("Blood Tabs", "warfarin", "clot prevention", "anemia and leukopenia", "HemaLab"),
]


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Return a small but realistic dataframe matching the production schema."""
    rng = random.Random(42)
    rows = []
    for i in range(SYNTHETIC_ROWS):
        name, comp, uses, side, mfr = _SAMPLES[i % len(_SAMPLES)]
        rows.append(
            {
                "Medicine Name": f"{name} {i}",
                "Composition": comp,
                "Uses": uses,
                "Side_effects": side,
                "Image URL": "http://example.com/img.png",
                "Manufacturer": mfr,
                "Excellent Review %": rng.randint(20, 90),
                "Average Review %": rng.randint(5, 60),
                "Poor Review %": rng.randint(0, 30),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory, synthetic_df: pd.DataFrame) -> Path:
    """Persist the synthetic dataframe so file-loader tests have a target."""
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    synthetic_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def config():
    """Loaded default configuration."""
    return load_config()
