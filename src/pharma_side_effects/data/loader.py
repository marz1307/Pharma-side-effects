"""Dataset loading with input validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pharma_side_effects.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Medicine Name",
    "Composition",
    "Uses",
    "Side_effects",
    "Manufacturer",
)


def load_dataset(
    path: str | Path, required_columns: tuple[str, ...] = REQUIRED_COLUMNS
) -> pd.DataFrame:
    """Load the medicine dataset from disk with validation.

    Supports ``.xls``, ``.xlsx``, and ``.csv`` based on file suffix.

    Args:
        path: Path to the source file.
        required_columns: Column names that must be present.

    Returns:
        Loaded :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file has an unsupported suffix or is missing
            required columns.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(p)
    elif suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    logger.info("Loaded %d rows from %s", len(df), p)
    return df
