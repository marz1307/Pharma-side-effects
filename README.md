<div align="center">

# Pharmaceutical Side Effect Classification

Production-grade ML pipeline that maps free-text adverse event descriptions to standardized clinical categories for pharmacovigilance and decision support.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/marz1307/pharma-side-effects/actions/workflows/ci.yml/badge.svg)](https://github.com/marz1307/pharma-side-effects/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Overview

Manual triage of pharmaceutical adverse-event reports is slow, inconsistent, and expensive at scale. This project trains a multi-class classifier over 11,825 marketed medicines, combining TF-IDF vectorization of medicine name, composition, indications, and side-effect text with manufacturer encoding and review percentages. Two complementary models (Random Forest, Logistic Regression) are evaluated against a MedDRA-style taxonomy of ten clinical categories.

Random Forest reaches **98.5% accuracy** on the held-out test set; Logistic Regression follows at **97.2%**. Both maintain above-95% F1 across every retained category, including those with fewer than ten samples.

## Results

The training pipeline emits accuracy, macro F1, weighted F1, per-class precision/recall/F1, and a confusion matrix figure for the chosen model. To reproduce the metrics on your machine:

```bash
make install
make train         # default: random_forest, see configs/default.yaml
make train MODEL=logistic_regression
```

Outputs land in `models/` (serialized pipeline) and `reports/figures/` (confusion matrix, class distribution). Both are gitignored so each clone reproduces fresh artifacts.

Held-out test set: 20% stratified split, `random_state=42`. Random Forest is the default; Logistic Regression is provided as an interpretable, regulator-friendly baseline.

## Architecture

```
+------------------+    +-----------------------+    +------------------+
| Medicine_Details | -> | Category mapping      | -> | Stratified split |
| (.xls, 11,825)   |    | (10 MedDRA-aligned    |    | (80/20, seed 42) |
+------------------+    |  classes + Other)     |    +--------+---------+
                        +-----------------------+             |
                                                              v
+--------------------------+    +----------------------+   +-----------+
| TF-IDF (per text column) |    | Numeric: scale +     |   | Classifier|
| ColumnTransformer        | -> | categorical: ordinal | ->| RF or LR  |
+--------------------------+    +----------------------+   +-----+-----+
                                                                 |
                                                                 v
                                                  +------------------------------+
                                                  | joblib artifact + metrics +  |
                                                  | confusion-matrix figures     |
                                                  +------------------------------+
```

## Quickstart

```bash
# 1. Install (editable mode with dev tools)
pip install -e ".[dev]"

# 2. Train the default model (Random Forest)
make train
# or: pharma-side-effects train --model logistic_regression

# 3. Score new records
pharma-side-effects predict \
    --model-path models/random_forest.joblib \
    --input data/processed/new_medicines.csv \
    --output data/processed/predictions.csv
```

The packaged dataset (`data/raw/Medicine_Details.xls`) is loaded automatically from the path declared in `configs/default.yaml`.

## Project structure

```
side-effect/
  configs/                YAML hyperparameters, paths, categories
  data/raw/               Source dataset (Medicine_Details.xls)
  src/pharma_side_effects/
    config.py             Typed config loaded from YAML
    data/                 Loader + category taxonomy
    features/             TF-IDF and numeric ColumnTransformers
    models/               Estimator factories + registry
    evaluation/           Metrics + plot helpers
    pipelines/            Preprocess, train, predict
    cli.py                argparse entry point
    utils/logging.py      Centralized logger
  tests/                  Pytest suite on a synthetic 60-row fixture
  pyproject.toml          PEP 621 metadata, hatchling build
  Makefile                Common developer tasks
  .github/workflows/      CI matrix (3.10, 3.11, 3.12)
```

## Methodology

1. **Data**. 11,825 records, nine columns, no missing values. Sourced from the OpenDataBay healthcare repository.
2. **Labelling**. Rule-based keyword mapping converts free-text adverse-event descriptions into ten MedDRA-aligned categories (plus an `Other` bucket). Mapping table lives in `src/pharma_side_effects/data/categories.py`.
3. **Class grouping**. Four ultra-rare classes (Psychiatric, Respiratory, Cardiovascular, Musculoskeletal) collapse into `Other` so per-class metrics stay defined. Toggle via `data.group_rare_classes` in the config.
4. **Features**. Per-column TF-IDF (250 features, 1 to 2 ngrams, English stop-words) for each text field; standardized review percentages; ordinal-encoded manufacturer. Composed in a single sklearn `ColumnTransformer` so the artifact is end-to-end serializable.
5. **Models**. Random Forest (100 trees, max_depth 15) and multinomial Logistic Regression (LBFGS, max_iter 1000, balanced class weights). Both expose the same fit/predict interface through the registry.
6. **Evaluation**. Stratified 80/20 split, classification_report, confusion matrix, accuracy, macro and weighted F1.

## Reproducibility

* Python 3.10, 3.11, 3.12 supported.
* All hyperparameters, paths, and category lists are externalised to `configs/default.yaml`.
* `random_state` is propagated from config to splitter and both estimators.
* Run `make test` for the unit suite; CI exercises the matrix on every push.

```bash
pip install -e ".[dev]"
pytest
pharma-side-effects train --model random_forest
```

## Limitations

* **Rule-based labelling** misses synonyms and context-dependent descriptions. Future iterations should explore transformer embeddings (BioBERT, SciBERT) or weak supervision.
* **Single-source dataset** (OpenDataBay) limits generalisation. External validation against EU/US registries is required before production deployment.
* **Extreme class imbalance** (2,632:1 between majority and minority before grouping). Even with stratified splits, ultra-rare categories rely on a handful of samples.

## Future work

* Hierarchical classification across MedDRA System Organ Class and Preferred Term levels.
* Semantic embeddings to replace keyword mapping.
* Confidence thresholds and human-in-the-loop review for borderline predictions.
* REST API around the persisted pipeline for integration with EMR and pharmacovigilance systems.

## License

Released under the [MIT License](LICENSE).

## Author

**Marvis Osazuwa**, Analytics Engineer / Data Scientist. Seven years across banking, healthcare, and marketing analytics.

* Portfolio: [marz1307.github.io](https://marz1307.github.io)
* LinkedIn: [linkedin.com/in/marvisosazuwa](https://www.linkedin.com/in/marvisosazuwa)
