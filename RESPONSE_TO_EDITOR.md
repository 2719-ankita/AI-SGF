
# Response to Editor & Reviewers (Code & Reproducibility)

This repository contains the **complete, deterministic pipeline** corresponding to the paper.

- **Methodology clarity**: All steps (preprocess, features, models, thresholds) are explicit in code and `config.yaml`.
- **Data access**: see `DATA.md` for dataset and schema mapping.
- **Reproducibility**: `Makefile` and `reproduce.sh` produce artifacts with metrics and configuration snapshots.
- **Statistics**: `scripts/stats_ci.py` reports 95% bootstrap CIs over CV folds.
- **Overfitting controls**: Train AE only on normal sessions within each fold, evaluate on held-out fold; threshold and epochs are configurable.
- **Baselines & parity**: Stacked RBM + Logistic Regression baseline; `scripts/hparam_search.py` for fair tuning.
- **Ablation**: `scripts/ablation.py` quantifies feature contributions.
