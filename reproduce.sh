#!/usr/bin/env bash
set -euo pipefail
python scripts/preprocess.py --config config.yaml
python scripts/inject_attacks.py --config config.yaml --attack-intensity medium
python scripts/train_baselines.py --config config.yaml --model dbn
python scripts/train_ai_sgf.py --config config.yaml
python scripts/evaluate.py --config config.yaml
python scripts/stats_ci.py --artifacts artifacts
