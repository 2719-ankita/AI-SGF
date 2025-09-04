# AI-SGF: AI-Augmented Smart Grid Framework for Cyber Intrusion Detection in EV Charging

Reproducible code release for the paper:
*Artificial intelligence-augmented smart grid architecture for cyber intrusion detection and mitigation in electric vehicle charging infrastructure.*

## What this repo provides
- Deterministic pipeline (seeded) with **5-fold CV**
- **DBN baseline** (stacked RBM + Logistic Regression)
- **AI-SGF** anomaly detector (autoencoder + Gaussian z-threshold)
- Feature set: energy_kwh, duration_sec, avg_kw, session_cost, cost_per_kwh, start_hour, day_of_week
- Full logs + JSON metrics in `artifacts/`

> Note: The OpenEnergyHub dataset has **no native attack labels**. Provide your own `label` column **or** use the transparent synthetic labeling step (`scripts/inject_attacks.py`) which logs exactly what was injected.

## Quick start
```bash
conda env create -f environment.yml
conda activate ai-sgf-evcs
# or: pip install -r requirements.txt

# Put dataset at data/openenergyhub_ev_charging.csv (see DATA.md)
make reproduce
# or run steps:
# python scripts/preprocess.py --config config.yaml
# python scripts/inject_attacks.py --config config.yaml --attack-intensity medium
# python scripts/train_baselines.py --config config.yaml --model dbn
# python scripts/train_ai_sgf.py --config config.yaml
# python scripts/evaluate.py --config config.yaml
# python scripts/stats_ci.py --artifacts artifacts
```

See `RESPONSE_TO_EDITOR.md` for how this addresses each editorial concern.
