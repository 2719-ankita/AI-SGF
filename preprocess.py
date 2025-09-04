
import argparse, yaml, os, pandas as pd, numpy as np
from ai_sgf.data import load_csv, harmonize_columns, feature_engineer
from ai_sgf.utils import ensure_dir

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    csv_path = cfg['data']['csv_path']
    df = load_csv(csv_path)
    df = harmonize_columns(df, cfg)
    df = feature_engineer(df)
    df = df.dropna(subset=['start_time','end_time','energy_kwh','duration_sec'])
    ensure_dir("data")
    out_path = "data/processed.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {out_path} with {len(df)} rows.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
