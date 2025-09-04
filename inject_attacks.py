
import argparse, yaml, os, json
import pandas as pd
from ai_sgf.synthetic import inject_synthetic_attacks, ATTACK_LABEL
from ai_sgf.utils import ensure_dir

def main(cfg_path, intensity):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    df = pd.read_parquet("data/processed.parquet")
    df2 = inject_synthetic_attacks(df, intensity=intensity, seed=cfg.get('seed', 42))
    ensure_dir("data")
    out_path = f"data/with_labels.parquet"
    df2.to_parquet(out_path, index=False)
    n_attacks = int(df2[ATTACK_LABEL].sum())
    with open("data/attack_injection.log", "w") as f:
        f.write(f"Injected {n_attacks} attacks at intensity={intensity}\n")
    print(f"Saved {out_path}. Injected attacks: {n_attacks}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--attack-intensity", default="medium", choices=["low","medium","high"])
    args = ap.parse_args()
    main(args.config, args.attack_intensity)
