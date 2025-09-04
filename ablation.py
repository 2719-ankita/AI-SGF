
import argparse, yaml, os, numpy as np, pandas as pd, itertools, json
from sklearn.model_selection import StratifiedKFold
from ai_sgf.data import standardize_features
from ai_sgf.models.autoencoder import AEConfig, train_autoencoder
from ai_sgf.synthetic import ATTACK_LABEL
from ai_sgf.metrics import compute_metrics
from ai_sgf.utils import set_seed, timestamped_dir

def run_subset(df, features, cfg):
    X, mu, sigma = standardize_features(df, features)
    y = df[ATTACK_LABEL].astype(int).values
    skf = StratifiedKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=cfg.get("seed",42))
    ms = []
    for tr, te in skf.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        Xtr_norm = Xtr[ytr==0]
        ae_cfg = AEConfig(input_dim=X.shape[1],
                          hidden_dims=cfg['models']['ai_sgf_autoencoder']['hidden_dims'],
                          lr=cfg['models']['ai_sgf_autoencoder']['lr'],
                          epochs=cfg['models']['ai_sgf_autoencoder']['epochs'],
                          batch_size=cfg['models']['ai_sgf_autoencoder']['batch_size'])
        model, re_tr = train_autoencoder(Xtr_norm, ae_cfg)
        import torch
        with torch.no_grad():
            Xte_t = torch.tensor(Xte, dtype=torch.float32)
            recon = model(Xte_t)
            re = ((recon - Xte_t)**2).mean(dim=1).numpy()
        mu_re = re_tr.mean(); sigma_re = re_tr.std() + 1e-8
        z = np.abs(re - mu_re) / sigma_re
        T = cfg.get('threshold_z', 3.0)
        yhat = (z > T).astype(int)
        ms.append(compute_metrics(yte, yhat))
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0] if k!="confusion_matrix"}

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed",42))
    df = pd.read_parquet("data/with_labels.parquet")
    all_feats = cfg['features']
    outdir = timestamped_dir("artifacts", "ablation")
    rows = []
    for k in range(1, len(all_feats)+1):
        for subset in itertools.combinations(all_feats, k):
            m = run_subset(df, list(subset), cfg)
            rows.append({"features": list(subset), **m})
    with open(os.path.join(outdir, "ablation.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Ablation results saved to {outdir}/ablation.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
