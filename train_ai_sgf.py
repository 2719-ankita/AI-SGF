
import argparse, yaml, os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from ai_sgf.data import standardize_features
from ai_sgf.models.autoencoder import AEConfig, train_autoencoder
from ai_sgf.synthetic import ATTACK_LABEL
from ai_sgf.metrics import compute_metrics
from ai_sgf.utils import set_seed, timestamped_dir, save_json

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed",42))
    df = pd.read_parquet("data/with_labels.parquet")
    features = cfg['features']
    X, mu, sigma = standardize_features(df, features)
    y = df[ATTACK_LABEL].astype(int).values

    outdir = timestamped_dir("artifacts", "ai_sgf")
    save_json({"features": features, "mu": mu.tolist(), "sigma": sigma.tolist()}, os.path.join(outdir, "scaler.json"))

    skf = StratifiedKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=cfg.get("seed",42))
    metrics = []
    zs = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        Xtr_norm = Xtr[ytr==0]

        ae_cfg = AEConfig(input_dim=X.shape[1],
                          hidden_dims=cfg['models']['ai_sgf_autoencoder']['hidden_dims'],
                          lr=cfg['models']['ai_sgf_autoencoder']['lr'],
                          epochs=cfg['models']['ai_sgf_autoencoder']['epochs'],
                          batch_size=cfg['models']['ai_sgf_autoencoder']['batch_size'],
                          device="cpu")
        model, re_tr = train_autoencoder(Xtr_norm, ae_cfg)

        import torch
        with torch.no_grad():
            Xte_t = torch.tensor(Xte, dtype=torch.float32)
            recon = model(Xte_t)
            re = ((recon - Xte_t)**2).mean(dim=1).numpy()

        mu_re = re_tr.mean()
        sigma_re = re_tr.std() + 1e-8
        z = np.abs(re - mu_re) / sigma_re
        zs.append(z.mean())
        T = cfg.get('threshold_z', 3.0)
        yhat = (z > T).astype(int)
        m = compute_metrics(yte, yhat)
        metrics.append(m)

    agg = {k: float(np.mean([m[k] for m in metrics if k in m])) for k in metrics[0] if k!="confusion_matrix"}
    save_json({"per_fold": metrics, "aggregate": agg, "avg_z": float(np.mean(zs))}, os.path.join(outdir, "metrics.json"))
    print("AI-SGF metrics (aggregate):", agg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
