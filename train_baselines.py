
import argparse, yaml, os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from ai_sgf.data import standardize_features
from ai_sgf.models.dbn import build_dbn, DBNConfig
from ai_sgf.synthetic import ATTACK_LABEL
from ai_sgf.metrics import compute_metrics
from ai_sgf.utils import set_seed, timestamped_dir, save_json

def main(cfg_path, model_name):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed",42))
    df = pd.read_parquet("data/with_labels.parquet")
    features = cfg['features']
    X, mu, sigma = standardize_features(df, features)
    y = df[ATTACK_LABEL].astype(int).values

    outdir = timestamped_dir("artifacts", f"{model_name}")
    save_json({"features": features, "mu": mu.tolist(), "sigma": sigma.tolist()}, os.path.join(outdir, "scaler.json"))

    if model_name == "dbn":
        dbn_cfg = DBNConfig(
            rbm_layers=cfg['models']['dbn']['rbm_layers'],
            rbm_learning_rate=cfg['models']['dbn']['rbm_learning_rate'],
            rbm_n_iter=cfg['models']['dbn']['rbm_n_iter'],
            classifier_C=cfg['models']['dbn']['classifier_C'],
        )
        model = build_dbn(dbn_cfg, n_features=X.shape[1])
    else:
        raise ValueError("Unknown model")

    skf = StratifiedKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=cfg.get("seed",42))
    metrics = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        m = compute_metrics(yte, yhat)
        metrics.append(m)

    agg = {k: float(np.mean([m[k] for m in metrics if k in m])) for k in metrics[0] if k!="confusion_matrix"}
    save_json({"per_fold": metrics, "aggregate": agg}, os.path.join(outdir, "metrics.json"))
    print("DBN metrics (aggregate):", agg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, choices=["dbn"])
    args = ap.parse_args()
    main(args.config, args.model)
