
import argparse, yaml, os, numpy as np, pandas as pd, itertools, json
from sklearn.model_selection import StratifiedKFold
from ai_sgf.data import standardize_features
from ai_sgf.models.dbn import build_dbn, DBNConfig
from ai_sgf.synthetic import ATTACK_LABEL
from ai_sgf.metrics import compute_metrics
from ai_sgf.utils import set_seed, timestamped_dir

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed",42))
    df = pd.read_parquet("data/with_labels.parquet")
    features = cfg['features']
    X, mu, sigma = standardize_features(df, features)
    y = df[ATTACK_LABEL].astype(int).values

    search_space = {
        "rbm_layers": [[256,128],[128,64],[64,32]],
        "rbm_learning_rate": [0.005, 0.01, 0.02],
        "rbm_n_iter": [10, 20, 40],
        "classifier_C": [0.5, 1.0, 2.0],
    }
    combos = list(itertools.product(*search_space.values()))
    skf = StratifiedKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=cfg.get("seed",42))
    results = []
    for vals in combos:
        params = dict(zip(search_space.keys(), vals))
        dbn_cfg = DBNConfig(**params)
        ms = []
        for tr, te in skf.split(X,y):
            model = build_dbn(dbn_cfg, n_features=X.shape[1])
            model.fit(X[tr], y[tr])
            yhat = model.predict(X[te])
            ms.append(compute_metrics(y[te], yhat))
        agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0] if k!="confusion_matrix"}
        results.append({"params": params, "metrics": agg})
        print("Tried:", params, "->", agg)
    outdir = timestamped_dir("artifacts", "hparam_dbn")
    with open(os.path.join(outdir, "grid_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved grid search to {outdir}/grid_results.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
