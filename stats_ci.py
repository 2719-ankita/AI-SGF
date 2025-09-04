
import argparse, json, numpy as np, glob, os

def bootstrap_ci(scores, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    scores = np.asarray(scores)
    for _ in range(n_boot):
        sample = rng.choice(scores, size=len(scores), replace=True)
        boots.append(sample.mean())
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return lo, hi

def main(artifacts_dir):
    runs = sorted(glob.glob(os.path.join(artifacts_dir, "*")), key=os.path.getmtime)
    if not runs:
        print("No artifacts found.")
        return
    latest = runs[-1]
    mfile = os.path.join(latest, "metrics.json")
    if not os.path.exists(mfile):
        print(f"No metrics at {latest}")
        return
    with open(mfile) as f:
        M = json.load(f)
    per_fold = M.get("per_fold", [])
    for metric in ["accuracy","precision","recall","f1"]:
        vals = [m[metric] for m in per_fold if metric in m]
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        print(f"{metric} 95% CI=({lo:.4f}, {hi:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    main(args.artifacts)
