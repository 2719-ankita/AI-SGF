
import argparse, yaml, os, glob, json

def main(cfg_path):
    arts = sorted(glob.glob("artifacts/*"), key=os.path.getmtime)
    if not arts:
        print("No artifacts found.")
        return
    latest = arts[-1]
    mfile = os.path.join(latest, "metrics.json")
    if not os.path.exists(mfile):
        print(f"No metrics in {latest}")
        return
    with open(mfile) as f:
        metrics = json.load(f)
    print(f"Latest run: {latest}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
