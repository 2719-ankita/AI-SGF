
import os, json, random, numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def timestamped_dir(root: str, prefix: str):
    ensure_dir(root)
    import datetime as dt
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"{prefix}_{ts}")
    ensure_dir(path)
    return path
