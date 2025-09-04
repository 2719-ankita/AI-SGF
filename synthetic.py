
import numpy as np
import pandas as pd

ATTACK_LABEL = "label"  # 0 normal, 1 attack

def inject_synthetic_attacks(df: pd.DataFrame, intensity: str = "medium", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    if intensity == "low":
        frac = 0.05
    elif intensity == "high":
        frac = 0.25
    else:
        frac = 0.12
    
    n_attacks = int(max(1, frac * n))
    out[ATTACK_LABEL] = 0

    idx = rng.choice(n, size=n_attacks, replace=False)
    splits = np.array_split(idx, 4)
    # Overload / energy depletion
    for i in splits[0]:
        if 'avg_kw' in out.columns and 'duration_sec' in out.columns:
            out.loc[i, 'avg_kw'] = out.loc[i, 'avg_kw'] * rng.uniform(2.0, 5.0)
            out.loc[i, 'duration_sec'] = max(60.0, float(out.loc[i, 'duration_sec']) * rng.uniform(0.2, 0.6))
            out.loc[i, ATTACK_LABEL] = 1
    # Billing manipulation
    for i in splits[1]:
        if 'session_cost' in out.columns:
            out.loc[i, 'session_cost'] = float(out.loc[i, 'session_cost']) * rng.uniform(0.1, 0.5)
        if 'cost_per_kwh' in out.columns and not pd.isna(out.loc[i, 'cost_per_kwh']):
            out.loc[i, 'cost_per_kwh'] = float(out.loc[i, 'cost_per_kwh']) * rng.uniform(3.0, 8.0)
        out.loc[i, ATTACK_LABEL] = 1
    # Session spoofing
    for i in splits[2]:
        j = int(rng.integers(0, n))
        if 'start_time' in out.columns:
            out.loc[i, 'start_time'] = out.loc[j, 'start_time']
        if 'energy_kwh' in out.columns:
            out.loc[i, 'energy_kwh'] = float(out.loc[i, 'energy_kwh']) * rng.uniform(0.05, 0.2)
        out.loc[i, ATTACK_LABEL] = 1
    # False data injection
    for i in splits[3]:
        for c in ['energy_kwh','avg_kw','duration_sec','cost_per_kwh']:
            if c in out.columns and not pd.isna(out.loc[i, c]):
                out.loc[i, c] = float(out.loc[i, c]) + rng.normal(0, 5) + rng.choice([-20, 20])
        out.loc[i, ATTACK_LABEL] = 1
    return out
