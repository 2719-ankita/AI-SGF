
import pandas as pd
import numpy as np

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def harmonize_columns(df: pd.DataFrame, cfg) -> pd.DataFrame:
    s = cfg['data']
    dt_start = s['datetime_col_start']
    dt_end   = s['datetime_col_end']
    e_col    = s['energy_kwh_col']
    cost_col = s['session_cost_col']
    site_col = s['site_id_col']
    ren = {}
    if dt_start in df.columns: ren[dt_start] = 'start_time'
    if dt_end   in df.columns: ren[dt_end]   = 'end_time'
    if e_col    in df.columns: ren[e_col]    = 'energy_kwh'
    if cost_col in df.columns: ren[cost_col] = 'session_cost'
    if site_col in df.columns: ren[site_col] = 'site_id'
    df = df.rename(columns=ren)
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time']   = pd.to_datetime(df['end_time'], errors='coerce')
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['duration_sec'] = (out['end_time'] - out['start_time']).dt.total_seconds()
    out['avg_kw'] = out['energy_kwh'] / (out['duration_sec'] / 3600.0)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    if 'session_cost' in out.columns:
        out['cost_per_kwh'] = np.where(out['energy_kwh']>0, out['session_cost']/out['energy_kwh'], np.nan)
    else:
        out['cost_per_kwh'] = np.nan
    out['start_hour'] = out['start_time'].dt.hour
    out['day_of_week'] = out['start_time'].dt.dayofweek
    return out

def standardize_features(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].astype(float).values
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0) + 1e-8
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    Xz = (X - mu) / sigma
    return Xz, mu, sigma
