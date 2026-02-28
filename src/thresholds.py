from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from .utils import safe_col

def _bisection_solve(f, lo: float, hi: float, target: float, max_iter: int = 60) -> float:
    flo = f(lo) - target
    fhi = f(hi) - target
    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        return np.nan
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fmid = f(mid) - target
        if np.isnan(fmid): return np.nan
        if flo * fmid <= 0:
            hi = mid; fhi = fmid
        else:
            lo = mid; flo = fmid
    return (lo + hi) / 2

def infer_soil_cd_thresholds(df_group: pd.DataFrame, cfg: Dict[str, Any], pipe: Pipeline, crop_key: str) -> pd.DataFrame:
    cols = cfg["data"]["columns"]
    soil_col = safe_col(df_group, cols["soil_cd"])
    target_bcf = safe_col(df_group, cols["target_bcf"])
    limit = float(cfg["gb"]["cd_limits_mgkg"][crop_key])
    lo = float(cfg["thresholds"]["soil_cd_min"])
    hi = float(cfg["thresholds"]["soil_cd_max"])
    feat_df = df_group.drop(columns=[target_bcf], errors="ignore")
    results = []
    for i in range(len(df_group)):
        row = feat_df.iloc[[i]].copy()
        def f(s):
            row2 = row.copy()
            row2[soil_col] = s
            bcf_pred = float(pipe.predict(row2)[0])
            return s * bcf_pred
        thr = _bisection_solve(f, lo, hi, limit)
        results.append(thr)
    out = df_group.copy()
    out["soil_cd_threshold_mgkg"] = results
    out["crop_cd_limit_mgkg"] = limit
    out["gb_standard"] = cfg["gb"]["standard"]
    return out
