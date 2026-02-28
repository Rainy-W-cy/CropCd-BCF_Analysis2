from __future__ import annotations
from typing import Any, Dict
import pandas as pd
import numpy as np
from .utils import safe_col

def apply_bcf_rules(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    cols = cfg["data"]["columns"]
    bcf_col = safe_col(df, cols["target_bcf"])
    soil_col = cols.get("soil_cd")
    crop_col = cols.get("crop_cd")
    out = df.copy()
    if soil_col in out.columns and crop_col in out.columns:
        soil = pd.to_numeric(out[soil_col], errors="coerce")
        crop = pd.to_numeric(out[crop_col], errors="coerce")
        lod = cfg.get("lod", {}).get("crop_cd_mgkg", None)
        lod_used = False
        if lod is not None:
            crop = crop.mask(crop == 0, lod/2.0)
            lod_used = True
        out["bcf_calc"] = crop / soil
        out["lod_used"] = lod_used
    else:
        out["bcf_calc"] = np.nan
        out["lod_used"] = False
    out[bcf_col] = pd.to_numeric(out[bcf_col], errors="coerce")
    return out
