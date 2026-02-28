from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from .utils import safe_col

def data_audit(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in df.columns:
        miss = int(df[c].isna().sum())
        rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "missing_n": miss,
            "missing_rate": miss / n if n else 0.0,
            "n_unique": int(df[c].nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values(["missing_rate","n_unique"], ascending=[False, True])

def enforce_required_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    cols = cfg["data"]["columns"]
    safe_col(df, cols["target_bcf"])
    safe_col(df, cols["ph"])
    safe_col(df, cols["x"])
    safe_col(df, cols["y"])
