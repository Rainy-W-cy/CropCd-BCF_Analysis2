from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import re
import pandas as pd
from .utils import safe_col

SHEET_CROP_PATTERNS = [(re.compile(r"马铃薯"), "potato"), (re.compile(r"玉米"), "corn")]
SHEET_PH_PATTERNS = [(re.compile(r"强酸"), "strong_acid"), (re.compile(r"酸性"), "acid"),
                     (re.compile(r"中性"), "neutral"), (re.compile(r"碱性"), "alkaline")]

def _infer_crop_and_phbin(sheet_name: str) -> Tuple[Optional[str], Optional[str]]:
    crop, phbin = None, None
    for pat, val in SHEET_CROP_PATTERNS:
        if pat.search(sheet_name):
            crop = val; break
    for pat, val in SHEET_PH_PATTERNS:
        if pat.search(sheet_name):
            phbin = val; break
    return crop, phbin

def read_excel_as_long(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = cfg["data"]["excel_path"]
    mode = cfg["data"]["mode"]
    if mode == "table":
        sheet = cfg["data"].get("sheet_name") or 0
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        df["source_sheet"] = str(sheet)
        return df
    all_sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames = []
    for name, df in all_sheets.items():
        d = df.copy()
        d["source_sheet"] = name
        crop, phbin = _infer_crop_and_phbin(name)
        if crop is not None:
            d["crop"] = crop
        if phbin is not None:
            d["ph_bin_from_sheet"] = phbin
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

def add_ph_bins(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    ph_col = safe_col(df, cfg["data"]["columns"]["ph"])
    df = df.copy()
    df[ph_col] = pd.to_numeric(df[ph_col], errors="coerce")
    # Coerce YAML edges (may contain "inf"/"-inf" as strings) to floats
    edges = [float(x) for x in cfg["ph_bins"]["edges"]]
    labels = cfg["ph_bins"]["labels"]
    df["ph_bin"] = pd.cut(df[ph_col], bins=edges, labels=labels, include_lowest=True).astype("string")
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all")
