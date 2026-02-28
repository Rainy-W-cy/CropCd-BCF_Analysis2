from __future__ import annotations
import os, random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

def load_config(config_path: str | os.PathLike) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def safe_col(df, col: str) -> str:
    if col in df.columns:
        return col
    lower_map = {c.lower(): c for c in df.columns}
    if col.lower() in lower_map:
        return lower_map[col.lower()]
    raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)[:30]}...")
