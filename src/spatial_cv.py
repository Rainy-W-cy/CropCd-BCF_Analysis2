from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from .utils import safe_col

def make_spatial_groups(df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    x_col = safe_col(df, cfg["data"]["columns"]["x"])
    y_col = safe_col(df, cfg["data"]["columns"]["y"])
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    method = cfg["spatial_cv"]["method"]
    if method == "grid":
        nx = int(cfg["spatial_cv"]["grid"]["n_bins_x"])
        ny = int(cfg["spatial_cv"]["grid"]["n_bins_y"])
        x_edges = np.quantile(x[~np.isnan(x)], np.linspace(0, 1, nx+1))
        y_edges = np.quantile(y[~np.isnan(y)], np.linspace(0, 1, ny+1))
        gx = np.digitize(x, x_edges[1:-1], right=False)
        gy = np.digitize(y, y_edges[1:-1], right=False)
        return (gx*ny + gy).astype(int)
    if method == "kmeans":
        k = int(cfg["spatial_cv"]["kmeans"]["n_clusters"])
        coords = np.column_stack([x, y])
        mask = ~np.isnan(coords).any(axis=1)
        labels = np.full(len(df), -1, dtype=int)
        km = KMeans(n_clusters=k, random_state=int(cfg["project"]["seed"]), n_init=10)
        labels[mask] = km.fit_predict(coords[mask])
        labels[~mask] = labels.max() + 1
        return labels
    raise ValueError("Unknown spatial_cv.method")

def get_group_kfold(cfg: Dict[str, Any]) -> GroupKFold:
    return GroupKFold(n_splits=int(cfg["spatial_cv"]["n_splits"]))
