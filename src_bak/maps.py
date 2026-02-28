from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class MapCfg:
    lon_col: str = "lon"
    lat_col: str = "lat"
    out_subdir: str = "maps"
    vars_to_map: Optional[List[str]] = None
    by_group: bool = True
    s: float = 18.0
    alpha: float = 0.85
    clip_q_low: float = 0.02
    clip_q_high: float = 0.98

def _get(cfg: Dict) -> MapCfg:
    m = cfg.get("map_output", {}) or {}
    cols = cfg.get("data", {}).get("columns", {}) if isinstance(cfg.get("data", {}), dict) else {}
    return MapCfg(
        lon_col=str(m.get("lon_col", cols.get("lon", "lon"))),
        lat_col=str(m.get("lat_col", cols.get("lat", "lat"))),
        out_subdir=str(m.get("out_subdir", "maps")),
        vars_to_map=m.get("vars_to_map", None),
        by_group=bool(m.get("by_group", True)),
        s=float(m.get("s", 18.0)),
        alpha=float(m.get("alpha", 0.85)),
        clip_q_low=float(m.get("clip_q_low", 0.02)),
        clip_q_high=float(m.get("clip_q_high", 0.98)),
    )

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def plot_point_map(df: pd.DataFrame, lon: str, lat: str, val: str, title: str, out_png: Path, cfg: MapCfg):
    d = df[[lon, lat, val]].copy()
    d[lon] = _to_num(d[lon]); d[lat] = _to_num(d[lat]); d[val] = _to_num(d[val])
    d = d.dropna()
    if d.empty:
        return

    v = d[val].to_numpy()
    vmin = np.nanquantile(v, cfg.clip_q_low)
    vmax = np.nanquantile(v, cfg.clip_q_high)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(v), np.nanmax(v)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(d[lon], d[lat], c=d[val], s=cfg.s, alpha=cfg.alpha, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(lon); ax.set_ylabel(lat)

    # make it look like a map (equal aspect & tight extent)
    ax.set_aspect("equal", adjustable="box")
    x0, x1 = d[lon].min(), d[lon].max()
    y0, y1 = d[lat].min(), d[lat].max()
    padx = (x1 - x0) * 0.05 if x1 > x0 else 0.01
    pady = (y1 - y0) * 0.05 if y1 > y0 else 0.01
    ax.set_xlim(x0 - padx, x1 + padx)
    ax.set_ylim(y0 - pady, y1 + pady)

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(val)
    fig.tight_layout()
    fig.savefig(out_png, dpi=280)
    plt.close(fig)

def make_maps(df: pd.DataFrame, cfg: Dict, out_dir: str | Path) -> List[Path]:
    mc = _get(cfg)
    out_dir = Path(out_dir) / mc.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    lon = mc.lon_col
    lat = mc.lat_col
    if lon not in df.columns or lat not in df.columns:
        for cand_lon in ["longitude", "lng", "x", "X", "Lon", "LON"]:
            for cand_lat in ["latitude", "lat", "y", "Y", "Lat", "LAT"]:
                if cand_lon in df.columns and cand_lat in df.columns:
                    lon, lat = cand_lon, cand_lat

    if mc.vars_to_map:
        vars_ = [v for v in mc.vars_to_map if v in df.columns]
    else:
        vars_ = []
        for v in ["bcf", "bcf_calc", "soil_cd_mgkg", "crop_cd_mgkg", "ph", "pH"]:
            if v in df.columns:
                vars_.append(v)
        more = df.select_dtypes(include=[np.number]).columns.tolist()
        for v in more:
            if v not in vars_ and v.lower() not in [lon.lower(), lat.lower()]:
                vars_.append(v)
            if len(vars_) >= 6:
                break

    outs: List[Path] = []
    if (not mc.by_group) or ("crop" not in df.columns) or ("ph_bin" not in df.columns):
        for v in vars_:
            out_png = out_dir / f"map_all_{v}.png"
            plot_point_map(df, lon, lat, v, f"Point map | {v} (all)", out_png, mc)
            outs.append(out_png)
        return outs

    for (crop, ph), sub in df.groupby(["crop", "ph_bin"], dropna=False):
        if len(sub) < 10:
            continue
        safe = f"{crop}_{ph}".replace(" ", "")
        for v in vars_:
            out_png = out_dir / f"map_{safe}_{v}.png"
            plot_point_map(sub, lon, lat, v, f"Point map | {v} | ({crop}, {ph})", out_png, mc)
            outs.append(out_png)

    return outs
