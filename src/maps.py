from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import shapefile as pyshp
except Exception:
    pyshp = None

DEFAULT_ADCODE = "520526"
GEOJSON_URL_FULL = "https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json"
GEOJSON_URL_SIMPLE = "https://geo.datav.aliyun.com/areas_v3/bound/{adcode}.json"


@dataclass
class MapCfg:
    lon_col: str = "lon"
    lat_col: str = "lat"
    out_subdir: str = "maps"
    vars_to_map: Optional[List[str]] = None
    by_group: bool = True
    s: float = 14.0
    alpha: float = 0.9
    clip_q_low: float = 0.01
    clip_q_high: float = 0.99
    draw_boundary: bool = True
    adcode: str = DEFAULT_ADCODE
    boundary_path: Optional[str] = None
    boundary_color: str = "#3D4B5A"
    boundary_width: float = 1.8
    point_crs: Optional[str] = None


def _get(cfg: Dict) -> MapCfg:
    m = cfg.get("map_output", {}) or {}
    cols = cfg.get("data", {}).get("columns", {}) if isinstance(cfg.get("data", {}), dict) else {}
    return MapCfg(
        lon_col=str(m.get("lon_col", cols.get("lon", "lon"))),
        lat_col=str(m.get("lat_col", cols.get("lat", "lat"))),
        out_subdir=str(m.get("out_subdir", "maps")),
        vars_to_map=m.get("vars_to_map", None),
        by_group=bool(m.get("by_group", True)),
        s=float(m.get("s", 14.0)),
        alpha=float(m.get("alpha", 0.9)),
        clip_q_low=float(m.get("clip_q_low", 0.01)),
        clip_q_high=float(m.get("clip_q_high", 0.99)),
        draw_boundary=bool(m.get("draw_boundary", True)),
        adcode=str(m.get("adcode", DEFAULT_ADCODE)),
        boundary_path=m.get("boundary_path", None),
        boundary_color=str(m.get("boundary_color", "#3D4B5A")),
        boundary_width=float(m.get("boundary_width", 1.8)),
        point_crs=m.get("point_crs", None),
    )


def _resolve_boundary_path(boundary_path: Optional[str]) -> Optional[Path]:
    if not boundary_path:
        return None
    p = Path(boundary_path)
    if p.exists():
        return p
    candidates = [
        Path.cwd() / p,
        Path(__file__).resolve().parents[1] / p,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _read_online_boundary(adcode: str):
    if gpd is None:
        return None
    for url in (GEOJSON_URL_FULL.format(adcode=adcode), GEOJSON_URL_SIMPLE.format(adcode=adcode)):
        try:
            return gpd.read_file(url)
        except Exception:
            continue
    return None


def _load_boundary(mc: MapCfg):
    if gpd is not None and mc.boundary_path:
        p = _resolve_boundary_path(mc.boundary_path)
        if p is not None:
            try:
                gdf = gpd.read_file(p)
                print(f"[maps] using local boundary: {p}")
                return gdf
            except Exception as e:
                print(f"[maps] local boundary read failed: {e}")

    gdf = _read_online_boundary(mc.adcode)
    if gdf is not None:
        print(f"[maps] using online boundary adcode={mc.adcode}")
        return gdf

    return None


def _draw_boundary(ax, mc: MapCfg):
    """Return boundary bounds in lon/lat when available."""
    if not mc.draw_boundary:
        return None

    # Preferred path: geopandas
    if gpd is not None:
        gdf = _load_boundary(mc)
        if gdf is None:
            return None
        try:
            if gdf.crs is None:
                # Fallback for missing CRS metadata.
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            gll = gdf.to_crs("EPSG:4326")
            gll = gll[~gll.geometry.isna() & ~gll.geometry.is_empty]
            if not gll.empty:
                # Prefer boundary-only rendering to avoid fill/edge style inconsistencies.
                try:
                    gll.boundary.plot(
                        ax=ax,
                        color=mc.boundary_color,
                        linewidth=max(1.8, mc.boundary_width),
                        zorder=1,
                    )
                except Exception:
                    gll.plot(
                        ax=ax,
                        facecolor="none",
                        edgecolor=mc.boundary_color,
                        linewidth=max(1.8, mc.boundary_width),
                        zorder=1,
                    )
            return tuple(float(v) for v in gll.total_bounds)
        except Exception as e:
            print(f"[maps] boundary draw failed: {e}")
            return None

    # Fallback path: pyshp local boundary only.
    if pyshp is not None and mc.boundary_path:
        p = _resolve_boundary_path(mc.boundary_path)
        if p is not None:
            try:
                reader = pyshp.Reader(str(p))
                for shp in reader.shapes():
                    pts = np.asarray(shp.points, dtype=float)
                    if pts.size == 0:
                        continue
                    parts = list(shp.parts) + [len(pts)]
                    for i in range(len(parts) - 1):
                        seg = pts[parts[i]:parts[i + 1]]
                        if len(seg) < 2:
                            continue
                        ax.plot(
                            seg[:, 0],
                            seg[:, 1],
                            color=mc.boundary_color,
                            linewidth=max(1.8, mc.boundary_width),
                            zorder=1,
                        )
                return tuple(float(v) for v in reader.bbox)
            except Exception as e:
                print(f"[maps] pyshp boundary draw failed: {e}")

    if mc.draw_boundary:
        print("[maps] boundary skipped: geopandas/pyshp unavailable or boundary not readable")
    return None


def plot_point_map(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    val_col: str,
    title: str,
    out_path: Path,
    mc: MapCfg,
):
    d = df.copy()
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna(subset=[lon_col, lat_col, val_col])
    if d.empty:
        return

    # Optional transform for projected point coordinates.
    if gpd is not None and mc.point_crs and str(mc.point_crs).upper() != "EPSG:4326":
        try:
            pts = gpd.GeoDataFrame(
                d[[lon_col, lat_col]].copy(),
                geometry=gpd.points_from_xy(d[lon_col], d[lat_col]),
                crs=mc.point_crs,
            ).to_crs("EPSG:4326")
            d[lon_col] = pts.geometry.x
            d[lat_col] = pts.geometry.y
        except Exception as e:
            print(f"[maps] point CRS transform failed ({mc.point_crs} -> EPSG:4326): {e}")

    try:
        vmin = d[val_col].quantile(mc.clip_q_low)
        vmax = d[val_col].quantile(mc.clip_q_high)
    except Exception:
        vmin = float(np.nanmin(d[val_col].to_numpy()))
        vmax = float(np.nanmax(d[val_col].to_numpy()))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Draw boundary first.
    bnd = _draw_boundary(ax, mc)

    sc = ax.scatter(
        d[lon_col],
        d[lat_col],
        c=d[val_col],
        s=mc.s,
        alpha=mc.alpha,
        cmap="Spectral_r",
        edgecolors="k",
        linewidth=0.3,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
    )

    # Set extent to include points and boundary.
    x0, x1 = float(d[lon_col].min()), float(d[lon_col].max())
    y0, y1 = float(d[lat_col].min()), float(d[lat_col].max())
    if bnd is not None:
        bx0, by0, bx1, by1 = bnd
        x0, x1 = min(x0, bx0), max(x1, bx1)
        y0, y1 = min(y0, by0), max(y1, by1)
    padx = (x1 - x0) * 0.02 if x1 > x0 else 0.01
    pady = (y1 - y0) * 0.02 if y1 > y0 else 0.01
    ax.set_xlim(x0 - padx, x1 + padx)
    ax.set_ylim(y0 - pady, y1 + pady)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.set_title(val_col, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [maps] saved: {out_path.name}")


def make_maps(df: pd.DataFrame, cfg: Dict, out_root: Path) -> List[Path]:
    mc = _get(cfg)
    out_dir = Path(out_root) / mc.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    lon, lat = mc.lon_col, mc.lat_col
    if lon not in df.columns or lat not in df.columns:
        found = False
        for cand_lon in ["longitude", "long", "lng", "x", "X", "Lon", "LON"]:
            for cand_lat in ["latitude", "lat", "y", "Y", "Lat", "LAT"]:
                if cand_lon in df.columns and cand_lat in df.columns:
                    lon, lat = cand_lon, cand_lat
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"[maps] lon/lat columns not found: ({lon}, {lat})")
            return []

    vars_: List[str] = []
    if mc.vars_to_map:
        vars_ = [v for v in mc.vars_to_map if v in df.columns]
    else:
        priority = ["BCF", "bcf", "soil Cd", "soil_cd", "crop Cd", "crop_cd", "pH", "ph"]
        for p in priority:
            for col in df.columns:
                if p.lower() in col.lower() and col not in vars_:
                    vars_.append(col)
        exclude_keywords = [lon.lower(), lat.lower(), "bin", "type", "class", "name", "id", "code"]
        for col in df.select_dtypes(include=[np.number]).columns.tolist():
            if col in vars_:
                continue
            if any(k in col.lower() for k in exclude_keywords):
                continue
            vars_.append(col)
            if len(vars_) >= 8:
                break

    print(f"[maps] vars: {vars_}")
    print(f"[maps] lon/lat: {lon}, {lat}")

    outs: List[Path] = []
    if (not mc.by_group) or ("crop" not in df.columns):
        for v in vars_:
            safe = str(v).replace(" ", "_").replace("/", "_")
            out_png = out_dir / f"map_all_{safe}.png"
            plot_point_map(df, lon, lat, v, f"Map | {v} (All Data)", out_png, mc)
            outs.append(out_png)
        return outs

    group_cols = [c for c in ["crop", "ph_bin"] if c in df.columns]
    if not group_cols:
        for v in vars_:
            safe = str(v).replace(" ", "_").replace("/", "_")
            out_png = out_dir / f"map_all_{safe}.png"
            plot_point_map(df, lon, lat, v, f"Map | {v} (All Data)", out_png, mc)
            outs.append(out_png)
        return outs

    for name, sub in df.groupby(group_cols):
        if len(sub) < 5:
            continue
        if isinstance(name, tuple):
            group_tag = "_".join([str(n) for n in name])
            title_tag = " | ".join([str(n) for n in name])
        else:
            group_tag = str(name)
            title_tag = str(name)

        for v in vars_:
            if sub[v].dropna().empty:
                continue
            safe = str(v).replace(" ", "_").replace("/", "_")
            fname = f"map_{group_tag}_{safe}.png".replace("<", "lt").replace(">", "gt").replace(":", "")
            out_png = out_dir / fname
            plot_point_map(sub, lon, lat, v, f"{title_tag} | {v}", out_png, mc)
            outs.append(out_png)

    return outs
