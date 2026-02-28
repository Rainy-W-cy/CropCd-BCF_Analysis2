from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir, safe_col


@dataclass
class QCPaths:
    qc_dir: Path
    fig_dir: Path
    missing_rate_xlsx: Path
    numeric_desc_xlsx: Path
    bcf_quantiles_xlsx: Path
    group_stats_xlsx: Optional[Path]
    notes_xlsx: Path
    bcf_hist_png: Path
    bcf_box_png: Path


def _set_paper_fonts():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "axes.unicode_minus": False,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def run_optional_qc(
    df: pd.DataFrame,
    cfg: Dict,
    out_base: str | Path,
    *,
    group_cols: Tuple[str, ...] = ("crop", "ph_bin"),
) -> QCPaths:
    """
    Optional QC step:
    - does NOT create bcf_calc
    - does NOT write bcf_final.parquet
    - only reads BCF (target) and outputs QC tables/figures under outputs/qc/
    """
    out_base = Path(out_base)
    qc_dir = ensure_dir(out_base / "qc")
    fig_dir = ensure_dir(qc_dir / "figures")

    cols = (cfg.get("data", {}) or {}).get("columns", {}) or {}
    target_bcf = safe_col(df, cols.get("target_bcf", "BCF"))
    x_col = cols.get("x", "X")
    y_col = cols.get("y", "Y")
    ph_col = cols.get("ph", "pH")

    # 1) Missing rate
    miss = df.isna().mean().sort_values(ascending=False)
    miss_df = miss.reset_index()
    miss_df.columns = ["column", "missing_rate"]
    missing_rate_xlsx = qc_dir / "qc_missing_rate.xlsx"
    miss_df.to_excel(missing_rate_xlsx, index=False)

    # 2) Numeric describe
    num = df.select_dtypes(include="number")
    desc = num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    numeric_desc_xlsx = qc_dir / "qc_numeric_describe.xlsx"
    desc.to_excel(numeric_desc_xlsx)

    # 3) BCF quantiles + plots
    y = pd.to_numeric(df[target_bcf], errors="coerce")
    y = y[np.isfinite(y)]
    qs = y.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_frame("value")
    bcf_quantiles_xlsx = qc_dir / "qc_bcf_quantiles.xlsx"
    qs.to_excel(bcf_quantiles_xlsx)

    _set_paper_fonts()

    # histogram
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(y.values, bins=40, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("BCF")
    ax.set_ylabel("Count")
    ax.set_title("BCF distribution")
    fig.tight_layout()
    bcf_hist_png = fig_dir / "qc_bcf_hist.png"
    fig.savefig(bcf_hist_png, dpi=320)
    plt.close(fig)

    # boxplot
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.boxplot(y.values, vert=False)
    ax.set_xlabel("BCF")
    ax.set_title("BCF boxplot (outliers check)")
    fig.tight_layout()
    bcf_box_png = fig_dir / "qc_bcf_box.png"
    fig.savefig(bcf_box_png, dpi=320)
    plt.close(fig)

    # 4) Group stats (optional)
    usable_groups = [c for c in group_cols if c in df.columns]
    group_stats_xlsx: Optional[Path] = None
    if usable_groups:
        gstat = (
            df.assign(_bcf=pd.to_numeric(df[target_bcf], errors="coerce"))
              .groupby(usable_groups, dropna=False)["_bcf"]
              .agg(
                  n="count",
                  mean="mean",
                  median="median",
                  p95=lambda s: s.quantile(0.95),
                  p99=lambda s: s.quantile(0.99),
              )
              .reset_index()
              .sort_values("n", ascending=False)
        )
        group_stats_xlsx = qc_dir / "qc_group_stats.xlsx"
        gstat.to_excel(group_stats_xlsx, index=False)

    # 5) Notes (xy quick check)
    notes: List[Dict] = []
    if x_col in df.columns and y_col in df.columns:
        xy = df[[x_col, y_col]].dropna()
        notes.append({"item": "duplicated_xy_points", "value": int(xy.duplicated().sum())})
        notes.append({"item": f"{x_col}_min", "value": float(xy[x_col].min())})
        notes.append({"item": f"{x_col}_max", "value": float(xy[x_col].max())})
        notes.append({"item": f"{y_col}_min", "value": float(xy[y_col].min())})
        notes.append({"item": f"{y_col}_max", "value": float(xy[y_col].max())})
    else:
        notes.append({"item": "xy_check", "value": "X/Y not found -> skipped"})

    if ph_col in df.columns:
        ph = pd.to_numeric(df[ph_col], errors="coerce")
        notes.append({"item": "pH_non_null", "value": int(ph.notna().sum())})
        if ph.notna().any():
            notes.append({"item": "pH_min", "value": float(ph.min())})
            notes.append({"item": "pH_max", "value": float(ph.max())})

    notes_df = pd.DataFrame(notes)
    notes_xlsx = qc_dir / "qc_notes.xlsx"
    notes_df.to_excel(notes_xlsx, index=False)

    return QCPaths(
        qc_dir=qc_dir,
        fig_dir=fig_dir,
        missing_rate_xlsx=missing_rate_xlsx,
        numeric_desc_xlsx=numeric_desc_xlsx,
        bcf_quantiles_xlsx=bcf_quantiles_xlsx,
        group_stats_xlsx=group_stats_xlsx,
        notes_xlsx=notes_xlsx,
        bcf_hist_png=bcf_hist_png,
        bcf_box_png=bcf_box_png,
    )
