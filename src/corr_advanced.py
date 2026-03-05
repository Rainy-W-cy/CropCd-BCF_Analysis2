from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap


# =========================
# Config
# =========================
@dataclass
class CorrConfig:
    method: str = "pearson"
    min_rows: int = 30
    max_vars: int = 25
    cluster: bool = False  # reserved; keep order stable
    out_subdir: str = "corr_advanced"
    export_pdf: bool = True
    pdf_name: str = "corr_advanced_all.pdf"
    export_full: bool = True
    export_triangle: bool = False  # reserved

    # config-driven column control
    exclude_cols: List[str] = None
    prefer_cols: List[str] = None


def _get_cfg(cfg: Dict) -> CorrConfig:
    c = cfg.get("corr_advanced", {}) or {}
    exclude_cols = list(c.get("exclude_cols", []) or [])
    prefer_cols = list(c.get("prefer_cols", []) or [])

    return CorrConfig(
        method=str(c.get("method", "pearson")),
        min_rows=int(c.get("min_rows", 30)),
        max_vars=int(c.get("max_vars", 25)),
        cluster=bool(c.get("cluster", False)),
        out_subdir=str(c.get("out_subdir", "corr_advanced")),
        export_pdf=bool(c.get("export_pdf", True)),
        pdf_name=str(c.get("pdf_name", "corr_advanced_all.pdf")),
        export_full=bool(c.get("export_full", True)),
        export_triangle=bool(c.get("export_triangle", False)),
        exclude_cols=exclude_cols,
        prefer_cols=prefer_cols,
    )


# =========================
# Variable selection (data-driven)
# =========================
def _resolve_default_excludes_from_cfg(cfg: Dict) -> List[str]:
    """Default excludes: x/y coordinates from cfg."""
    cols_cfg = (cfg.get("data", {}) or {}).get("columns", {}) or {}
    xcol = cols_cfg.get("x")
    ycol = cols_cfg.get("y")
    out = []
    if xcol:
        out.append(str(xcol))
    if ycol:
        out.append(str(ycol))
    return out


def _select_numeric_vars(
    df: pd.DataFrame,
    max_vars: int,
    *,
    exclude_cols: Optional[List[str]] = None,
    prefer_cols: Optional[List[str]] = None,
) -> List[str]:
    """Select numeric columns from existing dataframe columns only."""
    exclude_cols = exclude_cols or []
    prefer_cols = prefer_cols or []

    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        return []

    drop = [c for c in exclude_cols if c in num.columns]
    if drop:
        num = num.drop(columns=drop, errors="ignore")
    if num.shape[1] == 0:
        return []

    miss = num.isna().mean(axis=0)
    var = num.var(axis=0, skipna=True)

    rank = pd.DataFrame({"col": miss.index, "miss": miss.values, "var": var.values})
    rank = rank.sort_values(["miss", "var"], ascending=[True, False])
    cols = rank["col"].tolist()

    cols = [c for c in cols if str(c).lower() != "bcf_calc"]

    if prefer_cols:
        front = [c for c in prefer_cols if c in cols]
        cols = front + [c for c in cols if c not in front]

    return cols[:max_vars]


def _heuristic_order_soil_science(cols: List[str]) -> List[str]:
    """Heuristic soil-science order."""

    def match_any(candidates: List[str]) -> List[str]:
        out: List[str] = []
        for cand in candidates:
            cand_l = cand.lower()
            for c in cols:
                if c.lower() == cand_l and c not in out:
                    out.append(c)
            for c in cols:
                if cand_l in c.lower() and c not in out:
                    out.append(c)
        return out

    block1 = match_any([
        "BCF", "bcf",
        "soil_cd", "soil_cd_mgkg", "cd_soil", "soil cd", "soilcd",
        "crop_cd", "crop_cd_mgkg", "cd_crop", "crop cd", "cropcd",
        "Cd", "cd",
    ])

    block2 = match_any([
        "pH", "ph",
        "SOM", "som", "SOC", "soc", "organic", "organic_matter",
        "CEC", "cec",
    ])

    block3 = match_any([
        "clay", "clay_content",
        "particle", "particle_content", "silt",
        "sand", "sand_content",
        "water", "water_content", "moisture",
        "bulk_density", "bulk", "density",
    ])

    block4 = match_any([
        "N", "n", "TN", "tn", "total_n", "nitrogen",
        "P", "p", "TP", "tp", "total_p", "phosphorus",
        "K", "k", "TK", "tk", "total_k", "potassium",
        "Mn", "mn", "Zn", "zn", "Cu", "cu", "Fe", "fe", "Mg", "mg", "Ca", "ca",
    ])

    block5 = match_any([
        "altitude", "elevation",
        "geology",
        "soil_type", "soil type",
        "land_use", "land use", "land_use_type",
        "vegetation", "vegetation_cover_type",
        "distance", "distance_from_pollution_source",
    ])

    ordered: List[str] = []
    for block in (block1, block2, block3, block4, block5):
        for c in block:
            if c in cols and c not in ordered:
                ordered.append(c)

    for c in cols:
        if c not in ordered:
            ordered.append(c)

    return ordered


# =========================
# Plotting
# =========================
def _set_paper_fonts():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })


def _plot_corr_full_annot_paper(
    corr: pd.DataFrame,
    title: str,
    out_png: Path,
    *,
    annotate: bool = True,
    fmt: str = "{:+.2f}",
    fontsize_title: int = 12,
    fontsize_tick_x: int = 9,
    fontsize_tick_y: int = 10,
    fontsize_annot: int = 8,
):
    _set_paper_fonts()

    corr = corr.copy()
    cols = list(corr.columns)
    corr = corr.loc[cols, cols]
    n = corr.shape[0]
    M = corr.to_numpy(dtype=float)

    fig_w = max(12, 0.55 * n)
    fig_h = max(10, 0.48 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#F9F7F1")
    ax.set_facecolor("#F9F7F1")

    cmap = LinearSegmentedColormap.from_list(
        "corr_soil",
        ["#A94442", "#E4A8AA", "#F9F0E0", "#CCE7E3", "#2F6C74"],
    )
    im = ax.imshow(M, vmin=-1, vmax=1, cmap=cmap)
    ax.set_title(title, fontsize=fontsize_title, pad=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=90, fontsize=fontsize_tick_x, fontweight="bold")
    ax.set_yticklabels(cols, fontsize=fontsize_tick_y, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=0.75, color="white")
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("#111827")

    if annotate and n <= 30:
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                if np.isfinite(v):
                    tcolor = "white" if abs(v) >= 0.55 else "#111827"
                    ax.text(
                        j,
                        i,
                        fmt.format(v),
                        ha="center",
                        va="center",
                        fontsize=fontsize_annot,
                        fontweight="bold",
                        color=tcolor,
                    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


# =========================
# Public API
# =========================
def build_group_correlations(
    df: pd.DataFrame,
    cfg: Dict,
    out_dir: str | Path,
    group_keys: Tuple[str, ...] = ("crop", "ph_bin"),
    prefer_cols: Optional[List[str]] = None,
) -> Dict[Tuple, pd.DataFrame]:
    """
    Build correlation matrices per group and export PNGs.
    Variable selection is DATA-DRIVEN.
    """
    conf = _get_cfg(cfg)

    out_dir = Path(out_dir) / conf.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    default_excludes = _resolve_default_excludes_from_cfg(cfg)
    exclude_cols = []
    for c in (default_excludes + (conf.exclude_cols or [])):
        if c not in exclude_cols:
            exclude_cols.append(c)

    prefer_cols_final: List[str] = []
    for c in ((conf.prefer_cols or []) + (prefer_cols or [])):
        if c not in prefer_cols_final:
            prefer_cols_final.append(c)

    usable_group_keys = [k for k in group_keys if k in df.columns]
    grouped = df.groupby(usable_group_keys, dropna=False) if usable_group_keys else [(("ALL",), df)]

    corr_maps: Dict[Tuple, pd.DataFrame] = {}

    for g, sub in grouped:
        sub = sub.copy()
        if len(sub) < conf.min_rows:
            continue

        cols = _select_numeric_vars(
            sub,
            conf.max_vars,
            exclude_cols=exclude_cols,
            prefer_cols=prefer_cols_final,
        )
        if len(cols) < 2:
            continue

        cols = _heuristic_order_soil_science(cols)

        corr = sub[cols].corr(method=conf.method)
        corr = corr.loc[cols, cols]

        g_key = g if isinstance(g, tuple) else (g,)
        corr_maps[g_key] = corr

        safe_g = "_".join([str(x) for x in g_key]).replace(" ", "")
        if conf.export_full:
            _plot_corr_full_annot_paper(
                corr,
                title=f"Method: {conf.method} | group={g_key}",
                out_png=out_dir / f"corr_full_{safe_g}.png",
                annotate=True,
            )

    return corr_maps


def export_corr_pdf(out_base: str | Path, cfg: Dict) -> Optional[Path]:
    """Export all correlation PNGs under corr_advanced directory into a single PDF."""
    conf = _get_cfg(cfg)
    if not conf.export_pdf:
        return None

    out_base = Path(out_base)
    out_dir = out_base / conf.out_subdir
    pdf_path = out_dir / conf.pdf_name

    pngs = sorted(out_dir.glob("*.png"))
    if not pngs:
        return None

    with PdfPages(pdf_path) as pdf:
        for p in pngs:
            img = plt.imread(str(p))
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(p.stem[:90], fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path
