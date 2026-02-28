from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

@dataclass
class CorrConfig:
    method: str = "pearson"
    min_rows: int = 30
    max_vars: int = 25
    cluster: bool = True
    out_subdir: str = "corr_advanced"
    export_pdf: bool = True
    pdf_name: str = "corr_advanced_all.pdf"

def _get_cfg(cfg: Dict) -> CorrConfig:
    c = cfg.get("corr_advanced", {}) or {}
    return CorrConfig(
        method=str(c.get("method", "pearson")),
        min_rows=int(c.get("min_rows", 30)),
        max_vars=int(c.get("max_vars", 25)),
        cluster=bool(c.get("cluster", True)),
        out_subdir=str(c.get("out_subdir", "corr_advanced")),
        export_pdf=bool(c.get("export_pdf", True)),
        pdf_name=str(c.get("pdf_name", "corr_advanced_all.pdf")),
    )

def _select_numeric_vars(df: pd.DataFrame, max_vars: int, prefer_cols: Optional[List[str]] = None) -> List[str]:
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        return []
    miss = num.isna().mean(axis=0)
    var = num.var(axis=0, skipna=True)
    rank = pd.DataFrame({"col": miss.index, "miss": miss.values, "var": var.values})
    rank = rank.sort_values(["miss", "var"], ascending=[True, False])
    cols = rank["col"].tolist()
    if prefer_cols:
        front = [c for c in prefer_cols if c in cols]
        cols = front + [c for c in cols if c not in front]
    return cols[:max_vars]

def _cluster_order(corr: pd.DataFrame) -> List[str]:
    if corr.shape[0] <= 2 or not _HAS_SCIPY:
        return corr.index.tolist()
    M = corr.fillna(0).to_numpy()
    dist = 1 - np.abs(M)
    np.fill_diagonal(dist, 0.0)
    iu = np.triu_indices(dist.shape[0], k=1)
    condensed = dist[iu]
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    return corr.index.to_numpy()[order].tolist()

def _heuristic_order(cols: List[str]) -> List[str]:
    """Make a more 'paper-like' order without needing user to specify."""
    def pick(keys):
        out=[]
        for k in keys:
            for c in cols:
                if c.lower() == k.lower() and c not in out:
                    out.append(c)
        return out

    # common soil/agri variables (best-effort)
    head = pick(["bcf","bcf_calc","soil_cd","soil_cd_mgkg","crop_cd","crop_cd_mgkg","cd","ph","pH","som","cec"])
    texture = pick(["clay","clay_content","particle","particle_content","sand","sand_content","water","water_content","bulk","bulk_density"])
    nutrients = pick(["n","p","k","mn","zn","cu","fe"])
    geo = pick(["altitude","elevation","geology","soil_type","land_use","land_use_type","vegetation","vegetation_cover_type","distance","distance_from_pollution_source","x","y","lon","lat"])
    ordered = head + texture + nutrients + geo
    # append remaining
    for c in cols:
        if c not in ordered:
            ordered.append(c)
    return ordered

def _plot_triangle_annot(corr: pd.DataFrame, title: str, out_png: Path, annotate: bool = True):
    # lower triangle only
    n = corr.shape[0]
    M = corr.to_numpy().copy()
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    M_masked = M.copy()
    M_masked[mask] = np.nan

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(M_masked, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=13)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=9)

    # grid lines
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate and n <= 30:
        for i in range(n):
            for j in range(i+1):  # lower incl diag
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=280)
    plt.close(fig)

def build_group_correlations(
    df: pd.DataFrame,
    cfg: Dict,
    out_dir: str | Path,
    group_keys: Tuple[str, ...] = ("crop", "ph_bin"),
    prefer_cols: Optional[List[str]] = None,
) -> Dict[Tuple, pd.DataFrame]:
    conf = _get_cfg(cfg)
    out_dir = Path(out_dir) / conf.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_maps: Dict[Tuple, pd.DataFrame] = {}

    for g, sub in df.groupby(list(group_keys), dropna=False):
        sub = sub.copy()
        if len(sub) < conf.min_rows:
            continue

        cols = _select_numeric_vars(sub, conf.max_vars, prefer_cols=prefer_cols)
        if len(cols) < 2:
            continue

        # paper-like ordering first
        cols = _heuristic_order(cols)

        corr = sub[cols].corr(method=conf.method)

        # optional clustering to reveal blocks (still keeps it readable)
        if conf.cluster:
            order = _cluster_order(corr)
            corr = corr.loc[order, order]

        corr_maps[g if isinstance(g, tuple) else (g,)] = corr

        safe_g = "_".join([str(x) for x in (g if isinstance(g, tuple) else (g,))]).replace(" ", "")
        _plot_triangle_annot(
            corr,
            title=f"Correlation (lower triangle annotated) | group={g}",
            out_png=out_dir / f"corr_tri_{safe_g}.png",
            annotate=True,
        )

    return corr_maps

def _plot_triangle_compare_annot(
    corr_a: pd.DataFrame,
    corr_b: pd.DataFrame,
    title: str,
    out_png: Path,
):
    cols = [c for c in corr_a.columns if c in corr_b.columns]
    corr_a = corr_a.loc[cols, cols]
    corr_b = corr_b.loc[cols, cols]
    n = len(cols)
    M = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i):
            M[i, j] = corr_a.iloc[i, j]   # lower
        for j in range(i + 1, n):
            M[i, j] = corr_b.iloc[i, j]   # upper

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(M, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=9)

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    if n <= 30:
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation (lower=A, upper=B)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=280)
    plt.close(fig)

def export_triangle_comparisons(
    corr_maps: Dict[Tuple, pd.DataFrame],
    out_dir: str | Path,
    title_prefix: str,
    pairs: List[Tuple[Tuple, Tuple]],
    out_prefix: str,
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs: List[Path] = []

    for a_key, b_key in pairs:
        if a_key not in corr_maps or b_key not in corr_maps:
            continue
        corr_a = corr_maps[a_key]
        corr_b = corr_maps[b_key]
        safe = f"{out_prefix}_{'_'.join(map(str,a_key))}__VS__{'_'.join(map(str,b_key))}".replace(' ', '')
        out_png = out_dir / f"triangle_annot_{safe}.png"
        _plot_triangle_compare_annot(
            corr_a, corr_b,
            title=f"{title_prefix}\nlower={a_key}, upper={b_key}",
            out_png=out_png
        )
        outs.append(out_png)
    return outs

def export_corr_pdf(out_base: str | Path, cfg: Dict) -> Optional[Path]:
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
