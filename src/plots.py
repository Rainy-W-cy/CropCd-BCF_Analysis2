from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PALETTE = {
    "bg": "#F9F7F1",
    "axis": "#1F2937",
    "line_ref": "#374151",
    "line_zero": "#4B5563",
    "scatter_resid": "#9CC9CD",
    "density_edge": "#0F172A",
    "hist_fill": "#EDC4A7",
    "hist_edge": "#FFFFFF",
}

PASTEL_CMAP = LinearSegmentedColormap.from_list(
    "soil_pastel",
    ["#E4A8AA", "#EDC4A7", "#F9F0E0", "#CCE7E3", "#80B5BA"],
)


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "axes.unicode_minus": False,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.edgecolor": PALETTE["axis"],
            "axes.labelcolor": PALETTE["axis"],
            "xtick.color": PALETTE["axis"],
            "ytick.color": PALETTE["axis"],
        }
    )


def hist_plot(values, title: str, out_png, bins: int = 30):
    """Histogram helper used by EDA notebooks."""
    _apply_paper_style()
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    out_png = Path(out_png)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.hist(
        x,
        bins=bins,
        color=PALETTE["hist_fill"],
        edgecolor=PALETTE["hist_edge"],
        linewidth=0.8,
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def obs_pred_density(
    y_true,
    y_pred,
    title: str,
    out_png: Path,
    *,
    target_label: str = "Target",
):
    """Paper-style Obs-Pred plot with density coloring and stronger contrast."""
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_pred >= 0)
    y_true = y_true[m]
    y_pred = y_pred[m]

    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)

    pts = np.vstack([y_true, y_pred])
    try:
        density = gaussian_kde(pts)(pts)
    except Exception:
        density = np.ones_like(y_true, dtype=float)

    order = np.argsort(density)
    x_plot = y_true[order]
    y_plot = y_pred[order]
    d_plot = density[order]

    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    sc = ax.scatter(
        x_plot,
        y_plot,
        c=d_plot,
        cmap=PASTEL_CMAP,
        s=34,
        alpha=0.95,
        edgecolors=PALETTE["density_edge"],
        linewidths=0.2,
    )

    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([mn, mx], [mn, mx], linewidth=2.3, color=PALETTE["line_ref"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    ax.set_xlabel(f"Observed {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.set_title(
        f"{title}\n"
        f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}",
        fontsize=12,
    )

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Point density", color=PALETTE["axis"])
    cb.ax.tick_params(labelsize=9, colors=PALETTE["axis"])

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


def residual_plot(
    y_true,
    y_pred,
    title: str,
    out_png: Path,
    *,
    target_label: str = "Target",
):
    """Paper-style residual plot with higher contrast."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_pred >= 0)
    y_true = y_true[m]
    y_pred = y_pred[m]
    resid = y_pred - y_true

    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.scatter(
        y_true,
        resid,
        s=24,
        alpha=0.95,
        color=PALETTE["scatter_resid"],
        edgecolors=PALETTE["density_edge"],
        linewidths=0.25,
    )
    ax.axhline(0.0, linewidth=2.1, color=PALETTE["line_zero"])

    ax.set_xlabel(f"Observed {target_label}")
    ax.set_ylabel(f"Residual (Predicted - Observed {target_label})")
    ax.set_title(title, fontsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


def obs_pred_scatter(y_true, y_pred, title: str, out_png: Path, *, target_label: str = "Target"):
    """Backward-compatible entrypoint."""
    obs_pred_density(
        y_true,
        y_pred,
        title=title,
        out_png=out_png,
        target_label=target_label,
    )
