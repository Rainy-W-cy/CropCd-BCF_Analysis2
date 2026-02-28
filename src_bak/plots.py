from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def obs_pred_density(y_true, y_pred, title: str, out_png: Path):
    """Density-style obs-pred plot (more readable than plain scatter)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    hb = ax.hexbin(y_true, y_pred, gridsize=35, mincnt=1)
    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([mn, mx], [mn, mx], linewidth=2)  # 1:1
    ax.set_xlabel("Observed BCF")
    ax.set_ylabel("Predicted BCF")
    ax.set_title(title, fontsize=12)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=260)
    plt.close(fig)

def residual_plot(y_true, y_pred, title: str, out_png: Path):
    """Residual plot (Pred-Obs) for diagnosing bias."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    resid = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(y_true, resid, s=14, alpha=0.75)
    ax.axhline(0.0, linewidth=2)
    ax.set_xlabel("Observed BCF")
    ax.set_ylabel("Residual (Pred - Obs)")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=260)
    plt.close(fig)

def obs_pred_scatter(y_true, y_pred, title: str, out_png: Path):
    """Backward-compatible entrypoint used by notebooks.

    We intentionally render density (hexbin) for readability.
    """
    obs_pred_density(y_true, y_pred, title=title, out_png=out_png)
