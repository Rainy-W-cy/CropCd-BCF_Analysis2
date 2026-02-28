from __future__ import annotations
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import matplotlib.pyplot as plt


def _as_dense_float_matrix(X) -> np.ndarray:
    """Convert X to dense float numpy array (supports numpy, pandas, scipy sparse)."""
    # scipy sparse
    try:
        import scipy.sparse as sp  # type: ignore
        if sp.issparse(X):
            X = X.toarray()
    except Exception:
        pass

    # pandas
    try:
        import pandas as pd  # type: ignore
        if isinstance(X, pd.DataFrame):
            # For plotting colors, encode non-numeric columns
            df = X.copy()
            for c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].astype("category").cat.codes.astype(float)
            return df.to_numpy(dtype=float)
    except Exception:
        pass

    # numpy
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be 2D")
    return arr.astype(float)


def shap_importance_combo(
    shap_values: np.ndarray,
    X,
    feature_names: Sequence[str],
    out_png: Path,
    title: str = "SHAP importance (bar + beeswarm)",
    max_display: int = 20,
    jitter: float = 0.22,
    dot_size: float = 18.0,
    alpha: float = 0.75,
):
    """
    Paper-style SHAP plot: mean|SHAP| bars + beeswarm in one figure.

    Requirements:
    - shap_values shape == X shape (n_samples, n_features)
    - X must be numeric or convertible to numeric (DataFrame allowed; non-numeric will be encoded for color only)
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    sv = np.asarray(shap_values, dtype=float)
    Xv = _as_dense_float_matrix(X)

    if sv.ndim != 2 or Xv.ndim != 2:
        raise ValueError("shap_values and X must be 2D arrays")
    if sv.shape != Xv.shape:
        raise ValueError(f"shape mismatch: shap_values {sv.shape} vs X {Xv.shape}")

    p = sv.shape[1]
    names = list(feature_names) if feature_names is not None else []
    if len(names) != p:
        names = [f"f{i}" for i in range(p)]

    mean_abs = np.nanmean(np.abs(sv), axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]

    mean_abs_ord = mean_abs[order]
    sv_ord = sv[:, order]
    X_ord = Xv[:, order]
    names_ord = [names[i] for i in order]

    # Normalize X for coloring per feature
    Xn = np.empty_like(X_ord)
    for j in range(X_ord.shape[1]):
        col = X_ord[:, j]
        lo = np.nanpercentile(col, 1)
        hi = np.nanpercentile(col, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            lo, hi = np.nanmin(col), np.nanmax(col)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            Xn[:, j] = 0.5
        else:
            Xn[:, j] = np.clip((col - lo) / (hi - lo), 0, 1)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    ax2 = ax.twiny()
    y = np.arange(len(names_ord))[::-1]

    ax2.barh(y, mean_abs_ord[::-1], height=0.72, alpha=0.45, edgecolor="black", linewidth=1.0)
    ax2.set_xlabel("Mean |SHAP|", fontweight="bold")
    ax2.tick_params(axis="x", labelsize=10)

    rng = np.random.default_rng(42)
    sc_last = None
    for row, y0 in enumerate(y):
        j = len(names_ord) - 1 - row
        vals = sv_ord[:, j]
        cols = Xn[:, j]
        msk = np.isfinite(vals) & np.isfinite(cols)
        vals = vals[msk]
        cols = cols[msk]
        if vals.size == 0:
            continue
        yj = y0 + rng.uniform(-jitter, jitter, size=vals.size)
        sc_last = ax.scatter(vals, yj, c=cols, s=dot_size, alpha=alpha, linewidths=0)

    ax.axvline(0.0, linewidth=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names_ord[::-1], fontweight="bold")
    ax.set_xlabel("SHAP Value", fontweight="bold")
    ax.set_ylabel("Features", fontweight="bold")
    ax.set_title(title, fontweight="bold")

    finite = sv_ord[np.isfinite(sv_ord)]
    if finite.size:
        q0, q1 = np.nanquantile(finite, [0.01, 0.99])
        pad = (q1 - q0) * 0.08 if q1 > q0 else 1.0
        ax.set_xlim(q0 - pad, q1 + pad)

    if sc_last is not None:
        cbar = fig.colorbar(sc_last, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label("Feature value (Low → High)", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)
