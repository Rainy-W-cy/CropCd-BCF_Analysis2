from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def hist_plot(values, title: str, out_png, bins: int = 30):
    """Histogram helper used by EDA notebooks."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    out_png = Path(out_png)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.hist(x, bins=bins, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=260)
    plt.close(fig)

def obs_pred_density(
    y_true,
    y_pred,
    title: str,
    out_png: Path,
    *,
    target_label: str = "Target"
):
    """Paper-style density Obs–Pred plot (scatter density)."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)& (y_pred >= 0)
    y_true = y_true[m]
    y_pred = y_pred[m]

    # metrics
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "axes.unicode_minus": False,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    sc = ax.scatter(
        y_true, y_pred,
        c=y_true,
        cmap="viridis",
        s=26,
        alpha=0.65,
        edgecolors="none"
    )

    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))

    ax.plot([mn, mx], [mn, mx], linewidth=2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    ax.set_xlabel(f"Observed {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")

    ax.set_title(
        f"{title}\n"
        f"R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}",
        fontsize=12
    )

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Sample density")

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)



def residual_plot(
    y_true,
    y_pred,
    title: str,
    out_png: Path,
    *,
    target_label: str = "Target"
):
    """Paper-style residual plot."""
    import numpy as np
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_pred >= 0)
    y_true = y_true[m]
    y_pred = y_pred[m]
    resid = y_pred - y_true

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "axes.unicode_minus": False,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(y_true, resid, s=18, alpha=0.7)
    ax.axhline(0.0, linewidth=2.0)

    ax.set_xlabel(f"Observed {target_label}")
    ax.set_ylabel(f"Residual (Predicted − Observed {target_label})")
    ax.set_title(title, fontsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

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
        target_label=target_label
    )

