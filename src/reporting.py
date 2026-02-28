from __future__ import annotations
from typing import List
from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def _read_image(path: Path):
    import matplotlib.image as mpimg
    return mpimg.imread(str(path))

def export_oneclick_pdf(outputs_dir: str | Path, pdf_name: str = "oneclick_report.pdf") -> Path:
    """One-click export: bundle key figures/tables under outputs/ into a single PDF."""
    outdir = Path(outputs_dir)
    repdir = outdir / "reports"
    repdir.mkdir(parents=True, exist_ok=True)
    pdf_path = repdir / pdf_name

    def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, max_rows: int = 35):
        show = df.head(max_rows)
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title(title, fontsize=14)
        tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def add_image_page(pdf: PdfPages, title: str, img_path: Path):
        img = _read_image(img_path)
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def add_grid_images(pdf: PdfPages, title: str, paths: List[Path], ncols: int = 2):
        n = len(paths)
        if n == 0:
            return
        ncols = max(1, ncols)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.69, 8.27))
        axes = np.array(axes).reshape(-1)
        for i, ax in enumerate(axes):
            ax.axis("off")
            if i < n:
                img = _read_image(paths[i])
                ax.imshow(img)
                ax.set_title(paths[i].stem[:60], fontsize=7)
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(pdf_path) as pdf:
        da = outdir / "data_audit.xlsx"
        if da.exists():
            add_table_page(pdf, "Data Audit (missingness & dtypes)", pd.read_excel(da))

        eda = outdir / "eda_tables.xlsx"
        if eda.exists():
            add_table_page(pdf, "EDA Descriptive Stats (grouped)", pd.read_excel(eda))

        corr = outdir / "figures" / "corr_heatmap_numeric.png"
        # Advanced correlation (original-style), if exists
        adv_dir = outdir / "corr_advanced"
        adv_pngs = sorted(adv_dir.glob("*.png"))
        for i in range(0, len(adv_pngs), 4):
            add_grid_images(pdf, "Advanced Correlation (by group / triangle)", adv_pngs[i:i+4], ncols=2)


        if corr.exists():
            add_image_page(pdf, "Correlation Heatmap (numeric)", corr)

        met = outdir / "metrics_spatialcv.xlsx"
        if met.exists():
            mdf = pd.read_excel(met)
            add_table_page(pdf, "Spatial-CV Metrics (R2/RMSE/MAE)", mdf, max_rows=40)
            for metric in ["r2", "rmse", "mae"]:
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                piv = mdf.pivot_table(index="group", columns="model", values=metric, aggfunc="mean")
                piv.plot(kind="bar", ax=ax)
                ax.set_title(f"Spatial-CV {metric.upper()} by group & model")
                ax.set_xlabel("group")
                ax.set_ylabel(metric)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        thr_sum = outdir / "thresholds_summary_quantiles.xlsx"
        if thr_sum.exists():
            ts = pd.read_excel(thr_sum, index_col=0).reset_index()
            add_table_page(pdf, "Soil Cd Threshold Quantiles (by crop & pH bin)", ts, max_rows=60)

        figdir = outdir / "figures"
        obs_imgs = sorted(figdir.glob("obs_pred_*.png"))
        resid_imgs = sorted(figdir.glob("resid_*.png"))
        for i in range(0, len(obs_imgs), 4):
            add_grid_images(pdf, "Observed vs Predicted (OOF, Spatial-CV) [density]", obs_imgs[i:i+4], ncols=2)
        for i in range(0, len(resid_imgs), 4):
            add_grid_images(pdf, "Residual plots (Pred-Obs)", resid_imgs[i:i+4], ncols=2)

                # Maps (lat/lon point maps), if exists
        maps_dir = outdir / "maps"
        maps_pngs = sorted(maps_dir.glob("*.png"))
        for i in range(0, len(maps_pngs), 4):
            add_grid_images(pdf, "Maps (lat/lon point maps)", maps_pngs[i:i+4], ncols=2)

        shapdir = outdir / "shap"
        shap_imgs = sorted(shapdir.glob("shap_summary_*.png"))
        for i in range(0, len(shap_imgs), 4):
            add_grid_images(pdf, "SHAP Summary (by group)", shap_imgs[i:i+4], ncols=2)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title("Report generated", fontsize=16)
        ax.text(0.02, 0.9, f"Outputs dir: {outdir}", fontsize=10)
        ax.text(0.02, 0.85, f"PDF: {pdf_path.name}", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

    return pdf_path
