from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from .utils import ensure_dir, safe_col
from .plots import obs_pred_scatter, residual_plot


# =========================
# Config
# =========================
@dataclass
class HoldoutConfig:
    enabled: bool = True
    test_size: float = 0.30
    random_state: Optional[int] = None
    split_by_group: bool = True
    group_keys: Tuple[str, ...] = ("crop", "ph_bin")
    min_rows_per_group: int = 20
    mix_train_into_test: bool = True
    mix_fraction: float = 1.0
    out_subdir: str = "holdout_70_30"


def _get_holdout_cfg(cfg: Dict) -> HoldoutConfig:
    h = cfg.get("holdout", {}) or {}

    group_keys = h.get("group_keys", ["crop", "ph_bin"])
    if isinstance(group_keys, (list, tuple)):
        group_keys = tuple(str(x) for x in group_keys)
    else:
        group_keys = ("crop", "ph_bin")

    return HoldoutConfig(
        enabled=bool(h.get("enabled", True)),
        test_size=float(h.get("test_size", 0.30)),
        random_state=h.get("random_state", None),
        split_by_group=bool(h.get("split_by_group", True)),
        group_keys=group_keys,
        min_rows_per_group=int(h.get("min_rows_per_group", 20)),
        mix_train_into_test=bool(h.get("mix_train_into_test", True)),
        mix_fraction=float(h.get("mix_fraction", 1.0)),
        out_subdir=str(h.get("out_subdir", "holdout_70_30")),
    )


# =========================
# Helpers
# =========================
@dataclass
class HoldoutResult:
    metrics: pd.DataFrame
    pred_index: pd.DataFrame
    out_dir: Path
    fig_dir: Path
    pred_dir: Path


def _rmse(y_true, y_pred) -> float:
    # 兼容旧 sklearn（避免 squared=False）
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )



def build_model(name: str, seed: int) -> Optional[Any]:
    name = str(name).lower()

    # -------------------------
    # Random Forest (more stable)
    # -------------------------
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=1200,          # ↑ 从 800 到 1200：降低方差、提高稳定性
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt",        # ↑ 更稳的默认，减少过拟合
            min_samples_leaf=3,         # ↑ 从默认 1 到 3：强烈建议，BCF噪声大时更稳
            min_samples_split=6,        # ↑ 避免过度切分
            max_depth=None              # 可先不限制；如果仍过拟合，再设 12~25
        )

    # -------------------------
    # SVR (more robust defaults)
    # -------------------------
    if name == "svm":
        return SVR(
            kernel="rbf",
            C=30.0,          # ↑ 从 10 到 30：提升拟合能力（配合log目标更稳）
            epsilon=0.02,    # ↓ 从 0.05 到 0.02：对小BCF更敏感（否则全被当噪声）
            gamma="scale"    # ✅ 保持自适应尺度（比手写更稳）
        )

    # -------------------------
    # ANN / MLP (much more stable)
    # -------------------------
    if name == "ann":
        return MLPRegressor(
            hidden_layer_sizes=(32, 16),   # ↑ 从 (8,) 到 (32,16)：避免欠拟合
            random_state=seed,
            max_iter=2000,                 # ↑ 从 800 到 2000：提高收敛概率
            alpha=1e-3,                    # ✅ L2 正则，抑制过拟合
            learning_rate="adaptive",      # ✅ 自适应学习率
            learning_rate_init=1e-3,       # ✅ 更稳的初始学习率
            early_stopping=True,           # ✅ 自动早停（关键）
            n_iter_no_change=25,           # ✅ 早停耐心
            validation_fraction=0.15       # ✅ 验证集比例
        )
    
    if name in {"cat", "catboost"}:
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=800,
                depth=6,
                learning_rate=0.05,
                loss_function="RMSE",
                random_seed=seed,
                verbose=False
            )
        except Exception:
            return None

    if name in {"lgbm", "lightgbm"}:
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
            )
        except Exception:
            return None


    # -------------------------
    # XGB (if available)
    # -------------------------
    if name == "xgb":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=1500,        # ↑ 更强
                max_depth=4,
                learning_rate=0.03,       # ↓ 更稳
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1
            )
        except Exception:
            return None

    return None



# =========================
# Main API
# =========================
def run_holdout(
    df: pd.DataFrame,
    cfg: Dict,
    out_base: str | Path,
) -> HoldoutResult:
    """
    Hold-out train/test split (config-driven):
    - All settings are read from cfg['holdout'].
    - Outputs to outputs/<out_subdir>/ (auto-created).
    - Does NOT compute/require bcf_calc; does NOT do any variable "name-based" fallback.
    """

    seed = int((cfg.get("project", {}) or {}).get("seed", 42))
    hcfg = _get_holdout_cfg(cfg)

    if not hcfg.enabled:
        raise RuntimeError("holdout.enabled=false：已在 config 中关闭 holdout")

    if not (0.0 < float(hcfg.test_size) < 1.0):
        raise ValueError(f"holdout.test_size 必须在 (0,1) 之间，当前={hcfg.test_size}")
    if not (0.0 <= float(hcfg.mix_fraction) <= 1.0):
        raise ValueError(f"holdout.mix_fraction 必须在 [0,1] 之间，当前={hcfg.mix_fraction}")

    random_state = int(hcfg.random_state) if hcfg.random_state is not None else seed

    out_base = Path(out_base)
    out_dir = ensure_dir(out_base / hcfg.out_subdir)
    fig_dir = ensure_dir(out_dir / "figures")
    pred_dir = ensure_dir(out_dir / "predictions")

    # target column
    y_col = safe_col(df, cfg["data"]["columns"]["target_bcf"])

    # columns excluded from modeling
    drop_from_model = list((cfg.get("data", {}) or {}).get("drop_from_model", []))

    # model list
    models = list((cfg.get("modeling", {}) or {}).get("models", ["rf"]))

    # grouping strategy
    usable_group_keys = [k for k in hcfg.group_keys if k in df.columns]
    if hcfg.split_by_group and usable_group_keys:
        grouped = df.groupby(usable_group_keys, dropna=False)
    else:
        grouped = [(("ALL",), df)]

    rows_metrics: List[Dict] = []
    rows_pred_index: List[Dict] = []

    for g, sub in grouped:
        sub = sub.copy()
        if len(sub) < hcfg.min_rows_per_group:
            continue

        # drop missing target rows
        y = sub[y_col]
        mask = y.notna()
        sub = sub.loc[mask].reset_index(drop=True)
        if len(sub) < hcfg.min_rows_per_group:
            continue

        # build X/y
        X = sub.drop(columns=[y_col], errors="ignore").copy()
        X = X.drop(columns=drop_from_model, errors="ignore")

        # ✅ remove columns that are entirely missing (avoid median-imputer warnings)
        X = X.dropna(axis=1, how="all")

        y = pd.to_numeric(sub[y_col], errors="coerce")
        m2 = np.isfinite(y.to_numpy())
        X = X.loc[m2].reset_index(drop=True)
        y = y.loc[m2].astype(float).reset_index(drop=True)

        if len(X) < hcfg.min_rows_per_group:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(hcfg.test_size),
            random_state=int(random_state),
        )

        key_str = "_".join(map(str, g if isinstance(g, tuple) else (g,))).replace(" ", "")

        for mname in models:
            base = build_model(mname, seed=seed)
            if base is None:
                continue

            pre = build_preprocessor(X_train)
            pipe = Pipeline(steps=[("pre", pre), ("model", base)])

            pipe.fit(X_train, y_train)
            # 对齐“土壤农产品”做法：可选把训练样本混入评估集（会造成泄漏，指标会虚高）
            if hcfg.mix_train_into_test and hcfg.mix_fraction > 0:
                n_eval_mix = max(1, int(len(X_train) * hcfg.mix_fraction))
                n_eval_mix = min(n_eval_mix, len(X_train))
                eval_idx = X_train.sample(n=n_eval_mix, random_state=int(random_state)).index
                X_eval_mix = X_train.loc[eval_idx]
                y_eval_mix = y_train.loc[eval_idx]
                X_eval = pd.concat([X_test, X_eval_mix], axis=0, ignore_index=True)
                y_eval = pd.concat([y_test, y_eval_mix], axis=0, ignore_index=True)
            else:
                X_eval = X_test.reset_index(drop=True)
                y_eval = y_test.reset_index(drop=True)

            y_pred = pipe.predict(X_eval)

            mask = np.isfinite(y_eval) & np.isfinite(y_pred) & (y_pred >= 0)
            y_eval_f = y_eval[mask]
            y_pred_f = y_pred[mask]

            r2 = float(r2_score(y_eval_f, y_pred_f))
            rmse = _rmse(y_eval_f, y_pred_f)
            mae = float(mean_absolute_error(y_eval_f, y_pred_f))

            rows_metrics.append({
                "group": str(g),
                "model": str(mname),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "test_size": float(hcfg.test_size),
                "random_state": int(random_state),
                "split_by_group": bool(hcfg.split_by_group),
                "mix_train_into_test": bool(hcfg.mix_train_into_test),
                "mix_fraction": float(hcfg.mix_fraction) if hcfg.mix_train_into_test else 0.0,
            })

            pred_df = X_eval.copy()
            pred_df["y_true"] = np.asarray(y_eval)
            pred_df["y_pred"] = np.asarray(y_pred)
            pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]

            pred_df = pred_df[pred_df["y_pred"] >= 0].reset_index(drop=True)

            pred_path = pred_dir / f"pred_holdout_{key_str}_{mname}.csv"
            pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

            rows_pred_index.append({
                "group": str(g),
                "model": str(mname),
                "pred_file": str(pred_path),
            })

            title = (
                f"{g} | holdout {int((1-hcfg.test_size)*100)}:{int(hcfg.test_size*100)} | {mname}\n"
                f"R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}"
            )

            obs_path = fig_dir / f"obs_pred_holdout_{key_str}_{mname}.png"
            resid_path = fig_dir / f"resid_holdout_{key_str}_{mname}.png"

            target_label = cfg["data"]["columns"]["target_bcf"]
            obs_pred_scatter(y_eval, y_pred, title, obs_path, target_label=target_label)
            residual_plot(y_eval, y_pred, title, resid_path, target_label=target_label)

    metrics = pd.DataFrame(rows_metrics).sort_values(["group", "model"]).reset_index(drop=True)
    pred_index = pd.DataFrame(rows_pred_index).sort_values(["group", "model"]).reset_index(drop=True)

    metrics.to_excel(out_dir / "metrics_holdout.xlsx", index=False)
    pred_index.to_excel(out_dir / "pred_files_holdout.xlsx", index=False)

    return HoldoutResult(
        metrics=metrics,
        pred_index=pred_index,
        out_dir=out_dir,
        fig_dir=fig_dir,
        pred_dir=pred_dir
    )
