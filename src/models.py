from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype, is_categorical_dtype

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from .utils import safe_col, set_seed

@dataclass
class ModelResult:
    name: str
    best_estimator: Any
    oof_pred: np.ndarray
    y_true: np.ndarray
    metrics: Dict[str, float]
    best_params: Dict[str, Any]

def _split_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    drop_cols = set(cfg["data"].get("drop_from_model", []))
    target = safe_col(df, cfg["data"]["columns"]["target_bcf"])
    usable = [c for c in df.columns if c != target and c not in drop_cols]
    for c in ["source_sheet","ph_bin_from_sheet","lod_used","bcf_calc"]:
        if c in usable: usable.remove(c)
    cat = [c for c in usable if is_object_dtype(df[c]) or is_string_dtype(df[c]) or is_categorical_dtype(df[c])]
    num = [c for c in usable if is_numeric_dtype(df[c]) and c not in cat]
    # Fallback: anything not numeric becomes categorical
    for c in usable:
        if c not in num and c not in cat:
            cat.append(c)
    return num, cat

def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
                             remainder="drop", verbose_feature_names_out=False)

def _get_model_and_space(name: str, seed: int):
    if name == "rf":
        return (RandomForestRegressor(random_state=seed, n_estimators=600, n_jobs=-1),
                {"model__max_depth":[None,6,10,14,20],
                 "model__min_samples_split":[2,4,8,12],
                 "model__min_samples_leaf":[1,2,4,6],
                 "model__max_features":["sqrt",0.6,0.8,1.0],
                 "model__n_estimators":[400,600,800]})
    if name == "xgb":
        base = XGBRegressor(random_state=seed, objective="reg:squarederror", n_jobs=-1)
        return (base, {"model__n_estimators":[400,600,800,1200],
                       "model__max_depth":[3,4,6,8,10],
                       "model__learning_rate":[0.01,0.03,0.05,0.1],
                       "model__subsample":[0.6,0.8,1.0],
                       "model__colsample_bytree":[0.6,0.8,1.0],
                       "model__reg_lambda":[0.1,1.0,10.0]})
    if name == "svm":
        return (SVR(), {"model__C":[0.5,1,2,5,10,20],
                        "model__epsilon":[0.01,0.05,0.1,0.2],
                        "model__gamma":["scale","auto"]})
    if name == "ann":
        # More stable convergence with moderate speed; early stopping still limits runtime.
        base = MLPRegressor(
            random_state=seed,
            max_iter=5000,
            early_stopping=True,
            n_iter_no_change=20,
            tol=1e-4,
            validation_fraction=0.1,
            learning_rate_init=1e-3,
            alpha=1e-4,
        )
        return (base, {"model__hidden_layer_sizes":[(64,32),(128,64),(128,128)],
                       "model__alpha":[1e-5,1e-4,1e-3],
                       "model__learning_rate_init":[5e-4,1e-3,2e-3]})
    raise ValueError("Unknown model")

def fit_oof_with_spatialcv(df: pd.DataFrame, cfg: Dict[str, Any], groups: np.ndarray, model_name: str, cv) -> ModelResult:
    seed = int(cfg["project"]["seed"]); set_seed(seed)
    target = safe_col(df, cfg["data"]["columns"]["target_bcf"])
    y = pd.to_numeric(df[target], errors="coerce").to_numpy()
    mask = ~np.isnan(y) & (groups >= 0)
    df2 = df.loc[mask].reset_index(drop=True)
    y2 = y[mask]; g2 = groups[mask]
    num_cols, cat_cols = _split_columns(df2, cfg)
    pre = _make_preprocessor(num_cols, cat_cols)
    model, space = _get_model_and_space(model_name, seed)
    pipe = Pipeline([("pre", pre), ("model", model)])

    tune = bool(cfg["modeling"]["tune"])
    if tune:
        search = RandomizedSearchCV(pipe, space, n_iter=int(cfg["modeling"]["tune_n_iter"]),
                                    scoring="neg_root_mean_squared_error",
                                    cv=cv.split(df2, y2, groups=g2),
                                    random_state=seed, n_jobs=-1)
        search.fit(df2, y2)
        best = search.best_estimator_
        best_params = dict(search.best_params_)
    else:
        best = pipe.fit(df2, y2)
        best_params = {}

    oof = np.full(len(df2), np.nan)
    for tr, te in cv.split(df2, y2, groups=g2):
        est = best
        est.fit(df2.iloc[tr], y2[tr])
        oof[te] = est.predict(df2.iloc[te])

    # Compatible with older sklearn without squared= parameter
    rmse = float(np.sqrt(mean_squared_error(y2, oof)))
    mae = float(mean_absolute_error(y2, oof))
    r2 = float(r2_score(y2, oof))
    return ModelResult(model_name, best, oof, y2, {"rmse":rmse, "mae":mae, "r2":r2}, best_params)
