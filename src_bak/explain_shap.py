from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

def compute_shap_for_pipeline(pipe: Pipeline, X: pd.DataFrame, max_rows: int = 800, random_state: int = 42) -> Tuple[np.ndarray, List[str]]:
    if len(X) > max_rows:
        Xs = X.sample(max_rows, random_state=random_state)
    else:
        Xs = X.copy()
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    Xt = pre.transform(Xs)
    feat_names = list(pre.get_feature_names_out())
    explainer = shap.Explainer(model, Xt, feature_names=feat_names)
    sv = explainer(Xt)
    return sv.values, feat_names

def save_shap_summary(shap_values: np.ndarray, feature_names: List[str], outpath: str) -> None:
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, features=None, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
