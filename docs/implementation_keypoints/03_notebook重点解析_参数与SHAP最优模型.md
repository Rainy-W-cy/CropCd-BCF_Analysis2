# notebooks 重点解析、参数调节与 SHAP 最优模型规则

## 1. notebook 与 src 对接关系（当前代码）

- `04_modeling_spatialcv.ipynb` -> `src.spatial_cv` + `src.models`
- `04b_holdout_70_30.ipynb` -> `src.holdout.run_holdout`
- `05_shap_new.ipynb` -> `src.holdout.build_model/build_preprocessor` + `src.shap_plots.shap_importance_combo`
- `08_maps.ipynb` -> `import src.maps as maps_mod`，即直接对接 `maps.py`

## 2. SHAP 如何选择“最优模型”（严格按现有代码）

在 `05_shap_new.ipynb`，流程是：

1. 读取 `outputs/<holdout.out_subdir>/metrics_holdout.xlsx`。
2. 候选模型来自 `config.yaml -> modeling.models` 与允许集合交集。
3. 对每个模型按 `n_test` 做加权聚合：
   - `r2_weighted = average(r2, weights=n_test)`
   - `rmse_weighted = average(rmse, weights=n_test)`
   - `mae_weighted = average(mae, weights=n_test)`
4. 排序规则：
   - 先按 `r2_weighted` 降序
   - 再按 `rmse_weighted` 升序
5. 取第一名作为唯一全局模型 `m`，后续所有组统一用这个 `m` 做 SHAP。

说明：`mae_weighted` 会计算并用于展示，但当前排序不把它作为主排序键。

## 3. SHAP 解释器自动选择逻辑

`05_shap_new.ipynb` 中 `_compute_shap_values`：

- 若模型属于树模型集合：`xgb/rf/catboost/lightgbm` -> `shap.TreeExplainer`。
- 否则走通用路径：
  - 抽样背景集（最多 120）
  - `shap.Explainer(est.predict, background)`。

## 4. 为什么 SHAP 里会出现 `source_sheet`、`crop_...` 等字段（以及现在如何避免）

当前 notebook 已做两层过滤：

1. 原始列过滤：
   - 默认排除 `source_sheet/sheet_name/ph_bin_from_sheet/lod_used/bcf_calc/crop/ph_bin/id...`
   - 还可叠加 `config.yaml -> shap.exclude_raw_cols`。
2. 编码后列名二次过滤：
   - 过滤前缀 `crop_`, `ph_bin_`, `source_sheet`, `group_`
   - 过滤包含 `_id/sample_id/source_sheet/...` 的特征名。

因此当前版本是“先删原始元数据列，再删 one-hot 残留列名”。

## 5. 参数调节入口（以 config.yaml 为准）

1. 空间 CV
   - `spatial_cv.n_splits`
   - `spatial_cv.method`（`grid` 或 `kmeans`）
   - `spatial_cv.grid.n_bins_x/n_bins_y`
   - `spatial_cv.kmeans.n_clusters`

2. 建模
   - `modeling.models`
   - `modeling.tune`
   - `modeling.tune_n_iter`
   - `modeling.max_train_rows`

3. Holdout
   - `holdout.test_size`（7:3 对应 `0.30`）
   - `holdout.split_by_group`
   - `holdout.group_keys`
   - `holdout.min_rows_per_group`
   - `holdout.mix_train_into_test`
   - `holdout.mix_fraction`

4. SHAP
   - `shap.sample_rows`
   - `shap.top_k`
   - `shap.max_display`（notebook 中读取，config 里可补）
   - `shap.exclude_raw_cols`（notebook 支持）

5. 地图
   - `map_output.lon_col/lat_col`
   - `map_output.boundary_path`
   - `map_output.draw_boundary`
   - `map_output.boundary_color/boundary_width`
   - `map_output.out_subdir`

## 6. 两个容易误判结果的点

1. Holdout 泄漏设置
   - `mix_train_into_test=true` 会导致评估集掺入训练样本，指标偏高。

2. 06 阈值反推的模型来源
   - 当前不是读取 SHAP 选出的全局最优模型。
   - 当前逻辑是优先 `xgb`，否则 `rf`。

## 7. 建议的日常执行顺序

1. 先跑 `04b` 产出 holdout 指标。
2. 再跑 `05` 做全局最优模型选择和 SHAP。
3. 然后跑 `08` 出带轮廓地图。
4. 最后跑 `07` 汇总 PDF。
