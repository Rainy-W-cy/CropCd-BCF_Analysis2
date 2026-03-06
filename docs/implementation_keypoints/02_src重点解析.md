# src 重点解析（函数级）

## 1. `src/utils.py`

- `load_config`：读取 `config/config.yaml`。
- `ensure_dir`：统一创建输出目录。
- `safe_col`：大小写容错找列名，找不到会抛错。

## 2. `src/data_io.py`

- `read_excel_as_long`
  - `mode=table`：单表读取。
  - `mode=sheets`：多 sheet 合并，并附加 `source_sheet`。
  - 根据 sheet 名尝试推断 `crop` 与 `ph_bin_from_sheet`。
- `add_ph_bins`
  - 用 `ph_bins.edges/labels` 做 `pd.cut` 生成 `ph_bin`。

## 3. `src/audit.py`

- `enforce_required_columns` 强依赖 `target_bcf/ph/x/y` 列存在。
- `data_audit` 输出缺失率、唯一值数、数据类型概览。

## 4. `src/spatial_cv.py`

- `make_spatial_groups`
  - `method=grid`：对 X/Y 做分位数分箱，再组合成组号。
  - `method=kmeans`：对坐标聚类得到空间组。
- `get_group_kfold`：返回 `GroupKFold(n_splits)`。

## 5. `src/models.py`（空间CV建模主干）

- `_split_columns`：
  - 数值/类别列自动拆分。
  - 显式剔除 `source_sheet/ph_bin_from_sheet/lod_used/bcf_calc`。
- `_make_preprocessor`：
  - 数值：中位数填补 + 标准化。
  - 类别：众数填补 + OneHot。
- `_get_model_and_space`：
  - 支持 `rf/xgb/svm/ann/catboost/lightgbm`。
  - 返回“模型 + 随机搜索空间”。
- `fit_oof_with_spatialcv`：
  - 若 `modeling.tune=true`，用 `RandomizedSearchCV`（评分：`neg_root_mean_squared_error`）。
  - 再按空间 CV 做 OOF 预测，输出 `rmse/mae/r2`。

## 6. `src/holdout.py`（7:3评估主干）

- `build_model`：提供 `rf/svm/ann/catboost/lightgbm/xgb` 的固定参数版本。
- `run_holdout`：
  - 支持按组切分（`split_by_group + group_keys`）。
  - 每组分别 train/test，再对每个模型训练和评估。
  - 输出 `metrics_holdout.xlsx`、预测明细 CSV、观测-预测图与残差图。
- 重难点：`mix_train_into_test`
  - 若为 `true`，会抽取训练样本混入评估集（代码注释已说明会造成泄漏）。
  - 这是导致指标看起来偏高的关键开关。

## 7. `src/shap_plots.py`

- `shap_importance_combo`：将 mean|SHAP| 柱图 + beeswarm 合并到一张图。
- 内置论文风格：米白背景、柔和渐变、深色边框、零轴线。
- 强校验：`shap_values` 与 `X` 形状必须一致，否则直接报错。

## 8. `src/explain_shap.py`

- 提供通用 SHAP 封装（`shap.Explainer` + summary_plot）。
- 当前主流程 notebook 实际主要使用的是 `05_shap_new.ipynb` 内自定义逻辑 + `src/shap_plots.py`。

## 9. `src/maps.py`

- `make_maps`：地图总入口。
- `_draw_boundary` 边界绘制优先级：
  1) `geopandas` 读本地 `boundary_path`
  2) `geopandas` 在线 adcode 边界
  3) `pyshp` 读本地 shp
- 轮廓控制参数：
  - `boundary_color` 默认 `#3D4B5A`
  - `boundary_width` 默认 `1.8`
- 重点：若依赖缺失或边界文件不可读，会跳过边界并打印原因。

## 10. `src/thresholds.py`

- `infer_soil_cd_thresholds`：对每个样本做二分法求解土壤 Cd 阈值。
- 求解目标：令 `soil_cd * predicted_BCF = 作物限量`。
- 风险点：依赖 `cfg['data']['columns']['soil_cd']`，配置缺失时会报错。

## 11. `src/reporting.py`

- `export_oneclick_pdf`：将表格和图片按页面拼接进一个 PDF。
- 会扫描 `outputs` 下关键产物（含 maps、spatialcv、corr、threshold、shap）。
