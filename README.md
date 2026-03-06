# 土壤-作物Cd与BCF分析（按任务书技术路线）
## 虚拟环境创建
python -m venv .venv
## 运行环境安装
python -m pip install -r requirements.txt
或pip install -r requirements.txt
## 运行顺序（Jupyter）
notebooks/：
1) 01_data_prep.ipynb 数据整理
2) 02_qc_optional.ipynb 数据分析
3) 03_eda_feature.ipynb 也可不跑此03，相关性简单分析
4) 03b_corr_advanced.ipynb相关度分析
5) 04_modeling_spatialcv.ipynb 空间交叉验证
6) 04b_holdout_70_30.ipynb 7:3划分方法
7) 05_shap_new.ipynb shap图
8) 06_thresholds_gb.ipynb 阈值分析也可不做
9) 08_maps 空间分布地图
10) 07_reports_pdf.ipynb 输出为pdf

## 按 Cell 运行教程（推荐）
为避免重复运行旧版本单元，按下面执行：
1) `01_data_prep.ipynb`：运行 `code cell 1`（唯一主流程）
2) `02_qc_optional.ipynb`：顺序运行 `code cell 1 -> 2 -> 3`
3) `03_eda_feature.ipynb`：运行 `code cell 1`
4) `03b_corr_advanced.ipynb`：只运行 `code cell 2`（`03 Corr Advanced (FINAL)`）；`code cell 1` 可跳过
5) `04_modeling_spatialcv.ipynb`：运行 `code cell 1`
6) `04b_holdout_70_30.ipynb`：顺序运行 `code cell 0 -> 1 -> 2`；`code cell 3` 可跳过
7) `05_shap_new.ipynb`：运行 `code cell 1`（当前主用）
8) `05_shap.ipynb`：老版本流程，如需运行建议只用 `code cell 3`（论文风格输出）
9) `06_thresholds_gb.ipynb`：运行 `code cell 1`
10) `08_maps.ipynb`：运行 `code cell 1`
11) `07_reports_pdf.ipynb`：运行 `code cell 1`

## 最小复现路径（常用）
1) 先跑 `01_data_prep.ipynb`（生成 `outputs/clean_data.parquet`）
2) 再跑 `04b_holdout_70_30.ipynb`（7:3+holdout）
3) 然后按需要跑 `03b/05/06/08`
4) 最后跑 `07_reports_pdf.ipynb`

所有参数在 config/config.yaml。修改后重新跑对应 notebook 即可复现更新结果。
## 更改目标元素
1) 如果想更改目标元素比如预测作物cd，更改config.yaml中target_bcf变量为表格中相应变量即可
2) 相关性分析方面，应排除经纬度，同时prefer的元素可以自行设置
