# 土壤-作物Cd与BCF分析（按任务书技术路线）
## 运行环境安装
pip install -r requirements.txt
## 运行顺序（Jupyter）
notebooks/：
1) 01_data_prep.ipynb 数据整理
2) 02_qc_optional.ipynb数据分析
3) 03_eda_feature.ipynb 也可不跑此03，相关性简单分析
4) 03b_corr_advanced.ipynb相关度分析
5) 04_modeling_spatialcv.ipynb 空间交叉验证
6) 04b_holdout_70_30.ipynb 7:3划分
7) 05_shap.ipynb 
8) 06_thresholds_gb.ipynb 阈值分析也可不做
9) 08_maps 分布地图
10) report

所有参数在 config/config.yaml。修改后重新跑对应 notebook 即可复现更新结果。
## 更改目标元素
1) 如果想更改目标元素比如预测作物cd，更改config.yaml中target_bcf变量为表格中相应变量即可
2) 相关性分析方面，应排除经纬度，同时prefer的元素可以自行设置