---
name: soil-fig-style
description: 统一 soil_task_project 的图像配色、边框、网格线与导出规范；用于 SHAP、地图和模型对比图保持论文级一致风格。
---
# Soil Fig Style Skill

## 目的
统一 `soil_task_project` 的图片风格（配色、边框、网格线、字体与导出参数），保证论文级视觉一致性，可长期复用。

## 适用范围
- SHAP 总结图（bar + beeswarm）
- 模型对比柱状图（全局 + 分组）
- 相关性热图
- 点位地图（边界轮廓）

## 统一视觉基线
- 背景色：`#F9F7F1`
- 主字体：`serif`（`Times New Roman`, `DejaVu Serif`, `STIXGeneral`）
- 标题/坐标轴字重：`bold`
- 网格线：`linestyle='--'`, `alpha=0.20~0.30`, `color='#D9D7CF'`
- 图框边线（spines）：
  - 科研主图（SHAP/热图）：`linewidth=1.6~1.8`, `color='#111827'`
  - 柱状对比图：`linewidth=0.35~0.6`, `color='#7C7A73'`
- 导出：`bbox_inches='tight'`

## 配色规范

### A. SHAP 渐变色（连续）
用于 SHAP beeswarm 点颜色、密度类渐变：
- `['#A94442', '#E4A8AA', '#F9F0E0', '#CCE7E3', '#2F6C74']`

### B. 论文柔和离散色（模型对比柱状图）
用于模型对比（不同模型不同色）：
- `['#80B5BA', '#CCE7E3', '#EDC4A7', '#E4A8AA', '#B7A8C9', '#9FB6D9', '#BFD7C1', '#D6C5B2']`

说明：
- 这是从“土壤农产品”项目的 Pastel 系（`#E4A8AA/#EDC4A7/#F9F0E0/#CCE7E3/#80B5BA`）中和扩展得到的多色柔和方案。
- 不使用高饱和荧光色，不做纯灰化处理。

### C. 地图边界色
- 边界轮廓色：`#3D4B5A`
- 边界宽度：`1.8`（必要时 `2.0`）

## 结构化执行规则

### 1) SHAP 图
- 背景固定 `#F9F7F1`
- bar 使用 `#EDC4A7`（`alpha≈0.6`）
- beeswarm 点边框用深色（`#0F172A`）并保持细线（`linewidths≈0.2`）
- SHAP 0 轴线：`#374151`, `linewidth≈1.7`

### 2) 模型对比柱状图
- 使用“论文柔和离散色”并按模型名称稳定映射
- 仅用明度+色相区分，不要单色深浅梯度
- 同一模型在所有图中颜色保持一致
- 全局图、按组图、单组图使用同一映射

### 3) 热图/相关图
- 背景 `#F9F7F1`
- 图框 `#111827`, `1.8`
- 网格用白线或浅灰虚线，不抢占主体

### 4) 地图图
- 边界必须可见（读取失败即报错或日志明确）
- 点图色带可保持 `Spectral_r`，但边框色深、线宽细
- 坐标网格保留轻量虚线（`alpha≈0.3`）

## 代码模板（Matplotlib）
```python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BG = "#F9F7F1"
GRID = "#D9D7CF"
EDGE_SOFT = "#7C7A73"
EDGE_DARK = "#111827"

SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_soil",
    ["#A94442", "#E4A8AA", "#F9F0E0", "#CCE7E3", "#2F6C74"],
)

MODEL_PALETTE = [
    "#80B5BA", "#CCE7E3", "#EDC4A7", "#E4A8AA",
    "#B7A8C9", "#9FB6D9", "#BFD7C1", "#D6C5B2",
]

def model_color_map(models):
    models = [str(m) for m in models]
    return {m: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, m in enumerate(models)}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "axes.unicode_minus": False,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})
```

## 出图验收清单（必须通过）
- [ ] 背景是否统一为 `#F9F7F1`（或透明轴叠加时保持米白主底）
- [ ] 同一模型在不同图是否同色
- [ ] 网格线是否“可读但不抢眼”
- [ ] 边框线粗细是否统一（主图 1.6~1.8；柱图 0.35~0.6）
- [ ] 色彩是否柔和（无高饱和刺眼色）
- [ ] 导出分辨率是否满足论文（建议 240~340 dpi）

## 项目落地约定
- 新增任何绘图模块时，优先复用本 Skill 的色板与参数。
- 若新增库/依赖（如专用配色工具），同步更新 `requirements.txt`。

