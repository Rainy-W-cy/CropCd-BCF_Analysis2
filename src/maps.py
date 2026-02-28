from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# ================= 配置区域 =================
DEFAULT_ADCODE = "520526"  # 默认威宁县
# 阿里云 DataV 接口: 优先尝试 _full (含子区域), 失败则退回 .json (仅轮廓)
GEOJSON_URL_FULL = "https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json"
GEOJSON_URL_SIMPLE = "https://geo.datav.aliyun.com/areas_v3/bound/{adcode}.json"

@dataclass
class MapCfg:
    lon_col: str = "lon"
    lat_col: str = "lat"
    out_subdir: str = "maps"
    vars_to_map: Optional[List[str]] = None
    by_group: bool = True
    s: float = 20.0        # 点的大小
    alpha: float = 0.9     # 点的透明度
    clip_q_low: float = 0.01
    clip_q_high: float = 0.99
    draw_boundary: bool = True  
    adcode: str = DEFAULT_ADCODE 
    boundary_color: str = "#333333" 
    boundary_width: float = 1.5     

def _get(cfg: Dict) -> MapCfg:
    m = cfg.get("map_output", {}) or {}
    cols = cfg.get("data", {}).get("columns", {}) if isinstance(cfg.get("data", {}), dict) else {}
    return MapCfg(
        lon_col=str(m.get("lon_col", cols.get("lon", "lon"))),
        lat_col=str(m.get("lat_col", cols.get("lat", "lat"))),
        out_subdir=str(m.get("out_subdir", "maps")),
        vars_to_map=m.get("vars_to_map", None),
        by_group=bool(m.get("by_group", True)),
        s=float(m.get("s", 20.0)),
        adcode=str(m.get("adcode", DEFAULT_ADCODE))
    )

def load_online_boundary(adcode: str) -> Optional[gpd.GeoDataFrame]:
    """
    智能加载地图轮廓：优先加载含子区域的 _full 版本，失败则降级加载轮廓版
    """
    url_full = GEOJSON_URL_FULL.format(adcode=adcode)
    url_simple = GEOJSON_URL_SIMPLE.format(adcode=adcode)
    
    try:
        # print(f"正在尝试下载详细轮廓: {url_full} ...")
        gdf = gpd.read_file(url_full)
        return gdf
    except Exception:
        try:
            print(f"⚠️ 详细轮廓下载失败 (404)，尝试基础轮廓: {url_simple} ...")
            gdf = gpd.read_file(url_simple)
            return gdf
        except Exception as e:
            print(f"⚠️ 无法加载任何地图轮廓 (Adcode: {adcode}): {e}")
            return None

def plot_point_map(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    val_col: str,
    title: str,
    out_path: Path,
    mc: MapCfg
):
    """
    绘制地图核心函数
    """
    # 1. 数据清洗与类型强制转换 (关键修复)
    df_plot = df.copy()
    
    # [关键修复] 强制转为数值型，无法转换的变为 NaN (例如 '酸性' 这种文本)
    df_plot[val_col] = pd.to_numeric(df_plot[val_col], errors='coerce')
    
    # 删除空值 (包括刚才转换失败的文本数据)
    df_plot = df_plot.dropna(subset=[lon_col, lat_col, val_col])
    
    # 如果该列全是文本（比如 ph_bin），转换后就空了，直接跳过不画
    if df_plot.empty:
        # print(f"  ℹ️ 跳过变量 {val_col}: 数据非数值或全为空")
        return

    # 计算分位数 (现在肯定是数值了，不会报错)
    try:
        vmin = df_plot[val_col].quantile(mc.clip_q_low)
        vmax = df_plot[val_col].quantile(mc.clip_q_high)
    except Exception as e:
        print(f"  ⚠️ 计算分位数失败 {val_col}: {e}")
        return

    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # 3. 绘制底图 (边界)
    if mc.draw_boundary:
        gdf_boundary = load_online_boundary(mc.adcode)
        if gdf_boundary is not None:
            gdf_boundary.plot(
                ax=ax, 
                facecolor="none",       
                edgecolor=mc.boundary_color, 
                linewidth=mc.boundary_width, 
                zorder=1                
            )
            # 背景色
            gdf_boundary.plot(ax=ax, facecolor="#f0f0f0", alpha=0.3, zorder=0)

    # 4. 绘制散点
    sc = ax.scatter(
        df_plot[lon_col],
        df_plot[lat_col],
        c=df_plot[val_col],
        s=mc.s,
        alpha=mc.alpha,
        cmap="Spectral_r", 
        edgecolors="k",     
        linewidth=0.3,
        vmin=vmin,
        vmax=vmax,
        zorder=2            
    )

    # 5. 美化
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # 颜色条
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.set_title(val_col, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)

    # 6. 保存
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 已保存地图: {out_path.name}")

def make_maps(df: pd.DataFrame, cfg: Dict, out_root: Path) -> List[Path]:
    mc = _get(cfg)
    out_dir = out_root / mc.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 自动识别列名
    lon, lat = mc.lon_col, mc.lat_col
    if lon not in df.columns or lat not in df.columns:
        found = False
        for cand_lon in ["longitude", "long", "lng", "x", "X", "Lon", "LON"]:
            for cand_lat in ["latitude", "lat", "y", "Y", "Lat", "LAT"]:
                if cand_lon in df.columns and cand_lat in df.columns:
                    lon, lat = cand_lon, cand_lat
                    found = True
                    break
            if found: break
        if not found:
            print(f"⚠️ Warning: 找不到经纬度列 ({lon}, {lat})，跳过地图绘制。")
            return []

    # 变量筛选逻辑
    vars_ = []
    if mc.vars_to_map:
        # 用户指定的变量
        vars_ = [v for v in mc.vars_to_map if v in df.columns]
    else:
        # 自动推断变量: 优先画 BCF, Cd, pH 等
        priority_keys = ["BCF", "bcf", "soil Cd", "soil_cd", "crop Cd", "crop_cd", "pH", "ph"]
        # 1. 先找优先变量
        for p in priority_keys:
            for col in df.columns:
                if p.lower() in col.lower() and col not in vars_:
                    vars_.append(col)
        # 2. 补充其他数值列 (排除经纬度和一些中间变量)
        exclude_keywords = [lon.lower(), lat.lower(), "bin", "type", "class", "name", "id", "code"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col not in vars_ and len(vars_) < 8: # 限制最多自动画8张
                is_excluded = any(k in col.lower() for k in exclude_keywords)
                if not is_excluded:
                    vars_.append(col)

    print(f"🗺️ 准备绘制地图，变量: {vars_}")
    print(f"📍 使用经纬度列: {lon}, {lat}")
    print(f"🏙️ 目标区域 Adcode: {mc.adcode}")

    outs: List[Path] = []
    
    # --- 绘图循环 ---
    
    # 1. 绘制不分组的整体图 (All Data)
    if (not mc.by_group) or ("crop" not in df.columns):
        for v in vars_:
            safe_v = str(v).replace(" ", "_").replace("/", "_")
            out_png = out_dir / f"map_all_{safe_v}.png"
            plot_point_map(df, lon, lat, v, f"Map | {v} (All Data)", out_png, mc)
            outs.append(out_png)
        return outs

    # 2. 按分组绘制 (Crop x pH)
    # 检查是否有必要的分组列
    group_cols = [c for c in ["crop", "ph_bin"] if c in df.columns]
    
    if not group_cols:
         # 没有分组列，就只画整体
        for v in vars_:
            safe_v = str(v).replace(" ", "_").replace("/", "_")
            out_png = out_dir / f"map_all_{safe_v}.png"
            plot_point_map(df, lon, lat, v, f"Map | {v} (All Data)", out_png, mc)
            outs.append(out_png)
        return outs

    # 开始分组循环
    groups = df.groupby(group_cols)
    for name, sub_df in groups:
        # name 可能是元组也可能是单个值
        if isinstance(name, tuple):
            group_tag = "_".join([str(n) for n in name])
            title_tag = " | ".join([str(n) for n in name])
        else:
            group_tag = str(name)
            title_tag = str(name)
            
        # 忽略太少数据的组
        if len(sub_df) < 5:
            continue
            
        for v in vars_:
            # 只有当该变量在该组内不全是空值时才画
            # 这里的 dropna 只是为了检查，具体的 to_numeric 转换在 plot_point_map 里做
            if sub_df[v].dropna().empty:
                continue
                
            safe_v = str(v).replace(" ", "_").replace("/", "_")
            
            # 构造文件名: map_玉米_酸性_BCF.png
            fname = f"map_{group_tag}_{safe_v}.png"
            # 简单清理文件名非法字符
            fname = fname.replace("<", "lt").replace(">", "gt").replace(":", "")
            
            out_png = out_dir / fname
            title = f"{title_tag} | {v}"
            
            plot_point_map(sub_df, lon, lat, v, title, out_png, mc)
            outs.append(out_png)

    return outs