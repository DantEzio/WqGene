"""
对每个模型的每个 feature 绘制 真实值 vs 预测值 对比图。
每个模型生成一张图，图中包含全部 12 个 feature 的子图（4区域×3指标），
动态查找全局最高误差时间节点并通过 textbox 标注月日。
所有图片保存至 results/preVStrue/ 目录。

用法: python plot_all_features_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
import os
import pandas as pd
from datetime import datetime, timedelta

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================
# 字体设置（加大字体）
# ============================================================
config = {
    "font.family": 'serif',
    "font.size": 14,
    "mathtext.fontset": 'stix',
    "font.serif": ['Arial'],
    'axes.unicode_minus': False,
}
rcParams.update(config)
label_size = 18
tick_size = 13
legend_size = 11

# ============================================================
# 模型配置：TL 版本 + 普通训练版本
# ============================================================
MODEL_CONFIGS = {
    # --- TL (迁移学习) 版本 ---
    'Gene_TL': {
        'path': 'step1_gene_model_trans_2block/results/gene_model_transfer_learning/prediction.csv',
        'title': 'WqGene-LL',
        'base_hue': '#2166ac',
        'marker_color': '#d62728',
    },
    'LSTM_TL': {
        'path': 'step1_LSTM_model_trans/results/LSTM_model_transfer_learning/prediction.csv',
        'title': 'LSTM-LL',
        'base_hue': '#d6604d',
        'marker_color': '#1f77b4',
    },
    'MLP_TL': {
        'path': 'step1_MLP_model_trans/results/MLP_model_transfer_learning/prediction.csv',
        'title': 'MLP-LL',
        'base_hue': '#1a9641',
        'marker_color': '#d62728',
    },
    # --- 普通 Training 版本 ---
    'Gene_Train': {
        'path': 'step1_gene_model_trans_2block/results/gene_model/prediction.csv',
        'title': 'WqGene',
        'base_hue': '#4393c3',
        'marker_color': '#d62728',
    },
    'LSTM_Train': {
        'path': 'step1_LSTM_model_trans/results/LSTM_model/prediction.csv',
        'title': 'LSTM',
        'base_hue': '#d6604d',
        'marker_color': '#1f77b4',
    },
    'MLP_Train': {
        'path': 'step1_MLP_model_trans/results/MLP_model/prediction.csv',
        'title': 'MLP',
        'base_hue': '#1a9641',
        'marker_color': '#d62728',
    },
}

# 画图起始日期
PLOT_START_DATE = datetime(2025, 7, 13)

# 12 个 feature 的名称和分组（★ 2block: Gene模型 ANA+ANO合并）
# 原始4区域(ANA, ANO, OX, EFF) 用于 LSTM/MLP
# 合并3区域(ANA_ANO, OX, EFF) 用于 Gene 模型
FEATURE_GROUPS_ORIGINAL = ['ANA', 'ANO', 'OX', 'EFF']
FEATURE_GROUPS_MERGED = ['ANA_ANO', 'OX', 'EFF']
FEATURE_INDICES_ORIGINAL = [1, 2, 3]  # LSTM/MLP: 每区域3指标
FEATURE_INDICES_MERGED = [1, 2, 3, 4, 5, 6]  # Gene: ana_ano有6指标, ox/eff各3指标

# Feature 显示名（含上下标）：DO, NH₄⁺, NO₃⁻
FEAT_DISPLAY_NAMES = {
    1: r'DO (mg/L)',
    2: r'NH$_{4}^{+}$ (mg/L)',
    3: r'NO$_{3}^{-}$ (mg/L)',
}

# EFF 区域第1个特征特殊名称：COD
EFF_FEAT1_NAME = r'COD (mg/L)'

# 反归一化参数：每个 (region, feat_idx) 对应的原始最大值（Min-Max, raw_min=0）
# ★ 2block: Gene模型使用 ANA_ANO 合并键
DENORM_MAX = {
    # --- 原始4区域 (LSTM/MLP) ---
    ('ANA', 1): 0.30,   # DO-ANA
    ('ANA', 2): 52.73,  # AN-ANA
    ('ANA', 3): 1.69,   # NN-ANA
    ('ANO', 1): 0.70,   # DO-ANO
    ('ANO', 2): 50.69,  # AN-ANO
    ('ANO', 3): 10.72,  # NN-ANO
    ('OX',  1): 5.89,   # DO-OX
    ('OX',  2): 15.58,  # AN-OX
    ('OX',  3): 29.14,  # NN-OX
    ('EFF', 1): 23.75,  # COD-EFF
    ('EFF', 2): 13.81,  # TN-EFF
    ('EFF', 3): 3.39,   # AN-EFF
    # --- 合并区域 (Gene 2block: ANA+ANO → ANA_ANO, 6维输出) ---
    ('ANA_ANO', 1): 0.30,   # DO-ANA (原ANA_1)
    ('ANA_ANO', 2): 52.73,  # AN-ANA (原ANA_2)
    ('ANA_ANO', 3): 1.69,   # NN-ANA (原ANA_3)
    ('ANA_ANO', 4): 0.70,   # DO-ANO (原ANO_1)
    ('ANA_ANO', 5): 50.69,  # AN-ANO (原ANO_2)
    ('ANA_ANO', 6): 10.72,  # NN-ANO (原ANO_3)
}

# 区域显示名（★ 2block: 增加 ANA_ANO 合并显示名）
REGION_DISPLAY_NAMES = {
    'ANA': 'Anaerobic',
    'ANO': 'Anoxic',
    'ANA_ANO': 'Anaerobic & Anoxic',  # ★ 2block 合并
    'OX':  'Oxic',
    'EFF': 'Effluent',
}


def hex_to_rgb(hex_color):
    """将十六进制颜色转为 RGB 元组 (0~1)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def generate_shades(base_hex, n_shades=12):
    """
    基于基础色生成 n_shades 组深浅配对 (real_color, pred_color)。
    同一模型内不同子图用深浅区分，保留色系一致性。
    
    Returns:
        list of (real_hex, pred_hex) tuples
    """
    base_rgb = np.array(hex_to_rgb(base_hex))
    
    pairs = []
    for i in range(n_shades):
        # real: 从 base 到较深的渐变
        depth_factor = 0.55 + 0.45 * (i / max(1, n_shades - 1))
        real_rgb = np.clip(base_rgb * depth_factor, 0, 1)
        
        # pred: 从 base 到较浅的渐变（用互补亮度）
        light_factor = 1.15 + 0.5 * (i / max(1, n_shades - 1))
        pred_rgb = np.clip(base_rgb + (1 - base_rgb) * (light_factor - 1) * 0.5, 0, 1)
        
        def rgb_to_hex(rgb):
            return '#' + ''.join(f'{int(c * 255):02x}' for c in rgb)
        
        pairs.append((rgb_to_hex(real_rgb), rgb_to_hex(pred_rgb)))
    
    return pairs


def load_dates():
    """从 wq.xlsx 读取实际日期列表"""
    try:
        df = pd.read_excel('data_WQ/wq.xlsx')
        return df['time'].tolist()
    except Exception as e:
        print(f"警告: 无法读取 wq.xlsx: {e}")
        return None


def generate_dates_for_plot(time_steps, actual_dates):
    """将 time_step 映射为实际日期"""
    plot_dates = []
    for ts in time_steps:
        ts_int = int(ts)
        if actual_dates is not None and ts_int < len(actual_dates):
            plot_dates.append(actual_dates[ts_int])
        else:
            if plot_dates:
                last = plot_dates[-1]
                if hasattr(last, 'strftime'):
                    plot_dates.append(last + timedelta(minutes=15))
                else:
                    try:
                        plot_dates.append(pd.to_datetime(last) + timedelta(minutes=15))
                    except:
                        plot_dates.append(last)
            else:
                plot_dates.append(datetime(2025, 5, 30) + timedelta(days=ts_int))
    return plot_dates


def format_date_short(date_obj):
    """格式化日期用于刻度标签（短格式 mm-dd）"""
    if hasattr(date_obj, 'strftime'):
        return date_obj.strftime('%m-%d')
    try:
        return str(pd.to_datetime(date_obj))[5:10]
    except:
        return str(date_obj)


def find_model_max_error(path, is_tl=False):
    """
    找出单个模型的最高误差时间节点。
    
    Args:
        path: prediction.csv 路径
        is_tl: 是否为 TL 模型（需加偏移量）
    
    Returns:
        dict: {row_index, time_step, date_str, max_mae}
    """
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    # ★ 兼容两种列名: _real_X (LSTM/MLP) 和 _true_X (Gene模型)
    real_cols = sorted([c for c in df.columns if 'real' in c.lower() or 'true' in c.lower()])
    pred_cols = sorted([c for c in df.columns if 'pred' in c.lower()])
    mae = np.mean(np.abs(df[pred_cols].values - df[real_cols].values), axis=1)
    row_idx = int(np.argmax(mae))   # 行索引（关键！）
    val = float(mae[row_idx])
    ts = int(df['time_step'].values[row_idx])
    
    # TL 模型加偏移量
    if is_tl:
        ts += 4263
    
    # 获取日期
    dates = load_dates()
    date_str = ''
    if dates is not None and ts < len(dates):
        d = dates[ts]
        if hasattr(d, 'strftime'):
            date_str = d.strftime('%Y-%m-%d')
        else:
            try:
                date_str = str(pd.to_datetime(d))[:10]
            except:
                date_str = ''
    
    return {
        'row_index': row_idx,     # 行索引，可直接用于其他模型
        'time_step': ts,
        'date_str': date_str,
        'max_mae': val,
    }


def add_max_error_textbox(fig, max_info):
    """
    在图的右上角添加一个 textbox，标注最高误差的月日。
    """
    if max_info is None:
        return
    
    date_str = max_info['date_str']
    # 提取月日部分（英文格式避免字体问题）
    month_day = ''
    try:
        dt = pd.to_datetime(date_str)
        month_day = dt.strftime('%b %d')
    except:
        month_day = date_str
    
    text = f"Max Error Point\n{month_day}\nMAE = {max_info['max_mae']:.4f}"
    
    # 在 figure 级别添加 textbox
    fig.text(0.97, 0.98, text,
             transform=fig.transFigure,
             fontsize=label_size - 2,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.6',
                       fc='#fff5f5', edgecolor='#d62728',
                       linewidth=1.8, alpha=0.92),
             fontweight='bold')


def plot_single_model(model_key, model_cfg, output_dir, max_info):
    """
    为单个模型绘制一张包含全 feature 子图的对比图。
    ★ 2block: Gene模型使用3区域(ANA_ANO/OX/EFF)，LSTM/MLP使用4区域(ANA/ANO/OX/EFF)
    """
    path = model_cfg['path']
    if not os.path.exists(path):
        print(f"  [跳过] 文件不存在: {path}")
        return None
    
    # ★ 2block: 根据模型类型选择区域配置
    is_gene_model = 'Gene' in model_key
    if is_gene_model:
        FEATURE_GROUPS = FEATURE_GROUPS_MERGED  # ['ANA_ANO', 'OX', 'EFF']
        # Gene模型: ana_ano有6指标, ox有3指标, eff有3指标 → 共12特征
        REGION_FEAT_INDICES = {
            'ANA_ANO': [1, 2, 3, 4, 5, 6],  # 6维 (原ANA_1~3 + ANO_1~3)
            'OX': [1, 2, 3],
            'EFF': [1, 2, 3],
        }
        n_rows = 4  # ANA_ANO(2行) + OX(1行) + EFF(1行)
        n_cols = 3  # 每行最多3列 (ana_ano需要2行来显示6个指标)
        n_shades_total = 12  # 总共12个特征
    else:
        FEATURE_GROUPS = FEATURE_GROUPS_ORIGINAL  # ['ANA', 'ANO', 'OX', 'EFF']
        REGION_FEAT_INDICES = {rg: FEATURE_INDICES_ORIGINAL for rg in FEATURE_GROUPS}
        n_rows = len(FEATURE_GROUPS)
        n_cols = 3
        n_shades_total = 12
    
    df = pd.read_csv(path)
    time_steps = df['time_step'].values
    
    # TL 模型的 time_step 从 0 开始重新编号，需加偏移量与 wq.xlsx 索引对齐
    if 'TL' in model_key:
        offset = 4263
        time_steps = time_steps + offset
    
    actual_dates = load_dates()
    dates = generate_dates_for_plot(time_steps, actual_dates)
    
    # ---- 截取从 PLOT_START_DATE 开始的数据 ----
    start_indices = []
    for i, d in enumerate(dates):
        try:
            dt = d if hasattr(d, 'strftime') else pd.to_datetime(d)
            if hasattr(dt, 'timestamp') and dt >= PLOT_START_DATE:
                start_indices.append(i)
        except:
            pass
    
    if start_indices:
        start_idx = start_indices[0]
        df = df.iloc[start_idx:].reset_index(drop=True)
        time_steps = df['time_step'].values
        dates = dates[start_idx:]
    
    x_pos = np.arange(len(dates))
    marker_color = model_cfg['marker_color']
    
    # 生成该模型的配色组
    shades = generate_shades(model_cfg['base_hue'], n_shades=n_shades_total)
    
    # ★ 2block: 创建图形布局
    # Gene模型: 4行(ANA_ANO占2行+OX+EFF) × 3列 = 12子图
    # LSTM/MLP: 4行(ANA+ANO+OX+EFF) × 3列 = 12子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 17))
    
    stats_list = []
    shade_idx = 0
    
    # ★ 2block: 构建区域→(行,列)位置的映射
    plot_positions = []  # list of (region, feat_idx, row, col)
    if is_gene_model:
        # Gene模型: ANA_ANO前3指标→Row0, 后3指标→Row1; OX→Row2; EFF→Row3
        ana_ano_feats = REGION_FEAT_INDICES['ANA_ANO']  # [1,2,3,4,5,6]
        for i, feat in enumerate(ana_ano_feats[:3]):
            plot_positions.append(('ANA_ANO', feat, 0, i))
        for i, feat in enumerate(ana_ano_feats[3:]):
            plot_positions.append(('ANA_ANO', feat, 1, i))
        for i, feat in enumerate(REGION_FEAT_INDICES['OX']):
            plot_positions.append(('OX', feat, 2, i))
        for i, feat in enumerate(REGION_FEAT_INDICES['EFF']):
            plot_positions.append(('EFF', feat, 3, i))
    else:
        # LSTM/MLP: 每个区域1行
        for row_idx, region in enumerate(FEATURE_GROUPS):
            for col_idx, feat_idx in enumerate(REGION_FEAT_INDICES[region]):
                plot_positions.append((region, feat_idx, row_idx, col_idx))
    
    # ★ 2block: 按位置逐个子图绘制
    for region, feat_idx, row_idx, col_idx in plot_positions:
        ax = axes[row_idx, col_idx]
        
        # ★ 兼容两种列名格式: _real_X (LSTM/MLP & TL Gene) 和 _true_X (非TL Gene)
        if is_gene_model:
            # Gene模型: ANA_ANO → 映射回原始 ANA/ANO
            if region == 'ANA_ANO':
                if feat_idx <= 3:
                    orig_region = 'ANA'
                    orig_idx = feat_idx  # 保持1-based索引 (_real_1, _real_2...)
                else:
                    orig_region = 'ANO'
                    orig_idx = feat_idx - 3  # 4→1, 5→2, 6→3 (保持1-based)
            else:
                orig_region = region
                orig_idx = feat_idx  # 保持1-based索引
            
            # 先尝试 _real_X 格式 (TL Gene / LSTM / MLF)，再尝试 _true_X 格式 (非TL Gene)
            real_col = f'{orig_region}_real_{orig_idx}'
            pred_col = f'{orig_region}_pred_{orig_idx}'
            if real_col not in df.columns or pred_col not in df.columns:
                # 回退到 _true_0-based 格式 (非TL Gene模型)
                real_col = f'{orig_region}_true_{orig_idx - 1}'
                pred_col = f'{orig_region}_pred_{orig_idx - 1}'
        else:
            real_col = f'{region}_real_{feat_idx}'
            pred_col = f'{region}_pred_{feat_idx}'
        
        if real_col not in df.columns or pred_col not in df.columns:
            ax.text(0.5, 0.5, f'{real_col}\nnot found',
                    ha='center', va='center', fontsize=tick_size,
                    transform=ax.transAxes)
            continue
        
        real_vals = df[real_col].values
        pred_vals = df[pred_col].values
        mae = np.abs(pred_vals - real_vals)
        
        # ---- 反归一化（Min-Max, raw_min=0）----
        raw_max = DENORM_MAX.get((region, feat_idx), 1.0)
        real_vals = real_vals * raw_max
        pred_vals = pred_vals * raw_max
        mae = mae * raw_max
        
        # 当前子图的配色（深浅变化）
        real_color, pred_color = shades[shade_idx]
        shade_idx += 1
        
        # ---- 绘制真实值 ----
        ax.plot(x_pos, real_vals, color=real_color, linewidth=1.3,
                alpha=0.4, label='Observed', zorder=3)
        
        # ---- 绘制预测值 ----
        ax.plot(x_pos, pred_vals, color=pred_color, linewidth=1.3,
                alpha=0.85, linestyle='--', label='Predicted', zorder=3)
        
        # ---- 标注最高误差时间节点 ----
        # 直接使用普通模型的行索引（row_index），不通过 time_step 查找
        target_row_idx = max_info['row_index'] if max_info else None
        if target_row_idx is not None and target_row_idx < len(real_vals):
            tgt_idx = target_row_idx
            tgt_real = real_vals[tgt_idx]
            tgt_pred = pred_vals[tgt_idx]
            
            # 垂直虚线
            ax.axvline(x=tgt_idx, color=marker_color,
                       linestyle=':', linewidth=1.5, alpha=0.75, zorder=4)
            # 散点标记
            ax.scatter([tgt_idx], [tgt_real], color=real_color,
                       s=50, zorder=5, edgecolors='white', linewidths=0.8)
            ax.scatter([tgt_idx], [tgt_pred], color=pred_color,
                       s=50, zorder=5, edgecolors='white', linewidths=0.8,
                       marker='s')
            
            # 日期 textbox 放在红色竖线上方
            date_display = ''
            try:
                dt = pd.to_datetime(max_info['date_str'])
                date_display = dt.strftime('%b %d')
            except:
                date_display = str(max_info.get('date_str', ''))
            
            y_max_val = max(np.max(real_vals), np.max(pred_vals))
            y_range = np.max(pred_vals) - np.min(real_vals)
            text_y = y_max_val + y_range * 0.08
            
            ax.text(tgt_idx, text_y,
                    date_display,
                    fontsize=legend_size + 1,
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.35',
                              fc='white', edgecolor='#cccccc',
                              linewidth=0.8, alpha=0.95),
                    color='black', fontweight='bold',
                    zorder=10)
        
        # ---- 坐标轴设置（无 title）----
        ax.grid(True, alpha=0.15)
        
        # Y 轴标签：用 feature 名字
        # ★ 2block: Gene模型 ANA_ANO 的 feat 4~6 对应原 ANO 区域指标
        if region == 'EFF' and feat_idx == 1:
            ylabel = EFF_FEAT1_NAME
        elif is_gene_model and region == 'ANA_ANO':
            # feat 1~3 → 原ANA指标, feat 4~6 → 原ANO指标
            if feat_idx <= 3:
                ylabel = FEAT_DISPLAY_NAMES.get(feat_idx, f'Feat_{feat_idx}')
            else:
                # 映射回原始ANO的1~3
                orig_feat = feat_idx - 3
                ylabel = FEAT_DISPLAY_NAMES.get(orig_feat, f'Feat_{orig_feat}')
        else:
            ylabel = FEAT_DISPLAY_NAMES.get(feat_idx, f'Feat_{feat_idx}')
        ax.set_ylabel(ylabel, fontsize=tick_size + 1)
        
        # 只在最底行显示 X 轴标签
        if row_idx == n_rows - 1:
            ax.set_xlabel('Date', fontsize=tick_size + 1)
        
        # X 轴刻度
        n_pts = len(dates)
        n_ticks = min(5, max(4, n_pts))
        tick_positions = [int(i * (n_pts - 1) / (n_ticks - 1))
                          for i in range(n_ticks)]
        tick_labels = [format_date_short(dates[i]) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=tick_size - 1)
        ax.tick_params(axis='y', labelsize=tick_size - 1)
        
        # 每个子图都有 legend（右上角）
        ax.legend(loc='upper right', fontsize=legend_size,
                  framealpha=0.88, edgecolor='gray')
        
        # 统计信息（仅保存 CSV，不在图中显示）
        feat_mae_mean = np.mean(mae)
        ss_res = np.sum((real_vals - pred_vals) ** 2)
        ss_tot = np.sum((real_vals - np.mean(real_vals)) ** 2)
        feat_r2 = 1 - ss_res / (ss_tot + 1e-10)
        stats_list.append({
            'Feature': f'{region}_{feat_idx}',
            'MAE_mean': feat_mae_mean,
            'MAE_max': np.max(mae),
            'R2': feat_r2,
        })
    
    # 区域行标签（左侧旋转文字）
    # ★ 2block: Gene模型 ANA_ANO 跨两行(Row0,1)，只在Row0标注
    if is_gene_model:
        # Row 0: ANA_ANO 标签
        axes[0, 0].annotate(REGION_DISPLAY_NAMES.get('ANA_ANO', 'ANA_ANO'),
                              xy=(-0.24, 0.5),
                              xycoords=axes[0, 0].transAxes,
                              fontsize=label_size, fontweight='bold',
                              ha='right', va='center', rotation=90)
        # Row 2: OX 标签
        axes[2, 0].annotate(REGION_DISPLAY_NAMES.get('OX', 'OX'),
                              xy=(-0.24, 0.5),
                              xycoords=axes[2, 0].transAxes,
                              fontsize=label_size, fontweight='bold',
                              ha='right', va='center', rotation=90)
        # Row 3: EFF 标签
        axes[3, 0].annotate(REGION_DISPLAY_NAMES.get('EFF', 'EFF'),
                              xy=(-0.24, 0.5),
                              xycoords=axes[3, 0].transAxes,
                              fontsize=label_size, fontweight='bold',
                              ha='right', va='center', rotation=90)
    else:
        for row_idx, region in enumerate(FEATURE_GROUPS):
            axes[row_idx, 0].annotate(REGION_DISPLAY_NAMES.get(region, region),
                                      xy=(-0.24, 0.5),
                                      xycoords=axes[row_idx, 0].transAxes,
                                      fontsize=label_size, fontweight='bold',
                                      ha='right', va='center', rotation=90)
    
    # 横向中轴线 title（放在图上方中间位置，靠近图）
    fig.text(0.5, 0.995, model_cfg['title'],
             transform=fig.transFigure,
             fontsize=label_size + 4, fontweight='bold',
             ha='center', va='top')
    
    plt.tight_layout(rect=[0.16, 0, 0.98, 0.97])
    
    # 保存图片
    safe_name = model_key.replace('_', '-')
    output_path = os.path.join(output_dir, f'{safe_name}_all_features.png')
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()
    
    # 保存统计 CSV
    stats_df = pd.DataFrame(stats_list)
    stats_csv = os.path.join(output_dir, f'{safe_name}_feature_stats.csv')
    stats_df.to_csv(stats_csv, index=False)
    
    print(f"  已保存: {output_path}")
    print(f"  统计表: {stats_csv}")
    
    return stats_df


MAX_ERROR_TIME_STEP_FALLBACK = 6146


def main():
    print("=" * 60)
    print("绘制各模型全 Feature 对比图")
    print(f"起始日期: {PLOT_START_DATE.strftime('%Y-%m-%d')}")
    print("输出目录: results/preVStrue/")
    print("=" * 60)
    
    output_dir = os.path.join('results', 'preVStrue')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录已创建: {output_dir}")
    
    # 使用普通(Train)模型的最高误差行索引，统一应用到所有LL模型
    print("\n正在查找普通模型的最高误差位置...")
    
    train_paths = {
        'Gene': 'step1_gene_model_trans_2block/results/gene_model/prediction.csv',
        'LSTM': 'step1_LSTM_model_trans/results/LSTM_model/prediction.csv',
        'MLP':  'step1_MLP_model_trans/results/MLP_model/prediction.csv',
    }
    
    global_max_mae = -1
    unified_max_info = None
    
    for model_name, path in train_paths.items():
        info = find_model_max_error(path, is_tl=False)
        if info and info['max_mae'] > global_max_mae:
            global_max_mae = info['max_mae']
            unified_max_info = info
    
    if unified_max_info:
        print(f"  统一最高误差点: row_index={unified_max_info['row_index']}, "
              f"日期={unified_max_info['date_str']}, "
              f"MAE={unified_max_info['max_mae']:.6f}")
    else:
        print("  警告: 未能找到最高误差位置")
        unified_max_info = None
    print()
    
    all_stats = {}
    
    for model_key, model_cfg in MODEL_CONFIGS.items():
        print(f"--- {model_cfg['title']} ---")
        stats_df = plot_single_model(model_key, model_cfg, output_dir, unified_max_info)
        if stats_df is not None:
            all_stats[model_key] = stats_df
        print()
    
    # 汇总统计
    print("=" * 60)
    print("汇总统计")
    print("=" * 60)
    
    summary_rows = []
    for model_key, stats_df in all_stats.items():
        for _, row in stats_df.iterrows():
            summary_rows.append({
                'Model': model_key,
                'Feature': row['Feature'],
                'MAE_mean': row['MAE_mean'],
                'MAE_max': row['MAE_max'],
                'R2': row['R2'],
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, 'all_models_feature_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"汇总统计已保存: {summary_path}")
        
        print("\n各模型平均 MAE（跨所有 feature）:")
        model_avg = summary_df.groupby('Model')['MAE_mean'].mean().sort_values()
        for m, v in model_avg.items():
            print(f"  {m}: {v:.6f}")
    
    print("\n完成！")


if __name__ == '__main__':
    main()
