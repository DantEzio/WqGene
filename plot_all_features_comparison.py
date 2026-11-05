"""
plot_all_features_comparison.py - "real vs predicted" comparison for every output feature
=========================================================
Purpose:
    draw a 4-row x 3-col comparison figure per model covering all 12 output features
    (four zones Anaerobic / Anoxic / Oxic / Effluent x three indicators DO / AN / NN):
    1) Overlay the truth and prediction curves in each panel for a direct visual comparison
    2) Automatically locate the global maximum-error time point and annotate the month-day with a textbox
    3) Label the zones on the left; the x axis is the actual date and only the bottom row shows the Date label
Input:
    step1_*/.../prediction.csv (real columns contain true/real, prediction columns contain pred)
    data_WQ/wq.xlsx (time column, used to convert the time axis)
Output:
    results/preVStrue/{model name}_all_features.png
Usage:
    python plot_all_features_comparison.py       # run from the project root
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
# Font setup (larger fonts)
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
# Model configuration: TL version plus the plain training version
# ============================================================
MODEL_CONFIGS = {
    # --- TL (transfer learning) version ---
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
    # --- Plain training version ---
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

# Start date of the plots
PLOT_START_DATE = datetime(2025, 7, 13)

# Names and grouping of the 12 features (2block: Gene model merges ANA+ANO)
# Original 4 regions (ANA, ANO, OX, EFF) used by LSTM/MLP
# Merge the 3 regions (ANA_ANO, OX, EFF) for the Gene model
FEATURE_GROUPS_ORIGINAL = ['ANA', 'ANO', 'OX', 'EFF']
FEATURE_GROUPS_MERGED = ['ANA_ANO', 'OX', 'EFF']
FEATURE_INDICES_ORIGINAL = [1, 2, 3]  # LSTM/MLP: 3 indicators per region
FEATURE_INDICES_MERGED = [1, 2, 3, 4, 5, 6]  # Gene: ana_ano has 6 indicators, ox/eff 3 each

# Feature display names (with sub/superscripts): DO, NH4+, NO3-
FEAT_DISPLAY_NAMES = {
    1: r'DO (mg/L)',
    2: r'NH$_{4}^{+}$ (mg/L)',
    3: r'NO$_{3}^{-}$ (mg/L)',
}

# Special name of the first EFF feature: COD
EFF_FEAT1_NAME = r'COD (mg/L)'

# De-normalization parameters: original max value of each (region, feat_idx) (Min-Max, raw_min=0)
# 2block: the Gene model uses the merged ANA_ANO key
DENORM_MAX = {
    # --- Original 4 regions (LSTM/MLP) ---
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
    # --- Merged region (Gene 2block: ANA+ANO -> ANA_ANO, 6-dim output) ---
    ('ANA_ANO', 1): 0.30,   # DO-ANA (originally ANA_1)
    ('ANA_ANO', 2): 52.73,  # AN-ANA (originally ANA_2)
    ('ANA_ANO', 3): 1.69,   # NN-ANA (originally ANA_3)
    ('ANA_ANO', 4): 0.70,   # DO-ANO (originally ANO_1)
    ('ANA_ANO', 5): 50.69,  # AN-ANO (originally ANO_2)
    ('ANA_ANO', 6): 10.72,  # NN-ANO (originally ANO_3)
}

# Region display names (2block adds the merged ANA_ANO name)
REGION_DISPLAY_NAMES = {
    'ANA': 'Anaerobic',
    'ANO': 'Anoxic',
    'ANA_ANO': 'Anaerobic & Anoxic',  # 2block merged
    'OX':  'Oxic',
    'EFF': 'Effluent',
}


def hex_to_rgb(hex_color):
    """Convert a hex color to an RGB tuple (0-1)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def generate_shades(base_hex, n_shades=12):
    """
    generate n_shades dark/light pairs (real_color, pred_color) from a base color.
    panels of the same model differ in shade to keep the palette consistent.
    
    Returns:
        list of (real_hex, pred_hex) tuples
    """
    base_rgb = np.array(hex_to_rgb(base_hex))
    
    pairs = []
    for i in range(n_shades):
        # real: gradient from the base color to darker shades
        depth_factor = 0.55 + 0.45 * (i / max(1, n_shades - 1))
        real_rgb = np.clip(base_rgb * depth_factor, 0, 1)
        
        # pred: gradient from the base color to lighter shades (complementary brightness)
        light_factor = 1.15 + 0.5 * (i / max(1, n_shades - 1))
        pred_rgb = np.clip(base_rgb + (1 - base_rgb) * (light_factor - 1) * 0.5, 0, 1)
        
        def rgb_to_hex(rgb):
            return '#' + ''.join(f'{int(c * 255):02x}' for c in rgb)
        
        pairs.append((rgb_to_hex(real_rgb), rgb_to_hex(pred_rgb)))
    
    return pairs


def load_dates():
    """Read the list of actual dates from wq.xlsx"""
    try:
        df = pd.read_excel('data_WQ/wq.xlsx')
        return df['time'].tolist()
    except Exception as e:
        print(f"Warning: cannot read wq.xlsx: {e}")
        return None


def generate_dates_for_plot(time_steps, actual_dates):
    """Map time_step to the actual date"""
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
    """Format a date for tick labels (short mm-dd form)"""
    if hasattr(date_obj, 'strftime'):
        return date_obj.strftime('%m-%d')
    try:
        return str(pd.to_datetime(date_obj))[5:10]
    except:
        return str(date_obj)


def find_model_max_error(path, is_tl=False):
    """
    find the time point with the maximum error for a single model.
    
    Args:
        path: path of prediction.csv
        is_tl: whether this is a TL model (an offset must be added)
    
    Returns:
        dict: {row_index, time_step, date_str, max_mae}
    """
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    # Supports both column-name styles: _real_X (LSTM/MLP) and _true_X (Gene model)
    real_cols = sorted([c for c in df.columns if 'real' in c.lower() or 'true' in c.lower()])
    pred_cols = sorted([c for c in df.columns if 'pred' in c.lower()])
    mae = np.mean(np.abs(df[pred_cols].values - df[real_cols].values), axis=1)
    row_idx = int(np.argmax(mae))   # row index (important!)
    val = float(mae[row_idx])
    ts = int(df['time_step'].values[row_idx])
    
    # Add the offset for TL models
    if is_tl:
        ts += 4263
    
    # Get the dates
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
        'row_index': row_idx,     # row index, usable directly for the other models
        'time_step': ts,
        'date_str': date_str,
        'max_mae': val,
    }


def add_max_error_textbox(fig, max_info):
    """
    add a textbox in the top-right corner of the figure with the month-day of the maximum error.
    """
    if max_info is None:
        return
    
    date_str = max_info['date_str']
    # Take the month-day part (English format avoids font issues)
    month_day = ''
    try:
        dt = pd.to_datetime(date_str)
        month_day = dt.strftime('%b %d')
    except:
        month_day = date_str
    
    text = f"Max Error Point\n{month_day}\nMAE = {max_info['max_mae']:.4f}"
    
    # Add a textbox at the figure level
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
    draw a comparison figure containing all feature subplots for a single model.
    2block: the Gene model uses 3 regions (ANA_ANO/OX/EFF), LSTM/MLP use 4 (ANA/ANO/OX/EFF)
    """
    path = model_cfg['path']
    if not os.path.exists(path):
        print(f"  [skipped] file not found: {path}")
        return None
    
    # 2block: pick the region configuration based on the model type
    is_gene_model = 'Gene' in model_key
    if is_gene_model:
        FEATURE_GROUPS = FEATURE_GROUPS_MERGED  # ['ANA_ANO', 'OX', 'EFF']
        # Gene model: ana_ano has 6 targets, ox 3, eff 3 -> 12 features in total
        REGION_FEAT_INDICES = {
            'ANA_ANO': [1, 2, 3, 4, 5, 6],  # 6 dims (originally ANA_1-3 + ANO_1-3)
            'OX': [1, 2, 3],
            'EFF': [1, 2, 3],
        }
        n_rows = 4  # ANA_ANO (2 rows) + OX (1 row) + EFF (1 row)
        n_cols = 3  # at most 3 columns per row (ana_ano needs 2 rows to show its 6 indicators)
        n_shades_total = 12  # 12 features in total
    else:
        FEATURE_GROUPS = FEATURE_GROUPS_ORIGINAL  # ['ANA', 'ANO', 'OX', 'EFF']
        REGION_FEAT_INDICES = {rg: FEATURE_INDICES_ORIGINAL for rg in FEATURE_GROUPS}
        n_rows = len(FEATURE_GROUPS)
        n_cols = 3
        n_shades_total = 12
    
    df = pd.read_csv(path)
    time_steps = df['time_step'].values
    
    # TL models renumber time_step from 0, so an offset is needed to align with the wq.xlsx index
    if 'TL' in model_key:
        offset = 4263
        time_steps = time_steps + offset
    
    actual_dates = load_dates()
    dates = generate_dates_for_plot(time_steps, actual_dates)
    
    # ---- Cut the series starting from PLOT_START_DATE ----
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
    
    # Build the color group of this model
    shades = generate_shades(model_cfg['base_hue'], n_shades=n_shades_total)
    
    # 2block: create the figure layout
    # Gene model: 4 rows (ANA_ANO takes 2 rows + OX + EFF) x 3 cols = 12 panels
    # LSTM/MLP: 4 rows (ANA+ANO+OX+EFF) x 3 cols = 12 panels
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 17))
    
    stats_list = []
    shade_idx = 0
    
    # 2block: build the region -> (row, col) position mapping
    plot_positions = []  # list of (region, feat_idx, row, col)
    if is_gene_model:
        # Gene model: ANA_ANO feats 1-3 -> Row0, feats 4-6 -> Row1; OX -> Row2; EFF -> Row3
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
        # LSTM/MLP: one row per region
        for row_idx, region in enumerate(FEATURE_GROUPS):
            for col_idx, feat_idx in enumerate(REGION_FEAT_INDICES[region]):
                plot_positions.append((region, feat_idx, row_idx, col_idx))
    
    # 2block: draw panels one by one by position
    for region, feat_idx, row_idx, col_idx in plot_positions:
        ax = axes[row_idx, col_idx]
        
        # Supports both column-name styles: _real_X (LSTM/MLP & TL Gene) and _true_X (non-TL Gene)
        if is_gene_model:
            # Gene model: ANA_ANO -> mapped back to the original ANA/ANO
            if region == 'ANA_ANO':
                if feat_idx <= 3:
                    orig_region = 'ANA'
                    orig_idx = feat_idx  # keep the 1-based index (_real_1, _real_2...)
                else:
                    orig_region = 'ANO'
                    orig_idx = feat_idx - 3  # 4->1, 5->2, 6->3 (keeping 1-based)
            else:
                orig_region = region
                orig_idx = feat_idx  # keep the 1-based index
            
            # Try the _real_X format first (TL Gene / LSTM / MLP), then the _true_X format (non-TL Gene)
            real_col = f'{orig_region}_real_{orig_idx}'
            pred_col = f'{orig_region}_pred_{orig_idx}'
            if real_col not in df.columns or pred_col not in df.columns:
                # Fall back to the _true_ 0-based format (non-TL Gene model)
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
        
        # ---- De-normalization (Min-Max, raw_min=0) ----
        raw_max = DENORM_MAX.get((region, feat_idx), 1.0)
        real_vals = real_vals * raw_max
        pred_vals = pred_vals * raw_max
        mae = mae * raw_max
        
        # Color palette of the current panel (varying shade)
        real_color, pred_color = shades[shade_idx]
        shade_idx += 1
        
        # ---- Plot the ground truth ----
        ax.plot(x_pos, real_vals, color=real_color, linewidth=1.3,
                alpha=0.4, label='Observed', zorder=3)
        
        # ---- Plot the predicted values ----
        ax.plot(x_pos, pred_vals, color=pred_color, linewidth=1.3,
                alpha=0.85, linestyle='--', label='Predicted', zorder=3)
        
        # ---- Annotate the time point with the largest error ----
        # Use the row index (row_index) of the plain model directly instead of looking up time_step
        target_row_idx = max_info['row_index'] if max_info else None
        if target_row_idx is not None and target_row_idx < len(real_vals):
            tgt_idx = target_row_idx
            tgt_real = real_vals[tgt_idx]
            tgt_pred = pred_vals[tgt_idx]
            
            # Vertical dashed line
            ax.axvline(x=tgt_idx, color=marker_color,
                       linestyle=':', linewidth=1.5, alpha=0.75, zorder=4)
            # Scatter markers
            ax.scatter([tgt_idx], [tgt_real], color=real_color,
                       s=50, zorder=5, edgecolors='white', linewidths=0.8)
            ax.scatter([tgt_idx], [tgt_pred], color=pred_color,
                       s=50, zorder=5, edgecolors='white', linewidths=0.8,
                       marker='s')
            
            # Place the date textbox above the red vertical line
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
        
        # ---- Axis setup (no title) ----
        ax.grid(True, alpha=0.15)
        
        # Y axis label: use the feature name
        # 2block: in the Gene model, ANA_ANO feats 4-6 correspond to the original ANO targets
        if region == 'EFF' and feat_idx == 1:
            ylabel = EFF_FEAT1_NAME
        elif is_gene_model and region == 'ANA_ANO':
            # feats 1-3 -> original ANA targets, feats 4-6 -> original ANO targets
            if feat_idx <= 3:
                ylabel = FEAT_DISPLAY_NAMES.get(feat_idx, f'Feat_{feat_idx}')
            else:
                # Map back to the original ANO targets 1-3
                orig_feat = feat_idx - 3
                ylabel = FEAT_DISPLAY_NAMES.get(orig_feat, f'Feat_{orig_feat}')
        else:
            ylabel = FEAT_DISPLAY_NAMES.get(feat_idx, f'Feat_{feat_idx}')
        ax.set_ylabel(ylabel, fontsize=tick_size + 1)
        
        # Show the X axis labels only in the bottom row
        if row_idx == n_rows - 1:
            ax.set_xlabel('Date', fontsize=tick_size + 1)
        
        # X axis ticks
        n_pts = len(dates)
        n_ticks = min(5, max(4, n_pts))
        tick_positions = [int(i * (n_pts - 1) / (n_ticks - 1))
                          for i in range(n_ticks)]
        tick_labels = [format_date_short(dates[i]) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=tick_size - 1)
        ax.tick_params(axis='y', labelsize=tick_size - 1)
        
        # Every panel has a legend (top-right)
        ax.legend(loc='upper right', fontsize=legend_size,
                  framealpha=0.88, edgecolor='gray')
        
        # Statistics (saved to CSV only, not shown in the figure)
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
    
    # Region row labels (rotated text on the left)
    # 2block: the Gene model ANA_ANO spans two rows (Row0, Row1); annotate only on Row0
    if is_gene_model:
        # Row 0: ANA_ANO labels
        axes[0, 0].annotate(REGION_DISPLAY_NAMES.get('ANA_ANO', 'ANA_ANO'),
                              xy=(-0.24, 0.5),
                              xycoords=axes[0, 0].transAxes,
                              fontsize=label_size, fontweight='bold',
                              ha='right', va='center', rotation=90)
        # Row 2: OX labels
        axes[2, 0].annotate(REGION_DISPLAY_NAMES.get('OX', 'OX'),
                              xy=(-0.24, 0.5),
                              xycoords=axes[2, 0].transAxes,
                              fontsize=label_size, fontweight='bold',
                              ha='right', va='center', rotation=90)
        # Row 3: EFF labels
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
    
    # Title of the horizontal center line (placed above the panel, close to it)
    fig.text(0.5, 0.995, model_cfg['title'],
             transform=fig.transFigure,
             fontsize=label_size + 4, fontweight='bold',
             ha='center', va='top')
    
    plt.tight_layout(rect=[0.16, 0, 0.98, 0.97])
    
    # Save the figure
    safe_name = model_key.replace('_', '-')
    output_path = os.path.join(output_dir, f'{safe_name}_all_features.png')
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()
    
    # Save the statistics CSV
    stats_df = pd.DataFrame(stats_list)
    stats_csv = os.path.join(output_dir, f'{safe_name}_feature_stats.csv')
    stats_df.to_csv(stats_csv, index=False)
    
    print(f"  Saved: {output_path}")
    print(f"  Statistics table: {stats_csv}")
    
    return stats_df


MAX_ERROR_TIME_STEP_FALLBACK = 6146


def main():
    print("=" * 60)
    print("Plot the all-feature comparison for each model")
    print(f"Start date: {PLOT_START_DATE.strftime('%Y-%m-%d')}")
    print("Output directory: results/preVStrue/")
    print("=" * 60)
    
    output_dir = os.path.join('results', 'preVStrue')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory created: {output_dir}")
    
    # Use the row index of the highest error of the plain (Train) model for all TL models
    print("\nLooking for the maximum-error position of the plain model...")
    
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
        print(f"  Unified maximum-error point: row_index={unified_max_info['row_index']}, "
              f"date={unified_max_info['date_str']}, "
              f"MAE={unified_max_info['max_mae']:.6f}")
    else:
        print("  Warning: could not locate the maximum-error position")
        unified_max_info = None
    print()
    
    all_stats = {}
    
    for model_key, model_cfg in MODEL_CONFIGS.items():
        print(f"--- {model_cfg['title']} ---")
        stats_df = plot_single_model(model_key, model_cfg, output_dir, unified_max_info)
        if stats_df is not None:
            all_stats[model_key] = stats_df
        print()
    
    # Summary statistics
    print("=" * 60)
    print("Summary statistics")
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
        print(f"Summary statistics saved: {summary_path}")
        
        print("\nMean MAE of each model (across all features):")
        model_avg = summary_df.groupby('Model')['MAE_mean'].mean().sort_values()
        for m, v in model_avg.items():
            print(f"  {m}: {v:.6f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
