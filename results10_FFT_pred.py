"""
results10_FFT_pred.py - prediction curves of the ablation models (all 12 output dims)
=========================================================
Purpose:
    12 output dims drawn as 4 rows x 3 cols (4 regions x 3 indicators),
    each panel overlays three curves: truth + WqGene-LL + the water-quality-only model,
    used to visualize the drop in fitting ability after "removing the gene input".
Layout notes (consistent with plot_all_features_comparison.py):
- No subplot titles; y labels are feature names (with mg/L units, de-normalized)
- Zone row labels rotated 90 degrees on the left (Anaerobic / Anoxic / Oxic / Effluent)
- x axis is date (mm-dd ticks); only the bottom row shows the Date label
- Legend at the top-right of each panel; large centered title at the top of the figure
- Data are cut from PLOT_START_DATE (2025-07-13) so fluctuations are easier to read
Input:
    prediction_paths: step1_*/.../prediction.csv (truth columns plus each model's prediction columns)
    data_WQ/wq.xlsx (time column, used for the date axis)
Output:
    results/results10_FFT_pred.png
Usage:
    python results10_FFT_pred.py              # run from the project root
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import pandas as pd
from datetime import datetime, timedelta

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================
# Font setup (consistent with plot_all_features_comparison.py)
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
# Model configuration: the full model plus the wq-only ablation model (per-step transfer learning)
# ============================================================
prediction_paths = {
    'FullTL': 'step1_gene_model_trans_2block_FullTL/results/gene_model_transfer_learning/prediction.csv',
    'AEGeneRemoveWithTime': 'step1_gene_model_trans_2block_FullTL - AEGeneRemoveWithTime/results/gene_model_transfer_learning/prediction.csv',
}

display_names = {
    'FullTL': 'WqGene-LL',
    'AEGeneRemoveWithTime': 'WqGene-LL (only water quality input)'
}
# Colors: truth in solid black, full model in dashed dark blue, wq-only in dashed red
model_colors = {
    'FullTL': '#08306b',
    'AEGeneRemoveWithTime': '#a50f15',
}
real_color = 'black'

# Start date of the plots (consistent with plot_all_features_comparison.py)
PLOT_START_DATE = datetime(2025, 7, 13)

# Output regions (4 rows) and indicators (3 columns)
REGION_ORDER = ['ANA', 'ANO', 'OX', 'EFF']
REGION_DISPLAY_NAMES = {
    'ANA': 'Anaerobic',
    'ANO': 'Anoxic',
    'OX': 'Oxic',
    'EFF': 'Effluent',
}
# Indicator display names (with sub/superscripts): DO, NH4+, NO3-
FEAT_DISPLAY_NAMES = {
    1: r'DO (mg/L)',
    2: r'NH$_{4}^{+}$ (mg/L)',
    3: r'NO$_{3}^{-}$ (mg/L)',
}
# Special EFF feature names: COD, TN, NH4+
EFF_FEAT_NAMES = {
    1: r'COD (mg/L)',
    2: r'TN (mg/L)',
    3: r'NH$_{4}^{+}$ (mg/L)',
}

# De-normalization parameters: original max value of each (region, feat_idx) (Min-Max, raw_min=0)
# (same as DENORM_MAX in plot_all_features_comparison.py)
DENORM_MAX = {
    ('ANA', 1): 0.30,   # DO-ANA
    ('ANA', 2): 52.73,  # AN-ANA
    ('ANA', 3): 1.69,   # NN-ANA
    ('ANO', 1): 0.70,   # DO-ANO
    ('ANO', 2): 50.69,  # AN-ANO
    ('ANO', 3): 10.72,  # NN-ANO
    ('OX', 1): 5.89,    # DO-OX
    ('OX', 2): 15.58,   # AN-OX
    ('OX', 3): 29.14,   # NN-OX
    ('EFF', 1): 23.75,  # COD-EFF
    ('EFF', 2): 13.81,  # TN-EFF
    ('EFF', 3): 3.39,   # AN-EFF
}


def load_dates():
    """Read the list of actual dates from wq.xlsx"""
    try:
        df = pd.read_excel('data_WQ/wq.xlsx')
        return df['time'].tolist()
    except Exception as e:
        print(f"Warning: cannot read wq.xlsx: {e}")
        return None


def format_date_short(date_obj):
    """Format a date for tick labels (short mm-dd form)"""
    if hasattr(date_obj, 'strftime'):
        return date_obj.strftime('%m-%d')
    try:
        return str(pd.to_datetime(date_obj))[5:10]
    except:
        return str(date_obj)


# ============================================================
# Read the prediction results
# ============================================================
model_dfs = {}
for model_name, path in prediction_paths.items():
    if os.path.exists(path):
        model_dfs[model_name] = pd.read_csv(path)
        print(f"Loaded {display_names[model_name]} predictions: {path} "
              f"({len(model_dfs[model_name])} samples)")
    else:
        print(f"Warning: Prediction file not found at {path}")

if len(model_dfs) == 0:
    raise SystemExit("ERROR: No prediction files found, abort.")

# TL models number time_step from 0, so an offset is needed to align with the wq.xlsx index
TL_OFFSET = 4263
first_model = list(model_dfs.keys())[0]
time_steps_rel = model_dfs[first_model]['time_step'].values
time_steps_abs = time_steps_rel + TL_OFFSET

# Date series
actual_dates = load_dates()
dates = []
for ts in time_steps_abs:
    if actual_dates is not None and ts < len(actual_dates):
        dates.append(actual_dates[ts])
    else:
        dates.append(PLOT_START_DATE + timedelta(minutes=15 * int(ts)))

# ---- Cut the series starting from PLOT_START_DATE ----
n_points = min(len(df) for df in model_dfs.values())
start_idx = 0
for i, d in enumerate(dates):
    try:
        dt = d if hasattr(d, 'strftime') else pd.to_datetime(d)
        if hasattr(dt, 'timestamp') and dt >= PLOT_START_DATE:
            start_idx = i
            break
    except:
        pass

plot_len = n_points - start_idx
x_pos = np.arange(plot_len)
dates_plot = dates[start_idx:n_points]
print(f"\nPlotting from {PLOT_START_DATE.date()}: {plot_len} points per feature")

# Console statistics: MSE of the two models per output dim (normalized values, full series)
for region in REGION_ORDER:
    for feat_idx in (1, 2, 3):
        real_col = f'{region}_real_{feat_idx}'
        pred_col = f'{region}_pred_{feat_idx}'
        real = model_dfs[first_model][real_col].values[:n_points]
        line = f"  {region}_{feat_idx:<2d}"
        for model_name, df in model_dfs.items():
            pred = df[pred_col].values[:n_points]
            mse = float(np.mean((pred - real) ** 2))
            line += f" | {display_names[model_name]} MSE: {mse:.6f}"
        print(line)

# ============================================================
# Create a 4-row x 3-col figure (12 output dims), laid out like plot_all_features_comparison.py
# ============================================================
n_rows = len(REGION_ORDER)
n_cols = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 17))

for row, region in enumerate(REGION_ORDER):
    for col_idx, feat_idx in enumerate((1, 2, 3)):
        ax = axes[row, col_idx]

        real_col = f'{region}_real_{feat_idx}'
        pred_col = f'{region}_pred_{feat_idx}'
        if real_col not in model_dfs[first_model].columns:
            ax.text(0.5, 0.5, f'{real_col}\nnot found',
                    ha='center', va='center', fontsize=tick_size,
                    transform=ax.transAxes)
            continue

        # ---- De-normalization (Min-Max, raw_min=0, same as the reference script) ----
        raw_max = DENORM_MAX.get((region, feat_idx), 1.0)
        real_vals = model_dfs[first_model][real_col].values[start_idx:n_points] * raw_max

        # ---- Ground truth (black solid line, semi-transparent) ----
        ax.plot(x_pos, real_vals, color=real_color, linewidth=1.3,
                alpha=0.45, label='Observed', zorder=3)

        # ---- Predicted values of each model (dashed) ----
        for model_name, df in model_dfs.items():
            pred_vals = df[pred_col].values[start_idx:n_points] * raw_max
            ax.plot(x_pos, pred_vals, color=model_colors[model_name],
                    linewidth=1.3, alpha=0.85, linestyle='--',
                    label=display_names[model_name], zorder=3)

        # ---- Axis setup (no title, y label uses the feature name) ----
        ax.grid(True, alpha=0.15)
        if region == 'EFF':
            ylabel = EFF_FEAT_NAMES.get(feat_idx, f'Feat_{feat_idx}')
        else:
            ylabel = FEAT_DISPLAY_NAMES.get(feat_idx, f'Feat_{feat_idx}')
        ax.set_ylabel(ylabel, fontsize=tick_size + 1)

        # Show the X axis labels only in the bottom row
        if row == n_rows - 1:
            ax.set_xlabel('Date', fontsize=tick_size + 1)

        # X axis date ticks (4-5 evenly spaced mm-dd labels)
        n_pts = len(dates_plot)
        n_ticks = min(5, max(4, n_pts))
        tick_positions = [int(i * (n_pts - 1) / (n_ticks - 1)) for i in range(n_ticks)]
        tick_labels = [format_date_short(dates_plot[i]) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=tick_size - 1)
        ax.tick_params(axis='y', labelsize=tick_size - 1)

        # Legend at the top-right of every panel
        ax.legend(loc='upper right', fontsize=legend_size,
                  framealpha=0.88, edgecolor='gray')

# Region row labels (rotated 90 degrees on the left)
for row, region in enumerate(REGION_ORDER):
    axes[row, 0].annotate(REGION_DISPLAY_NAMES[region],
                          xy=(-0.24, 0.5),
                          xycoords=axes[row, 0].transAxes,
                          fontsize=label_size, fontweight='bold',
                          ha='right', va='center', rotation=90)

# Large centered title at the top
fig.text(0.5, 0.995,
         f'{display_names["FullTL"]} vs {display_names["AEGeneRemoveWithTime"]}',
         transform=fig.transFigure,
         fontsize=label_size + 4, fontweight='bold',
         ha='center', va='top')

plt.tight_layout(rect=[0.16, 0, 0.98, 0.97])

# ============================================================
# Save results
# ============================================================
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'results10_FFT_pred.png')
plt.savefig(output_path, dpi=250, bbox_inches='tight')
plt.close()
print(f"\n[SUCC] Figure saved: {output_path}")

print("\n[SUCC] results10_FFT_pred completed successfully!")
