"""
transfer-learning N-Gene PDP composite figure (results8_PDP)
=========================================================
recombine the key subplots of the two source figures into one 4-row x 2-col paper-level composite figure:

- Row 1 (2 panels): PDP heatmaps (evolution over time)
    source: the two panels of the PDP column (right side) of results/shap_pdp_trans_heatmap_raw.png
          (ana_ano PDP heatmap / ox PDP heatmap)
          Corresponding code: results5_trans_SHAPPDP_heatmap.py
- Row 2 (1 panel spanning two columns): SHAP vs PDP cosine similarity curve
    source: row 3 of the same figure (SHAP vs PDP Cosine Similarity)
          Corresponding code: results5_trans_SHAPPDP_heatmap.py
- Rows 3 and 4 (2 panels each, 4 in total): PDP/LEF horizontal bar charts (two windows compared)
    source: rows 1-2 and cols 3-4 of results/imp_pdp_barplot_comparison.png
          (ana_ano LEF 0813-0913 / 0913-1013, ox LEF 0813-0913 / 0913-1013)
          Corresponding code: results6_trans_IMP_PDP_barplot.py

Output: results/PDP_analysis_composite.png
Usage: python results8_PDP.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================
# Path configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR

# results5 data source (SHAP/PDP evolution matrix over transfer steps)
SHAP_BASE_DIR = os.path.join(ROOT_DIR, "results", "Shapely for trans")
PDP_TRANS_BASE_DIR = os.path.join(ROOT_DIR, "results", "PDP for trans")

# results6 data source (PDP summary, per time window)
PDP_SUMMARY_DIR = os.path.join(ROOT_DIR, "results", "PDP")

WQ_RAW_PATH = os.path.join(ROOT_DIR, "data_WQ", "wq.xlsx")
GENE_INFO_PATH = os.path.join(ROOT_DIR, "gene_ref", "Gene_info.xlsx")

OUTPUT_PATH = os.path.join(ROOT_DIR, "results", "PDP_analysis_composite.png")

# Step -> WQ row offset: dataset window_size = 40, step_N maps to wq.iloc[N+39]
WINDOW_SIZE = 40

# results6 time window configuration
WINDOWS = {
    '0813-0913': 'Window2',
    '0913-1013': 'Window3'
}
DATE_LABELS = {
    '0813-0913': 'August 13-September 13',
    '0913-1013': 'September 13-October 13'
}

# 2block: ANA+ANO merged
ZONE_TYPES = ['ana_ano', 'ox']
ZONE_LABELS = {
    'ana_ano': 'Anaerobic & Anoxic zone',
    'ox': 'Oxic zone'
}

# Colors (blues throughout)
PDP_HEAT_CMAP = 'Blues'
PDP_BAR_COLOR = '#2166ac'  # blues
IMP_COLOR = '#2166ac'
DYNAMIC_COLOR = '#1b7837'

# Cosine curve styling (blues with different shades)
LINE_STYLES = {
    'ana_ano': {'color': '#2166ac', 'linestyle': '-',  'marker': 'o', 'markersize': 6, 'linewidth': 2.8, 'label': 'Anaerobic & Anoxic'},
    'ox':       {'color': '#67a9cf', 'linestyle': '-.', 'marker': '^', 'markersize': 8, 'linewidth': 3.0, 'label': 'Oxic'},
}


# ============================================================
# Generic loading function
# ============================================================
def load_wq_time_series(wq_path):
    """Read the time column from wq.xlsx and return a pandas Series"""
    print(f"Loading WQ time data from: {wq_path}")
    df = pd.read_excel(wq_path)
    time_col = df['time'] if 'time' in df.columns else df.iloc[:, 0]
    print(f"  Loaded {len(time_col)} time records")
    return time_col


def load_gene_name_mapping(gene_info_path):
    """Load the gene ID -> display name mapping {KO: name}"""
    print(f"Loading gene info from: {gene_info_path}")
    df = pd.read_excel(gene_info_path)
    gene_id_to_name = {}
    for _, row in df.iterrows():
        ko_id = row['KO']
        desc = str(row.get('Description', ''))
        if desc and desc.strip():
            name_part = desc.split(';')[0] if ';' in desc else desc
            name_part = name_part.strip()
            if name_part.startswith(ko_id):
                name_part = name_part[len(ko_id):].strip()
            gene_id_to_name[ko_id] = name_part if name_part else ko_id
        else:
            gene_id_to_name[ko_id] = ko_id
    print(f"  Loaded {len(gene_id_to_name)} gene name mappings")
    return gene_id_to_name


def discover_result_dirs(base_dir, prefix="SHAP"):
    """Scan the results subdirectories and return [(step_num, dir_path)] sorted ascending"""
    entries = []
    pattern = re.compile(r'^step_(\d+)$')
    if not os.path.isdir(base_dir):
        print(f"  WARNING: Directory not found: {base_dir}")
        return []
    file_prefix = f"{prefix}_feature_importance_"
    for name in sorted(os.listdir(base_dir)):
        match = pattern.match(name)
        if match:
            step_num = int(match.group(1))
            full_path = os.path.join(base_dir, name)
            if os.path.isdir(full_path) and any(
                f.startswith(file_prefix) and f.endswith(".csv")
                for f in os.listdir(full_path)
            ):
                entries.append((step_num, full_path))
    entries.sort(key=lambda x: x[0])
    if entries:
        print(f"  Found {len(entries)} {prefix} result folders: "
              f"step_{entries[0][0]} ~ step_{entries[-1][0]}")
    return entries


def load_and_build_matrix(result_dirs, model_type, fixed_genes,
                          value_column, prefix="SHAP"):
    """Build the [n_genes x n_steps] matrix (rows aligned to fixed_genes)"""
    n_genes = len(fixed_genes)
    matrix = np.zeros((n_genes, len(result_dirs)))
    step_nums = []
    gene_to_idx = {g: i for i, g in enumerate(fixed_genes)}

    for col_idx, (step_num, dir_path) in enumerate(result_dirs):
        csv_path = os.path.join(dir_path, f"{prefix}_feature_importance_{model_type}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            gene = row['gene']
            val = row.get(value_column, row.iloc[1])
            if gene in gene_to_idx:
                matrix[gene_to_idx[gene], col_idx] = val
        step_nums.append(step_num)

    print(f"  Built [{n_genes} x {len(step_nums)}] {prefix} matrix for {model_type}")
    return matrix, step_nums


def compute_cosine_similarity(vec_a, vec_b):
    """Cosine similarity of two vectors"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot_product / (norm_a * norm_b)


def load_pdp_summary(model_type, window_key):
    """
    load the PDP summary data from results/PDP (used by the row-3 and row-4 bar charts).
    source: results6 load_data(PDP_BASE_DIR, mtype, window_key, 'LEF')
    """
    window_name = WINDOWS[window_key]
    filepath = os.path.join(PDP_SUMMARY_DIR, f"{model_type}_{window_name}_{window_key}.xlsx")
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found!")
        return None
    df = pd.read_excel(filepath)
    value_col = 'mean_PDP' if 'mean_PDP' in df.columns else df.columns[1]
    return df[['gene', value_col]].rename(columns={value_col: 'value'})


# ============================================================
# Rows 3 and 4: horizontal bar charts (single panel)
# ============================================================
def plot_pdp_barh(ax, model_type, window_key, fixed_genes,
                  gene_display_names, show_ytick=True):
    """Draw the PDP horizontal bar chart (following results6 plot_imp_pdp_barchart)"""
    df = load_pdp_summary(model_type, window_key)
    if df is None:
        ax.text(0.5, 0.5, f'{model_type} {window_key}\nnot found',
                ha='center', va='center', transform=ax.transAxes)
        return

    df_sorted = df.set_index('gene').loc[fixed_genes].reset_index()
    values = df_sorted['value'].values
    n_genes = len(fixed_genes)
    y_pos = range(n_genes)

    ax.barh(y_pos, values, color=PDP_BAR_COLOR, edgecolor='white',
            linewidth=0.5, height=0.7, alpha=0.75)
    ax.set_yticks(y_pos)
    if show_ytick:
        ax.set_yticklabels(gene_display_names, fontsize=10)
    else:
        ax.set_yticklabels([''] * n_genes)

    ax.set_title(f'{ZONE_LABELS[model_type]} PDP (LEF)\n({DATE_LABELS[window_key]})',
                 fontsize=15, fontweight='bold', pad=8)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=11)


# ============================================================
# Main plotting routine
# ============================================================
def main():
    """4 x 2 PDP composite figure"""
    print("=" * 60)
    print("PDP Composite Figure (results8_PDP)")
    print("=" * 60)

    # ---- 1. Base data ----
    wq_time = load_wq_time_series(WQ_RAW_PATH)
    gene_name_map = load_gene_name_mapping(GENE_INFO_PATH)

    # ---- 2. Scan the SHAP / PDP step directories ----
    shap_dirs = discover_result_dirs(SHAP_BASE_DIR, prefix="SHAP")
    pdp_dirs = discover_result_dirs(PDP_TRANS_BASE_DIR, prefix="PDP")
    if not shap_dirs:
        print("ERROR: No SHAP result directories found!")
        return
    if not pdp_dirs:
        print("WARNING: No PDP dirs, fallback to SHAP dirs")
        pdp_dirs = shap_dirs

    shap_step_nums = set(s[0] for s in shap_dirs)
    pdp_step_nums = set(p[0] for p in pdp_dirs)
    common_step_nums = sorted(shap_step_nums & pdp_step_nums)
    if not common_step_nums:
        common_step_nums = [s[0] for s in shap_dirs]
    shap_dirs = [(s, p) for s, p in shap_dirs if s in common_step_nums]
    pdp_dirs = [(s, p) for s, p in pdp_dirs if s in common_step_nums]
    n_steps = len(common_step_nums)
    print(f"\nCommon steps: {n_steps} (step_{common_step_nums[0]} ~ step_{common_step_nums[-1]})")

    # ---- 3. Fix the gene order (shared by all panels) ----
    # Use the PDP summary file (ana_ano 0813-0913) as the fixed order,
    # Keep the gene order of the bar rows identical to the heatmap rows
    ref_df = load_pdp_summary('ana_ano', '0813-0913')
    if ref_df is None:
        print("ERROR: reference PDP summary not found!")
        return
    fixed_genes = ref_df['gene'].tolist()
    gene_display_names = [gene_name_map.get(g, g) for g in fixed_genes]
    n_genes = len(fixed_genes)
    print(f"\nFixed gene order ({n_genes} genes), display first 5: {gene_display_names[:5]}")

    # ---- 4. Time labels ----
    time_labels = []
    for step_num, _ in shap_dirs:
        row_idx = step_num + WINDOW_SIZE - 1
        if row_idx < len(wq_time):
            time_labels.append(wq_time.iloc[row_idx])
        else:
            time_labels.append(f"N/A (idx={row_idx})")
    print(f"Time range: {time_labels[0]} ~ {time_labels[-1]}")

    # ---- 5. Build the SHAP / PDP matrices (ana_ano & ox) ----
    shap_matrices, pdp_matrices = {}, {}
    for mtype in ZONE_TYPES:
        mat, _ = load_and_build_matrix(shap_dirs, mtype, fixed_genes,
                                       value_column='mean_abs_SHAP', prefix="SHAP")
        shap_matrices[mtype] = mat
        mat, _ = load_and_build_matrix(pdp_dirs, mtype, fixed_genes,
                                       value_column='mean_PDP', prefix="PDP")
        pdp_matrices[mtype] = mat

    # ---- 6. PDP color scale (raw, same as results5 _raw) ----
    all_pdp = np.concatenate([pdp_matrices[t].flatten() for t in ZONE_TYPES])
    pdp_vmax = np.nanmax(all_pdp) if len(all_pdp) > 0 else 0.05
    pdp_norm = Normalize(vmin=0.0, vmax=pdp_vmax)

    # ---- 7. Create the canvas: 7 rows x 2 cols (3 spacers widen the gaps between rows 1-2, 2-3 and 3-4) ----
    # height_ratios: heatmap, spacer, cosine, spacer, ana bars, spacer, ox bars
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(7, 2, height_ratios=[2.4, 0.4, 0.9, 0.5, 2.2, 0.45, 2.2],
                          hspace=0.15, wspace=0.10)

    # Keep references to the axes
    heatmap_axes = []

    # ---------- Row 1: two PDP heatmaps ----------
    # (following the PDP column of results5 plot_shap_pdp_heatmap)
    for col_idx, mtype in enumerate(ZONE_TYPES):
        ax = fig.add_subplot(gs[0, col_idx])
        heatmap_axes.append(ax)
        pdp_data = pdp_matrices[mtype]
        im = ax.imshow(pdp_data, aspect='auto', cmap=PDP_HEAT_CMAP,
                       norm=pdp_norm, origin='upper')
        ax.set_yticks(range(n_genes))
        if col_idx == 0:
            ax.set_yticklabels(gene_display_names, fontsize=13)
        else:
            ax.set_yticklabels([''] * n_genes)
        ax.set_title(f"{ZONE_LABELS[mtype]} - PDP values (Transfer Learning)",
                     fontsize=18, fontweight='bold', pad=10)
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=15)
        ax.grid(False)

    # Label (a) at the top-left corner of row 1 (outside the axes)
    heatmap_axes[0].text(-0.03, 1.12, '(a)', transform=heatmap_axes[0].transAxes,
                         ha='right', va='bottom', fontsize=24, fontweight='bold')

    # ---------- Row 2: SHAP vs PDP cosine similarity (spanning two columns; gs[1] is the spacer above it) ----------
    ax_cosine = fig.add_subplot(gs[2, :])
    x_positions = range(n_steps)
    markevery_map = {
        'ana_ano': max(1, n_steps // 15),
        'ox': max(1, n_steps // 10),
    }
    for mtype in ZONE_TYPES:
        shap_mat = shap_matrices[mtype]
        pdp_mat = pdp_matrices[mtype]
        cosine_sims = np.zeros(n_steps)
        for step in range(n_steps):
            cosine_sims[step] = compute_cosine_similarity(
                shap_mat[:, step], pdp_mat[:, step])
        style = LINE_STYLES[mtype]
        # Append the series mean to the legend label: "label: mean" (two decimals)
        mean_val = np.mean(cosine_sims)
        label = f"{style['label']}: {mean_val:.2f}"
        ax_cosine.plot(x_positions, cosine_sims,
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=style['linewidth'], marker=style['marker'],
                       markersize=style['markersize'], label=label,
                       markevery=markevery_map[mtype])
    ax_cosine.set_ylabel('Cosine Similarity', fontsize=20)
    ax_cosine.tick_params(axis='y', labelsize=16)
    ax_cosine.set_title('SHAP vs PDP Cosine Similarity (All Reactor Types)',
                        fontsize=18, fontweight='bold', pad=10)
    ax_cosine.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_cosine.axhline(y=1, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_cosine.axhline(y=-1, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_cosine.set_ylim(0.5, 1.5)
    ax_cosine.legend(loc='upper right', fontsize=13, framealpha=0.9)
    ax_cosine.grid(True, alpha=0.3)

    # Label (b) at the top-left corner of row 2 (outside the axes)
    ax_cosine.text(-0.02, 1.12, '(b)', transform=ax_cosine.transAxes,
                   ha='right', va='bottom', fontsize=24, fontweight='bold')

    # ---------- Rows 4 and 6: PDP horizontal bar charts ----------
    # (gs rows 3 and 5 are spacers) Row4 = ana_ano (two windows), Row6 = ox (two windows)
    # Corresponds to rows 1-2 and cols 3-4 of imp_pdp_barplot_comparison.png
    windows_order = ['0813-0913', '0913-1013']
    bar_row_idx = [4, 6]  # actual gs rows of ana_ano / ox
    for row_off, mtype in enumerate(ZONE_TYPES):
        for col_idx, window_key in enumerate(windows_order):
            ax = fig.add_subplot(gs[bar_row_idx[row_off], col_idx])
            plot_pdp_barh(ax, mtype, window_key, fixed_genes,
                          gene_display_names, show_ytick=(col_idx == 0))
            if col_idx == 0 and row_off == 0:
                ax.set_ylabel('Gene', fontsize=12)
                # Label (c) at the top-left corner of row 3 (outside the axes)
                ax.text(-0.015, 1.04, '(c)', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=24, fontweight='bold')

    # ---------- Time ticks ----------
    tick_indices = sorted(set([0, n_steps // 4, n_steps // 2,
                               3 * n_steps // 4, n_steps - 1]))
    x_tick_labels = []
    for ti in tick_indices:
        tl = time_labels[ti]
        if isinstance(tl, (datetime, pd.Timestamp)):
            x_tick_labels.append(tl.strftime('%Y-%m-%d'))
        else:
            x_tick_labels.append(str(tl)[:10])

    # Row 1 heatmaps: keep the ticks but hide the labels (avoids overlapping the cosine row title)
    for ax in heatmap_axes:
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([])
    # Row 2 cosine curve: show the full date labels
    ax_cosine.set_xticks(tick_indices)
    ax_cosine.set_xticklabels(x_tick_labels, rotation=30, ha='right', fontsize=12)

    fig.subplots_adjust(left=0.16, right=0.94, top=0.97, bottom=0.06)
    fig.tight_layout(rect=[0.12, 0, 0.98, 1])

    # ---------- Save ----------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nComposite PDP figure saved to: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
