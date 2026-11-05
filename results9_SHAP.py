"""
transfer-learning N-Gene SHAP composite figure (results9_SHAP)
=========================================================
recombine the key subplots of the two source figures into one 3-row x 4-col paper-level composite figure:

- Row 1 (2 panels, each spanning 2 columns): SHAP heatmaps (evolution over time)
    source: the two panels of the SHAP column (left side) of results/shap_pdp_trans_heatmap_raw.png
          (ana_ano SHAP heatmap / ox SHAP heatmap)
          Corresponding code: results5_trans_SHAPPDP_heatmap.py
- Row 2 (4 panels): IMP horizontal bar charts (2 zones x 2 time windows)
    source: rows 1-2 and cols 1-2 of results/imp_pdp_barplot_comparison.png
          (ana_ano IMP 0813-0913 / 0913-1013, ox IMP 0813-0913 / 0913-1013)
          Corresponding code: results6_trans_IMP_PDP_barplot.py
- Row 3 (4 panels): MT (n_gene_dynamic) horizontal bar charts (2 zones x Sep/Oct)
    source: row 3, cols 1-4 of results/imp_pdp_barplot_comparison.png
          (ana_ano Sep / Oct, ox Sep / Oct)
          Corresponding code: results6_trans_IMP_PDP_barplot.py
- Row 4 (2 panels, each spanning 2 columns): IMP vs MT Top10 gene Venn diagrams (2 zones)
    the IMP sets of the Sep and Oct months of the same zone are unioned, and so are the MT sets, giving two new sets
    data source: results7_top10gene.py:
          IMP: take the top 10 from results/IMP/*.xlsx by importance
          MT:  take the top 10 from data_GeneM/n_gene_dynamic.xlsx by the Sep/Oct columns descending

Input:
    results/Shapely for trans/step_*/SHAP_feature_importance_{ana_ano,ox}.csv
    results/IMP/{zone}_{window}_{range}.xlsx          (IMP bar chart data)
    data_GeneM/n_gene_dynamic.xlsx                    (MT dynamic data, sheet: ana_ano/ox)
    data_WQ/wq.xlsx, gene_ref/Gene_info.xlsx

Output:
    results/SHAP_analysis_composite.png               composite figure
    results/top10genev2.xlsx                          IMP/MT gene set statistics table

Usage: python results9_SHAP.py        # run from the project root
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================
# Path configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR

# results5 data source (SHAP evolution matrix over transfer steps)
SHAP_BASE_DIR = os.path.join(ROOT_DIR, "results", "Shapely for trans")

# results6 data source (IMP summary / MT dynamic data, per time window)
IMP_BASE_DIR = os.path.join(ROOT_DIR, "results", "IMP")
DYNAMIC_DATA_PATH = os.path.join(ROOT_DIR, "data_GeneM", "n_gene_dynamic.xlsx")

WQ_RAW_PATH = os.path.join(ROOT_DIR, "data_WQ", "wq.xlsx")
GENE_INFO_PATH = os.path.join(ROOT_DIR, "gene_ref", "Gene_info.xlsx")

OUTPUT_PATH = os.path.join(ROOT_DIR, "results", "SHAP_analysis_composite.png")
TOP10_V2_PATH = os.path.join(ROOT_DIR, "results", "top10genev2.xlsx")

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

# Colors (identical to the source figure)
SHAP_HEAT_CMAP = 'Blues'      # results5 SHAP heatmap
IMP_COLOR = '#2166ac'         # results6 IMP blue palette
DYNAMIC_COLOR = '#67a9cf'     # MT light blue
VENN_COLORS = ('#2166ac', '#67a9cf')  # Venn diagram: IMP / MT

# Hand-drawn Venn diagram geometry (data coordinates):
# Horizontal ellipse radii (RX, RY) and half the center distance DX; a smaller DX means a larger overlap
VENN_RX, VENN_RY = 0.42, 0.30
VENN_DX = 0.22


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


def get_fixed_gene_order(first_shap_dir):
    """Extract the fixed gene order from the first SHAP result (as in results5)"""
    csv_path = os.path.join(first_shap_dir, "SHAP_feature_importance_ana_ano.csv")
    df = pd.read_csv(csv_path)
    gene_list = df['gene'].tolist()
    print(f"  Fixed gene order ({len(gene_list)} genes): {gene_list[:5]} ...")
    return gene_list


def load_and_build_shap_matrix(result_dirs, model_type, fixed_genes):
    """Build the SHAP [n_genes x n_steps] matrix (rows aligned to fixed_genes)"""
    n_genes = len(fixed_genes)
    matrix = np.zeros((n_genes, len(result_dirs)))
    gene_to_idx = {g: i for i, g in enumerate(fixed_genes)}

    for col_idx, (step_num, dir_path) in enumerate(result_dirs):
        csv_path = os.path.join(dir_path, f"SHAP_feature_importance_{model_type}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            gene = row['gene']
            val = row.get('mean_abs_SHAP', row.iloc[1])
            if gene in gene_to_idx:
                matrix[gene_to_idx[gene], col_idx] = val

    print(f"  Built [{n_genes} x {len(result_dirs)}] SHAP matrix for {model_type}")
    return matrix


def load_imp_summary(model_type, window_key):
    """
    load the IMP summary data from results/IMP (used by the row-2 bar charts).
    source: results6 load_data(IMP_BASE_DIR, mtype, window_key, 'IMP')
    """
    window_name = WINDOWS[window_key]
    filepath = os.path.join(IMP_BASE_DIR, f"{model_type}_{window_name}_{window_key}.xlsx")
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found!")
        return None
    df = pd.read_excel(filepath)
    value_col = 'mean_SHAP' if 'mean_SHAP' in df.columns else df.columns[1]
    return df[['gene', value_col]].rename(columns={value_col: 'value'})


def load_dynamic_data(dynamic_path, sheet_name):
    """
    load the n_gene_dynamic data (given sheet, used by the row-3 bar charts).
    source: results6 load_dynamic_data
    """
    print(f"Loading dynamic data from: {dynamic_path} (sheet: {sheet_name})")
    df = pd.read_excel(dynamic_path, sheet_name=sheet_name)
    df.columns = ['Sep', 'Oct', 'gene']
    df['Sep'] = pd.to_numeric(df['Sep'], errors='coerce')
    df['Oct'] = pd.to_numeric(df['Oct'], errors='coerce')
    print(f"  Loaded {len(df)} genes with Sep/Oct values")
    return df


# ============================================================
# Rows 2 and 3: horizontal bar charts (single panel)
# ============================================================
def plot_imp_barh(ax, model_type, window_key, fixed_genes,
                  gene_display_names, show_ytick=True):
    """Draw the IMP horizontal bar chart (following rows 1-2 and cols 1-2 of results6)"""
    df = load_imp_summary(model_type, window_key)
    if df is None:
        ax.text(0.5, 0.5, f'{model_type} IMP {window_key}\nnot found',
                ha='center', va='center', transform=ax.transAxes)
        return

    df_sorted = df.set_index('gene').loc[fixed_genes].reset_index()
    values = df_sorted['value'].values
    n_genes = len(fixed_genes)
    y_pos = range(n_genes)

    ax.barh(y_pos, values, color=IMP_COLOR, edgecolor='white',
            linewidth=0.5, height=0.82, alpha=0.75)
    ax.set_yticks(y_pos)
    if show_ytick:
        ax.set_yticklabels(gene_display_names, fontsize=15)
    else:
        ax.set_yticklabels([''] * n_genes)

    ax.set_title(f'{ZONE_LABELS[model_type]} IMP\n({DATE_LABELS[window_key]})',
                 fontsize=19, fontweight='bold', pad=6)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=14)


def plot_dynamic_barh(ax, dynamic_df, month, zone_label, fixed_genes,
                      gene_display_names, show_ytick=True):
    """Draw the MT (n_gene_dynamic) horizontal bar chart (following row 3 of results6)"""
    df_sorted = dynamic_df.set_index('gene').loc[fixed_genes].reset_index()
    values = df_sorted[month].values
    n_genes = len(fixed_genes)
    y_pos = range(n_genes)

    ax.barh(y_pos, values, color=DYNAMIC_COLOR, edgecolor='white',
            linewidth=0.5, height=0.82, alpha=0.75)
    ax.set_yticks(y_pos)
    if show_ytick:
        ax.set_yticklabels(gene_display_names, fontsize=15)
    else:
        ax.set_yticklabels([''] * n_genes)

    ax.set_title(f'MT: {zone_label} {month}', fontsize=19, fontweight='bold', pad=6)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=14)


# ============================================================
# Row 4: IMP vs MT Top10 gene Venn diagram (single panel)
# ============================================================
def get_imp_top10(model_type, window_key):
    """IMP Top10 gene list (as in results7: the file is already sorted by importance, take the first 10)"""
    df = load_imp_summary(model_type, window_key)
    if df is None:
        return []
    return [str(g).strip() for g in df['gene'].head(10).tolist() if pd.notna(g)]


def get_mt_top10(dynamic_df, month):
    """MT Top10 gene list (as in results7: sort by the Sep/Oct column descending and take the first 10)"""
    df_sorted = dynamic_df.sort_values(by=month, ascending=False)
    return [str(g).strip() for g in df_sorted['gene'].head(10).tolist() if pd.notna(g)]


def plot_venn_imp_mt(ax, imp_genes, mt_genes, zone_label, gene_name_map):
    """IMP vs MT Top10 gene Venn diagram (hand-drawn horizontal ellipses with a large overlap; names centered in each region)"""
    # Order-preserving de-duplication (accepts the union of several months)
    imp_genes = list(dict.fromkeys(imp_genes))
    mt_genes = list(dict.fromkeys(mt_genes))
    imp_set, mt_set = set(imp_genes), set(mt_genes)
    only_imp = [g for g in imp_genes if g not in mt_set]   # keep the IMP importance order
    only_mt = [g for g in mt_genes if g not in imp_set]    # keep the MT value order
    both = [g for g in imp_genes if g in mt_set]

    def names(genes):
        return '\n'.join(gene_name_map.get(g, g) for g in genes)

    # Two horizontal ellipses with a small center distance (2*VENN_DX) -> large overlap
    for cx, color in [(-VENN_DX, VENN_COLORS[0]), (VENN_DX, VENN_COLORS[1])]:
        ax.add_patch(Ellipse((cx, 0.0), VENN_RX * 2, VENN_RY * 2,
                             facecolor=color, alpha=0.5, edgecolor='none'))

    # Set labels (above the ellipses)
    ax.text(-VENN_DX, VENN_RY + 0.05, 'IMP', ha='center', va='bottom',
            fontsize=18, fontweight='bold')
    ax.text(VENN_DX, VENN_RY + 0.05, 'MT', ha='center', va='bottom',
            fontsize=18, fontweight='bold')

    # Gene name text (centered in each region: left crescent at -VENN_RX, intersection at (0,0), right crescent at +VENN_RX)
    ax.text(-VENN_RX, 0.0, names(only_imp), ha='center', va='center', fontsize=16)
    ax.text(0.0, 0.0, names(both), ha='center', va='center', fontsize=16)
    ax.text(VENN_RX, 0.0, names(only_mt), ha='center', va='center', fontsize=16)

    # Axis limits (leaving room above for the set labels)
    ax.set_xlim(-(VENN_DX + VENN_RX) - 0.03, (VENN_DX + VENN_RX) + 0.03)
    ax.set_ylim(-(VENN_RY + 0.04), VENN_RY + 0.13)
    ax.set_aspect('auto')
    ax.axis('off')

    ax.set_title(f'{zone_label}', fontsize=19, fontweight='bold', pad=6)


# ============================================================
# Export the gene set statistics table (top10genev2.xlsx)
# ============================================================
def save_top10_v2(venn_stats, output_path):
    """
    the IMP set (Sep union Oct), the MT set (Sep union Oct) and the shared genes (IMP intersection MT) of each zone
    organize them into a single table and save it as Excel.
    """
    rows = []
    for stat in venn_stats:
        for index_name, genes in [
            ('IMP (Sep + Oct)', stat['imp_union']),
            ('MT (Sep + Oct)', stat['mt_union']),
            ('Common (IMP & MT)', stat['common']),
        ]:
            rows.append({
                'Zone': stat['zone_label'],
                'Index': index_name,
                'Count': len(genes),
                'Gene ID': '; '.join(genes),
                'Gene name': '; '.join(stat['gene_name_map'].get(g, g) for g in genes),
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"\nTop10 gene v2 table saved to: {output_path}")
    print(df[['Zone', 'Index', 'Count']].to_string(index=False))


# ============================================================
# Main plotting routine
# ============================================================
def main():
    """3 x 4 SHAP composite figure"""
    print("=" * 60)
    print("SHAP Composite Figure (results9_SHAP)")
    print("=" * 60)

    # ---- 1. Base data ----
    wq_time = load_wq_time_series(WQ_RAW_PATH)
    gene_name_map = load_gene_name_mapping(GENE_INFO_PATH)

    # ---- 2. Scan the SHAP step directories ----
    shap_dirs = discover_result_dirs(SHAP_BASE_DIR, prefix="SHAP")
    if not shap_dirs:
        print("ERROR: No SHAP result directories found!")
        return
    n_steps = len(shap_dirs)

    # ---- 3. Fix the gene order (shared by all panels, as in results5) ----
    fixed_genes = get_fixed_gene_order(shap_dirs[0][1])
    gene_display_names = [gene_name_map.get(g, g) for g in fixed_genes]
    n_genes = len(fixed_genes)
    print(f"\nGene display names (first 5): {gene_display_names[:5]}")

    # ---- 4. Time labels (step_N -> wq.iloc[N+39]) ----
    time_labels = []
    for step_num, _ in shap_dirs:
        row_idx = step_num + WINDOW_SIZE - 1
        if row_idx < len(wq_time):
            time_labels.append(wq_time.iloc[row_idx])
        else:
            time_labels.append(f"N/A (idx={row_idx})")
    print(f"Time range: {time_labels[0]} ~ {time_labels[-1]}")

    # ---- 5. Build the SHAP matrices (ana_ano & ox, raw color scale) ----
    shap_matrices = {}
    for mtype in ZONE_TYPES:
        shap_matrices[mtype] = load_and_build_shap_matrix(shap_dirs, mtype, fixed_genes)

    all_shap = np.concatenate([shap_matrices[t].flatten() for t in ZONE_TYPES])
    shap_vmax = np.nanmax(all_shap) if len(all_shap) > 0 else 0.05
    shap_norm = Normalize(vmin=0.0, vmax=shap_vmax)

    # ---- 6. Load the MT dynamic data ----
    dynamic_dfs = {
        'ana_ano': load_dynamic_data(DYNAMIC_DATA_PATH, 'ana_ano'),
        'ox': load_dynamic_data(DYNAMIC_DATA_PATH, 'ox'),
    }

    # ---- 7. Create the canvas: 5 rows x 4 cols (row 1 panels span 2 cols, row 2 is a spacer) ----
    # constrained_layout handles spanning subplots and the colorbar (tight_layout cannot and overlaps them)
    # row0: SHAP heatmap, row1: spacer, row2: IMP, row3: MT, row4: Venn (same height as the MT row)
    # Smaller canvas with tighter hspace/wspace -> less blank space inside and between panels; fonts look larger
    fig = plt.figure(figsize=(25, 24), layout='constrained')
    gs = fig.add_gridspec(5, 4, height_ratios=[3.2, 0.1, 2.7, 2.7, 2.0])
    fig.get_layout_engine().set(wspace=0.03, hspace=0.04)

    # ---------- Row 1: two SHAP heatmaps (each spanning 2 columns) ----------
    # (following the SHAP column of results5 plot_shap_pdp_heatmap)
    heatmap_axes = []
    for col_idx, mtype in enumerate(ZONE_TYPES):
        ax = fig.add_subplot(gs[0, col_idx * 2:col_idx * 2 + 2])
        heatmap_axes.append(ax)
        shap_data = shap_matrices[mtype]
        im = ax.imshow(shap_data, aspect='auto', cmap=SHAP_HEAT_CMAP,
                       norm=shap_norm, origin='upper')
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(gene_display_names, fontsize=17)
        ax.set_title(f"{ZONE_LABELS[mtype]} gene\nSHAP values (WqGene-LL)",
                     fontsize=22, fontweight='bold', pad=8)
        ax.grid(False)
        if col_idx == 0:
            # Label (a) at the top-left corner of row 1 (outside the axes)
            ax.text(-0.01, 1.02, '(a)', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=20, fontweight='bold')

    # Both heatmaps share one colorbar (same norm, identical color scale)
    cbar = fig.colorbar(im, ax=heatmap_axes, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=17)

    # Heatmap x axis: WQ time labels (5 key positions)
    tick_indices = sorted(set([0, n_steps // 4, n_steps // 2,
                               3 * n_steps // 4, n_steps - 1]))
    x_tick_labels = []
    for ti in tick_indices:
        tl = time_labels[ti]
        if isinstance(tl, (datetime, pd.Timestamp)):
            x_tick_labels.append(tl.strftime('%Y-%m-%d'))
        else:
            x_tick_labels.append(str(tl)[:10])
    for ax in fig.axes:
        if ax.get_images():  # heatmap axes
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(x_tick_labels, rotation=30, ha='right', fontsize=15)

    # ---------- Row 2: IMP horizontal bar charts (4 panels) ----------
    # (source: rows 1-2 and cols 1-2 of imp_pdp_barplot_comparison.png)
    # Column order: ana_ano 0813-0913, ana_ano 0913-1013, ox 0813-0913, ox 0913-1013
    imp_layout = [
        ('ana_ano', '0813-0913'),
        ('ana_ano', '0913-1013'),
        ('ox', '0813-0913'),
        ('ox', '0913-1013'),
    ]
    for col_idx, (mtype, window_key) in enumerate(imp_layout):
        ax = fig.add_subplot(gs[2, col_idx])
        plot_imp_barh(ax, mtype, window_key, fixed_genes,
                      gene_display_names, show_ytick=(col_idx == 0 or col_idx == 2))
        # Show only a few x ticks to avoid crowding
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if col_idx == 0:
            # Label (b) at the top-left corner of row 2 (outside the axes)
            ax.text(-0.06, 1.10, '(b)', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=20, fontweight='bold')

    # ---------- Row 3: MT dynamic-data bar charts (4 panels) ----------
    # (source: row 3, cols 1-4 of imp_pdp_barplot_comparison.png)
    # Column order: ana_ano Sep, ana_ano Oct, ox Sep, ox Oct
    mt_layout = [
        ('ana_ano', 'Sep', 'Ana&ano'),
        ('ana_ano', 'Oct', 'Ana&ano'),
        ('ox', 'Sep', 'Ox'),
        ('ox', 'Oct', 'Ox'),
    ]
    for col_idx, (mtype, month, zone_label) in enumerate(mt_layout):
        ax = fig.add_subplot(gs[3, col_idx])
        plot_dynamic_barh(ax, dynamic_dfs[mtype], month, zone_label, fixed_genes,
                          gene_display_names, show_ytick=(col_idx == 0 or col_idx == 2))

    # ---------- Row 4: IMP vs MT Top10 gene Venn diagrams (2 panels, each spanning 2 columns) ----------
    # (data source: results7_top10gene.py; for the same zone the Sep and Oct
    # IMP set union plus MT set union form the two new sets)
    MT_ZONE_LABELS = {'ana_ano': 'Ana&ano', 'ox': 'Ox'}
    month_to_window = {'Sep': '0813-0913', 'Oct': '0913-1013'}
    venn_stats = []  # gene set statistics per zone (used to export top10genev2.xlsx)
    for col_idx, mtype in enumerate(ZONE_TYPES):
        ax = fig.add_subplot(gs[4, col_idx * 2:col_idx * 2 + 2])
        # Union of the IMP / MT sets of different months, keeping order and de-duplicating
        imp_union = list(dict.fromkeys(
            get_imp_top10(mtype, month_to_window['Sep'])
            + get_imp_top10(mtype, month_to_window['Oct'])))
        mt_union = list(dict.fromkeys(
            get_mt_top10(dynamic_dfs[mtype], 'Sep')
            + get_mt_top10(dynamic_dfs[mtype], 'Oct')))
        common = [g for g in imp_union if g in set(mt_union)]
        venn_stats.append({
            'zone_label': ZONE_LABELS[mtype],
            'imp_union': imp_union,
            'mt_union': mt_union,
            'common': common,
            'gene_name_map': gene_name_map,
        })
        plot_venn_imp_mt(ax, imp_union, mt_union,
                         ZONE_LABELS[mtype], gene_name_map)
        if col_idx == 0:
            # Label (c) at the top-left corner of row 4 (outside the axes)
            ax.text(-0.01, 1.04, '(c)', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=20, fontweight='bold')

    # ---------- Save ----------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nComposite SHAP figure saved to: {OUTPUT_PATH}")

    # ---------- Export the gene set statistics table ----------
    save_top10_v2(venn_stats, TOP10_V2_PATH)

    print("Done!")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
