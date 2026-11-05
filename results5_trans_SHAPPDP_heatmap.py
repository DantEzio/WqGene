"""
results5_trans_SHAPPDP_heatmap.py - heatmap of SHAP / PDP sensitivity across transfer steps
=========================================================
Purpose:
    aggregate the per-step (checkpoint) results under results/Shapely for trans/ and results/PDP for trans/
    aggregate the gene sensitivity results into a matrix and show the SHAP / PDP values of the 21 n genes as a heatmap
    the evolution trend along the transfer learning process, plus a consistency measure between the two indicators.
- 3 rows x 2 cols: left column SHAP / right column PDP (ana_ano / ox)
  Row 3 is the SHAP vs PDP cosine similarity curve (2block scheme: ANA+ANO merged)
- x axis: actual WQ timestamps (from data_WQ/wq.xlsx)
- y axis: KEGG KO gene names (from gene_ref/Gene_info.xlsx)
- Colors: SHAP in blues (Blues), PDP in oranges (Oranges)
Input:
    results/Shapely for trans/step_{N}/SHAP_feature_importance_{zone}.csv
    results/PDP for trans/step_{N}/PDP_feature_importance_{zone}.csv
    data_WQ/wq.xlsx, gene_ref/Gene_info.xlsx, data_GeneM/n_gene_dynamic.xlsx
Output:
    results/shap_pdp_trans_heatmap_raw.png
Usage:
    python results5_trans_SHAPPDP_heatmap.py      # run from the project root
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime

# ============================================================
# Path configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
SHAP_BASE_DIR = os.path.join(ROOT_DIR, "results", "Shapely for trans")
PDP_BASE_DIR = os.path.join(ROOT_DIR, "results", "PDP for trans")
WQ_RAW_PATH = os.path.join(ROOT_DIR, "data_WQ", "wq.xlsx")
GENE_INFO_PATH = os.path.join(ROOT_DIR, "gene_ref", "Gene_info.xlsx")

# Step -> WQ row offset: dataset window_size = 40,
# Models are saved with i+1 (= window_end), so step_N maps to wq.iloc[N+39]
WINDOW_SIZE = 40

# ============================================================
# Normalization switch: True = normalize each time-step column, False = use the raw values
# ============================================================
USE_NORMALIZE = False

# The output filename is chosen automatically from USE_NORMALIZE
_OUTPUT_SUFFIX = "_normalized" if USE_NORMALIZE else "_raw"
OUTPUT_PATH = os.path.join(ROOT_DIR, "results", f"shap_pdp_trans_heatmap{_OUTPUT_SUFFIX}.png")


def load_wq_time_series(wq_path):
    """Read the time column from wq.xlsx and return a pandas Series"""
    print(f"Loading WQ time data from: {wq_path}")
    df = pd.read_excel(wq_path)
    if 'time' in df.columns:
        time_col = df['time']
    else:
        # Fallback: take the first column
        time_col = df.iloc[:, 0]
    print(f"  Loaded {len(time_col)} time records")
    return time_col


def load_gene_name_mapping(gene_info_path):
    """
    read the gene ID -> name mapping from Gene_info.xlsx.
    returns dict: {KO id: display name}
    """
    print(f"Loading gene info from: {gene_info_path}")
    df = pd.read_excel(gene_info_path)
    
    # Gene_info.xlsx columns: KO, L1, L2, Pathway, Description
    # Use the Description column as the display name, falling back to the KO id
    gene_id_to_name = {}
    for _, row in df.iterrows():
        ko_id = row['KO']
        # Description format, e.g. "K00844  HK; hexokinase [EC:2.7.1.1]"
        # Take the short name (the part before the first semicolon, or the Description itself)
        desc = str(row.get('Description', ''))
        if desc and desc.strip():
            # Try to extract a meaningful name part
            # Typical format: "K0xxxx  Name; description [EC:x.x.x.x]"
            name_part = desc.split(';')[0] if ';' in desc else desc
            # Strip the repeated KO id at the start
            name_part = name_part.strip()
            if name_part.startswith(ko_id):
                name_part = name_part[len(ko_id):].strip()
            gene_id_to_name[ko_id] = name_part if name_part else ko_id
        else:
            gene_id_to_name[ko_id] = ko_id
    
    print(f"  Loaded {len(gene_id_to_name)} gene name mappings")
    return gene_id_to_name


def discover_result_dirs(base_dir, prefix="SHAP"):
    """
    scan the result directory and return [(step_num, dir_path)] sorted ascending by number
    Folder name format: step_XXXXX
    prefix: distinguishes SHAP from PDP files
    """
    print(f"Scanning {prefix} result directory: {base_dir}")
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
            # Make sure the folder exists and contains the matching CSV
            if os.path.isdir(full_path) and any(
                f.startswith(file_prefix) and f.endswith(".csv")
                for f in os.listdir(full_path)
            ):
                entries.append((step_num, full_path))

    entries.sort(key=lambda x: x[0])
    if entries:
        print(f"  Found {len(entries)} {prefix} result folders: step_{entries[0][0]} ~ step_{entries[-1][0]}")
    return entries


def get_fixed_gene_order(first_shap_dir):
    """
    extract the fixed gene order (KO id list) from the first SHAP result,
    used to align the matrices of all later time steps.
    """
    csv_path = os.path.join(first_shap_dir, "SHAP_feature_importance_ana_ano.csv")  # 2block
    df = pd.read_csv(csv_path)
    gene_list = df['gene'].tolist()
    print(f"  Fixed gene order ({len(gene_list)} genes): {gene_list}")
    return gene_list


def load_and_build_matrix(result_dirs, model_type, fixed_genes, value_column, prefix="SHAP"):
    """
    build the [n_genes x n_steps] matrix of a given reactor type.

    Args:
        result_dirs: the [(step_num, path)] list returned by discover_result_dirs()
        model_type: 'ana' | 'ano' | 'ox'
        fixed_genes: fixed list of gene names (identical order)
        value_column: name of the value column to read from the CSV (mean_abs_SHAP or mean_PDP)
        prefix: "SHAP" or "PDP"

    Returns:
        np.ndarray: shape (n_genes, n_steps), plus the corresponding step_nums list
    """
    n_genes = len(fixed_genes)
    n_steps = len(result_dirs)
    matrix = np.zeros((n_genes, n_steps))
    step_nums = []

    # Build the gene -> row_index mapping
    gene_to_idx = {g: i for i, g in enumerate(fixed_genes)}

    for col_idx, (step_num, dir_path) in enumerate(result_dirs):
        csv_name = f"{prefix}_feature_importance_{model_type}.csv"
        csv_path = os.path.join(dir_path, csv_name)

        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, filling with zeros")
            continue

        df = pd.read_csv(csv_path)
        # Map gene names to fixed row indices (the CSV order may differ)
        for _, row in df.iterrows():
            gene = row['gene']
            val = row.get(value_column, row.iloc[1])
            if gene in gene_to_idx:
                matrix[gene_to_idx[gene], col_idx] = val

        step_nums.append(step_num)

    print(f"  Built [{n_genes} x {len(step_nums)}] {prefix} matrix for {model_type}")
    return matrix, step_nums


def normalize_per_step(matrices_dict):
    """
    min-max normalize the values of each time step (each column) to [0, 1].
    normalization is done per column so relative gene importance is comparable within each time step.

    Args:
        matrices_dict: {'ana': ndarray, 'ano': ndarray, 'ox': ndarray}
                        each with shape=(n_genes, n_steps)

    Returns:
        dict: normalized matrix with the same structure as the input
    """
    normalized = {}
    for mtype, mat in matrices_dict.items():
        n_genes, n_steps = mat.shape
        norm_mat = np.zeros_like(mat)
        for col in range(n_steps):
            col_data = mat[:, col]
            col_min = np.min(col_data)
            col_max = np.max(col_data)
            if col_max - col_min > 1e-12:
                norm_mat[:, col] = (col_data - col_min) / (col_max - col_min)
            else:
                norm_mat[:, col] = 0.0  # set to 0 when the whole column is constant
        normalized[mtype] = norm_mat
        print(f"  Normalized per-step for '{mtype}'")
    return normalized


def compute_cosine_similarity(vec_a, vec_b):
    """
    compute the cosine similarity of two vectors.

    Args:
        vec_a: np.ndarray, vector A
        vec_b: np.ndarray, vector B (same length as vec_a)

    Returns:
        float: cosine similarity in the range [-1, 1]
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot_product / (norm_a * norm_b)


def plot_shap_pdp_heatmap(shap_matrices, pdp_matrices, time_labels, gene_display_names,
                          output_path, use_normalize=False):
    """
    draw a 4x2 heatmap (left column SHAP Blues / right column PDP Oranges) plus the row-4 cosine similarity curve, then save.

    Args:
        shap_matrices: {'ana': ndarray, 'ano': ndarray, 'ox': ndarray} SHAP data
        pdp_matrices: {'ana': ndarray, 'ano': ndarray, 'ox': ndarray} PDP data
        time_labels: list of datetime/str, the time label of each step
        gene_display_names: list of str, gene display names (from Gene_info)
        output_path: path of the output figure
        use_normalize: bool, whether column normalization was applied (affects the color scale and labels)
    """
    # 2block: ANA+ANO merged
    types = ['ana_ano', 'ox']
    type_labels = {
        'ana_ano': 'Anaerobic & Anoxic zone gene',
        'ox': 'Oxic zone gene'
    }
    n_types = len(types)
    n_genes = len(gene_display_names)
    n_steps = len(time_labels)

    # ---- Figure size setup ----
    width_per_step = 0.06 if n_steps > 150 else (0.08 if n_steps > 80 else 0.10)
    max_fig_width = 32  # maximum width limit (two columns)
    fig_width = min(max(16, n_steps * width_per_step * 2), max_fig_width)
    
    # ---- Create the Figure and GridSpec ----
    fig = plt.figure(
        figsize=(fig_width, n_genes * 0.55 + 10),  # enlarge the overall figure
        constrained_layout=False
    )
    
    # Define the GridSpec: 3 rows (2 heatmap rows + 1 curve row) x 2 cols (2block: 4 rows became 3)
    gs = fig.add_gridspec(
        n_types + 1, 2,
        height_ratios=[1, 1, 1.0],  # row 3 is taller (cosine similarity)
        hspace=0.45,
        wspace=0.08
    )
    
    # Store all axes so the xticks can be set uniformly later
    all_axes = []

    # ---- Color scale setup ----
    if use_normalize:
        shap_vmin, shap_vmax = 0.0, 1.0
        pdp_vmin, pdp_vmax = 0.0, 1.0
        shap_cbar_label = 'Normalized SHAP'
        pdp_cbar_label = 'Normalized PDP'
        suptitle = 'N-Gene SHAP vs PDP Feature Importance Evolution (Per-Step Normalized)'
    else:
        all_shap = np.concatenate([shap_matrices[t].flatten() for t in types])
        all_pdp = np.concatenate([pdp_matrices[t].flatten() for t in types])
        shap_vmin, shap_vmax = 0.0, np.nanmax(all_shap) if len(all_shap) > 0 else 0.05
        pdp_vmin, pdp_vmax = 0.0, np.nanmax(all_pdp) if len(all_pdp) > 0 else 0.05
        shap_cbar_label = r'$|\phi_i^t|$ (SHAP)'
        pdp_cbar_label = 'PDP Value'
        suptitle = 'N-Gene SHAP vs PDP Feature Importance Evolution During Transfer Learning'

    shap_norm = Normalize(vmin=shap_vmin, vmax=shap_vmax)
    pdp_norm = Normalize(vmin=pdp_vmin, vmax=pdp_vmax)

    for idx, t in enumerate(types):
        # ===== Left: SHAP heatmap =====
        ax_shap = fig.add_subplot(gs[idx, 0])
        all_axes.append(ax_shap)
        
        shap_data = shap_matrices[t]

        im_shap = ax_shap.imshow(
            shap_data,
            aspect='auto',
            cmap='Blues',
            norm=shap_norm,
            origin='upper'
        )

        # Y axis: gene display names (shown only in the left column)
        ax_shap.set_yticks(range(n_genes))
        ax_shap.set_yticklabels(gene_display_names, fontsize=13)

        # Title
        ax_shap.set_title(f"{type_labels[t]} - SHAP values",
                         fontsize=22, fontweight='bold', pad=10)

        # Colorbar (placed tight against the heatmap)
        cbar_shap = plt.colorbar(im_shap, ax=ax_shap, fraction=0.03, pad=0.00)
        cbar_shap.ax.tick_params(labelsize=18)

        ax_shap.grid(False)
        
        # Add label (a) outside the top-left corner of row 1
        if idx == 0:
            pos = ax_shap.get_position()
            fig.text(pos.x0 + 0.02, pos.y1 + 0.06, '(a)',  # moved right, closer to the figure
                    transform=fig.transFigure,
                    fontsize=22, fontweight='bold', va='bottom', ha='left')

        # ===== Right: PDP heatmap =====
        ax_pdp = fig.add_subplot(gs[idx, 1])
        all_axes.append(ax_pdp)
        
        pdp_data = pdp_matrices[t]

        im_pdp = ax_pdp.imshow(
            pdp_data,
            aspect='auto',
            cmap='Oranges',
            norm=pdp_norm,
            origin='upper'
        )

        # Y axis: same gene order as the left column (labels suppressed to save space)
        ax_pdp.set_yticks(range(n_genes))
        ax_pdp.set_yticklabels([''] * n_genes)  # hide the y tick labels on the right

        # Title
        ax_pdp.set_title(f"{type_labels[t]} - PDP values",
                         fontsize=22, fontweight='bold', pad=10)

        # Colorbar (placed tight against the heatmap)
        cbar_pdp = plt.colorbar(im_pdp, ax=ax_pdp, fraction=0.03, pad=0.00)
        cbar_pdp.ax.tick_params(labelsize=18)

        ax_pdp.grid(False)
        
        # Add label (b) outside the top-left corner of the right panel in row 1
        if idx == 0:
            pos = ax_pdp.get_position()
            fig.text(pos.x0 + 0.02, pos.y1 + 0.06, '(b)',  # moved right, closer to the figure
                    transform=fig.transFigure,
                    fontsize=22, fontweight='bold', va='bottom', ha='left')

    # ===== Row 4: SHAP vs PDP cosine similarity curve (spans both columns) =====
    # Curve styling: color + line style + marker (2block: 3 curves became 2)
    line_styles = {
        'ana_ano': {'color': '#1f77b4', 'linestyle': '-',  'marker': 'o',  'markersize': 6, 'linewidth': 2.8, 'label': 'Anaerobic & Anoxic', 'markevery': max(1, n_steps // 15)},
        'ox':       {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^',  'markersize': 8, 'linewidth': 3.0, 'label': 'Oxic',             'markevery': max(1, n_steps // 10)}
    }
    
    x_positions = range(n_steps)
    
    # ---- Create spanning subplots with GridSpec ----
    ax_cosine = fig.add_subplot(gs[n_types, :])
    all_axes.append(ax_cosine)
    
    for t in types:
        shap_mat = shap_matrices[t]  # shape: (n_genes, n_steps)
        pdp_mat = pdp_matrices[t]   # shape: (n_genes, n_steps)
        
        # Compute the cosine similarity at each time step
        cosine_sims = np.zeros(n_steps)
        for step in range(n_steps):
            shap_vec = shap_mat[:, step]
            pdp_vec = pdp_mat[:, step]
            cosine_sims[step] = compute_cosine_similarity(shap_vec, pdp_vec)
        
        # Draw with a different plotting style
        style = line_styles[t]
        ax_cosine.plot(x_positions, cosine_sims,
                       color=style['color'],
                       linestyle=style['linestyle'],
                       linewidth=style['linewidth'],
                       marker=style['marker'],
                       markersize=style['markersize'],
                       label=style['label'],
                       markevery=style['markevery'])
    
    ax_cosine.set_ylabel('Cosine Similarity', fontsize=20)  # larger ylabel
    ax_cosine.tick_params(axis='y', labelsize=16)  # larger ytick
    ax_cosine.set_title('SHAP vs PDP Cosine Similarity (All Reactor Types)',
                        fontsize=22, fontweight='bold', pad=10)
    ax_cosine.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_cosine.axhline(y=1, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_cosine.axhline(y=-1, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_cosine.set_ylim(0.5, 1.5)
    ax_cosine.legend(loc='upper right', fontsize=13, framealpha=0.9)
    ax_cosine.grid(True, alpha=0.3)
    
    # Add label (c) outside the top-left corner of the last subplot
    pos = ax_cosine.get_position()
    fig.text(pos.x0 + 0.02, pos.y1 + 0.005, '(c)',  # moved right, closer to the figure
             transform=fig.transFigure,
             fontsize=22, fontweight='bold', va='bottom', ha='right')

    # X axis: time labels, always showing 5 key positions
    tick_indices = sorted(set([0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]))

    x_tick_labels = []
    for ti in tick_indices:
        tl = time_labels[ti]
        if isinstance(tl, (datetime, pd.Timestamp)):
            x_tick_labels.append(tl.strftime('%Y-%m-%d'))
        else:
            x_tick_labels.append(str(tl)[:10])

    # X axis: time labels; only rows 2 and 3 show xticks (smaller font), other panels hide them (2block: 4 rows became 3)
    n_total_axes = len(all_axes)
    # Row 2 holds the last three and second-to-last axes, row 3 the last axis
    show_xtick_indices = {n_total_axes - 3, n_total_axes - 2, n_total_axes - 1}  # Row2 left, Row2 right, Row3
    
    for ax_idx, ax in enumerate(all_axes):
        if ax_idx in show_xtick_indices:
            # Rows 3 and 4 show xticks with a smaller font
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(x_tick_labels, rotation=30, ha='right', fontsize=12)
        else:
            # Hide the xticks of the other panels
            ax.set_xticks([])
            ax.set_xticklabels([])

    fig.subplots_adjust(left=0.15, right=0.92, top=0.94, bottom=0.10, hspace=0.35, wspace=0.15)

    # Manually narrow the first two heatmap rows (Row 0 and Row 1)
    heatmap_axes = all_axes[:4]  # the first 4 axes: Row0(SHAP, PDP), Row1(SHAP, PDP)
    shrink_factor = 0.85  # narrow to 85% of the original
    for ax in heatmap_axes:
        pos = ax.get_position()
        new_width = pos.width * shrink_factor
        # Narrow the axis while keeping it centered (half the inset on each side)
        new_x0 = pos.x0 + (pos.width - new_width) / 2
        ax.set_position([new_x0, pos.y0, new_width, pos.height])

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nHeatmap saved to: {output_path}")


def main():
    """Main workflow"""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print("=" * 60)
    print("Trans-Learning Gene SHAP + PDP Heatmap Visualization")
    print("=" * 60)

    # 1. Load the WQ time series
    wq_time = load_wq_time_series(WQ_RAW_PATH)

    # 2. Load the gene name mapping
    gene_name_map = load_gene_name_mapping(GENE_INFO_PATH)

    # 3. Discover all SHAP and PDP result directories
    shap_dirs = discover_result_dirs(SHAP_BASE_DIR, prefix="SHAP")
    pdp_dirs = discover_result_dirs(PDP_BASE_DIR, prefix="PDP")
    
    if not shap_dirs:
        print("ERROR: No SHAP result directories found!")
        return
    
    if not pdp_dirs:
        print("WARNING: No PDP result directories found! Only SHAP will be plotted.")
        pdp_dirs = shap_dirs  # fallback: reuse the SHAP directory layout

    # 4. Obtain the common step list (intersection, keyed on SHAP)
    shap_step_nums = set(s[0] for s in shap_dirs)
    pdp_step_nums = set(p[0] for p in pdp_dirs)
    common_step_nums = sorted(shap_step_nums & pdp_step_nums)
    
    if not common_step_nums:
        print("WARNING: No common steps between SHAP and PDP! Using all SHAP steps.")
        common_step_nums = [s[0] for s in shap_dirs]
    
    # Keep only the directories of the shared steps
    shap_dirs_common = [(s, p) for s, p in shap_dirs if s in common_step_nums]
    pdp_dirs_common = [(s, p) for s, p in pdp_dirs if s in common_step_nums]
    
    print(f"\nCommon steps: {len(common_step_nums)} (step_{common_step_nums[0]} ~ step_{common_step_nums[-1]})")

    # 5. Obtain the fixed gene order (from the first SHAP result)
    fixed_genes = get_fixed_gene_order(shap_dirs_common[0][1])
    
    # Convert the gene IDs to display names
    gene_display_names = [gene_name_map.get(g, g) for g in fixed_genes]
    print(f"\nGene display names (first 5): {gene_display_names[:5]}")

    # 6. Extract the time label of each step
    time_labels = []
    for step_num, _ in shap_dirs_common:
        row_idx = step_num + WINDOW_SIZE - 1  # step_N -> wq.iloc[N+39]
        if row_idx < len(wq_time):
            time_labels.append(wq_time.iloc[row_idx])
        else:
            time_labels.append(f"N/A (idx={row_idx})")
    print(f"\nTime range: {time_labels[0]} ~ {time_labels[-1]}")

    # 7. Build the SHAP matrices (2block: ana_ano/ox)
    shap_matrices = {}
    for mtype in ['ana_ano', 'ox']:
        print(f"\nBuilding SHAP matrix for '{mtype}'...")
        mat, _ = load_and_build_matrix(shap_dirs_common, mtype, fixed_genes,
                                       value_column='mean_abs_SHAP', prefix="SHAP")
        shap_matrices[mtype] = mat

    # 8. Build the PDP matrices (2block: ana_ano/ox)
    pdp_matrices = {}
    for mtype in ['ana_ano', 'ox']:
        print(f"\nBuilding PDP matrix for '{mtype}'...")
        mat, _ = load_and_build_matrix(pdp_dirs_common, mtype, fixed_genes,
                                       value_column='mean_PDP', prefix="PDP")
        pdp_matrices[mtype] = mat

    # 9. Decide on column normalization via the USE_NORMALIZE switch
    if USE_NORMALIZE:
        print("\nNormalizing values per time step...")
        shap_matrices = normalize_per_step(shap_matrices)
        pdp_matrices = normalize_per_step(pdp_matrices)
    else:
        print("\nSkipping per-step normalization (USE_NORMALIZE = False)")

    # 10. Draw the heatmaps
    plot_shap_pdp_heatmap(shap_matrices, pdp_matrices, time_labels, gene_display_names,
                          OUTPUT_PATH, use_normalize=USE_NORMALIZE)
    print("\nDone!")


if __name__ == "__main__":
    main()
