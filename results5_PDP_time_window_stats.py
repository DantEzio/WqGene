"""
results5_PDP_time_window_stats.py - per-time-window gene PDP sensitivity summary (LEF)
=========================================================
Purpose:
    bucket the per-checkpoint PDP results into three time windows by actual date,
    average PDP of each gene inside the window, used as that window's gene sensitivity (LEF).
    results are saved in descending order of importance for results6_trans_IMP_PDP_barplot.py / results7_top10gene.py.

Time windows:
  - Window 1: Jul 13 - Aug 13  (Window1_0713-0813)
  - Window 2: Aug 13 - Sep 13  (Window2_0813-0913)
  - Window 3: Sep 13 - Oct 13 (Window3_0913-1013)

Input:
    results/PDP for trans/step_{N}/PDP_feature_importance_{zone}.csv
    data_WQ/wq.xlsx (time column, used to map step_N -> actual date)
    gene_ref/Gene_info.xlsx (KEGG -> gene name mapping)
Output:
    results/PDP/{zone}_{window}_{start-end}.xlsx
    (one Excel file per reactor type x time window, with gene / mean_PDP columns)
Usage:
    python results5_PDP_time_window_stats.py        # run from the project root
Note:
    The SHAP version is results5_SHAP_time_window_stats.py, with identical logic
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================
# Path configuration (consistent with results5, only the data source and output dir differ)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WQ_PATH = os.path.join(BASE_DIR, "data_WQ", "wq.xlsx")
PDP_BASE_DIR = os.path.join(BASE_DIR, "results", "PDP for trans")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "PDP")

# Time window definition (left-closed, right-open)
TIME_WINDOWS = [
    ("2025-07-13", "2025-08-13", "Window1_0713-0813"),
    ("2025-08-13", "2025-09-13", "Window2_0813-0913"),
    ("2025-09-13", "2025-10-13", "Window3_0913-1013"),
]

# Step -> WQ row offset: dataset window_size = 40,
# Models are saved with i+1 (= window_end), so step_N maps to wq.iloc[N+39]
WINDOW_SIZE = 40

MODEL_TYPES = ["ana_ano", "ox"]  # 2block: ANA+ANO merged


# ============================================================
# Data loading function (reuses the results5 logic, switched to PDP)
# ============================================================

def load_wq_time_series(wq_path):
    """Read the time column from wq.xlsx"""
    print(f"Loading WQ time data from: {wq_path}")
    df = pd.read_excel(wq_path)
    if 'time' in df.columns:
        time_col = df['time']
    else:
        time_col = df.iloc[:, 0]
    print(f"  Loaded {len(time_col)} time records")
    return time_col


def discover_pdp_dirs(base_dir):
    """Scan the PDP for trans directory and return [(step_num, dir_path)] sorted ascending"""
    print(f"Scanning PDP result directory: {base_dir}")
    entries = []
    pattern = re.compile(r'^step_(\d+)$')

    if not os.path.isdir(base_dir):
        print(f"  ERROR: Directory not found: {base_dir}")
        return []

    for name in sorted(os.listdir(base_dir)):
        match = pattern.match(name)
        if match:
            step_num = int(match.group(1))
            full_path = os.path.join(base_dir, name)
            if os.path.isdir(full_path) and any(
                f.startswith("PDP_feature_importance_") and f.endswith(".csv")
                for f in os.listdir(full_path)
            ):
                entries.append((step_num, full_path))

    entries.sort(key=lambda x: x[0])
    print(f"  Found {len(entries)} PDP result folders")
    return entries


def get_fixed_gene_order(first_pdp_dir):
    """Extract the fixed gene order from the first PDP result"""
    csv_path = os.path.join(first_pdp_dir, "PDP_feature_importance_ana_ano.csv")  # 2block
    df = pd.read_csv(csv_path)
    gene_list = df['gene'].tolist()
    print(f"  Fixed gene order ({len(gene_list)} genes)")
    return gene_list


def load_and_build_matrix(pdp_dirs, model_type, fixed_genes):
    """Build the [n_genes x n_steps] PDP matrix"""
    n_genes = len(fixed_genes)
    n_steps = len(pdp_dirs)
    matrix = np.zeros((n_genes, n_steps))
    step_nums = []
    gene_to_idx = {g: i for i, g in enumerate(fixed_genes)}

    for col_idx, (step_num, dir_path) in enumerate(pdp_dirs):
        csv_name = f"PDP_feature_importance_{model_type}.csv"
        csv_path = os.path.join(dir_path, csv_name)

        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            gene = row['gene']
            pdp_val = row.get('mean_PDP', row.iloc[1])
            if gene in gene_to_idx:
                matrix[gene_to_idx[gene], col_idx] = pdp_val

        step_nums.append(step_num)

    print(f"  Built [{n_genes} x {len(step_nums)}] matrix for {model_type}")
    return matrix, step_nums


# ============================================================
# Main logic
# ============================================================

def main():
    # Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # 1. Load the time series
    dates = load_wq_time_series(WQ_PATH)
    # Convert to datetime
    date_list = []
    for d in dates:
        if isinstance(d, (datetime, pd.Timestamp)):
            date_list.append(pd.to_datetime(d))
        else:
            date_list.append(pd.to_datetime(str(d)))

    # 2. Discover all PDP directories
    pdp_dirs = discover_pdp_dirs(PDP_BASE_DIR)
    if not pdp_dirs:
        print("ERROR: No PDP directories found!")
        return

    # 3. Obtain the fixed gene order
    first_dir = pdp_dirs[0][1]
    fixed_genes = get_fixed_gene_order(first_dir)

    # 4. Build a matrix for each reactor type
    matrices = {}
    step_nums_all = {}
    for model_type in MODEL_TYPES:
        mat, steps = load_and_build_matrix(pdp_dirs, model_type, fixed_genes)
        matrices[model_type] = mat
        step_nums_all[model_type] = steps

    n_steps_total = len(pdp_dirs)

    # 4.5 Compute the actual time of each step (same as the heatmap script: step_N -> wq.iloc[N+39])
    step_times = []
    for step_num, _ in pdp_dirs:
        row_idx = step_num + WINDOW_SIZE - 1
        if row_idx < len(date_list):
            step_times.append(date_list[row_idx])
        else:
            step_times.append(None)
    print(f"\nStep time range: {step_times[0]} ~ {step_times[-1]}")

    # 5. Compute statistics for each time window
    # Also collect the top gene info of each window x type for the final summary
    summary_records = []  # [(time_window, model_type, top_gene, max_IMP)]

    for win_start_str, win_end_str, win_name in TIME_WINDOWS:
        win_start = pd.to_datetime(win_start_str)
        win_end = pd.to_datetime(win_end_str)

        print(f"\n{'='*60}")
        print(f"Time Window: {win_name} ({win_start_str} ~ {win_end_str})")
        print(f"{'='*60}")

        # Find the step column indices belonging to this time window (using each step's actual time)
        window_indices = []
        for i, t in enumerate(step_times):
            if t is not None and win_start <= t < win_end:
                window_indices.append(i)

        print(f"  Found {len(window_indices)} time points in this window")

        if not window_indices:
            print("  WARNING: No data points in this window, skipping.")
            continue

        # 6. Average over each reactor type
        for model_type in MODEL_TYPES:
            mat = matrices[model_type]

            # Take the columns of this time window and average each row (gene)
            window_data = mat[:, window_indices]  # shape: (n_genes, n_window_steps)
            gene_means = np.mean(window_data, axis=1)  # shape: (n_genes,)

            # Build the DataFrame
            df_result = pd.DataFrame({
                'gene': fixed_genes,
                'mean_PDP': gene_means
            })

            # Sort by mean_PDP descending
            df_result = df_result.sort_values('mean_PDP', ascending=False).reset_index(drop=True)

            # Save as Excel
            output_file = os.path.join(OUTPUT_DIR, f"{model_type}_{win_name}.xlsx")
            df_result.to_excel(output_file, index=False)
            print(f"  [{model_type}] Saved: {output_file}  ({len(df_result)} genes)")

            # Collect the top gene info of each window x type
            top_gene = df_result.iloc[0]['gene']
            max_imp = df_result.iloc[0]['mean_PDP']
            summary_records.append((win_name, model_type, top_gene, max_imp))

    # 11. Build the summary table: top genes and IMP per time window x reactor type
    print(f"\n{'='*60}")
    print("Generating summary table of top genes...")
    print(f"{'='*60}")

    df_summary = pd.DataFrame(summary_records, columns=[
        'Time_Window', 'Reactor_Type', 'Top_Gene', 'Max_PDP'
    ])
    summary_output = os.path.join(OUTPUT_DIR, "summary_top_genes.xlsx")
    
    # Save with openpyxl and format the Max_PDP column in scientific notation
    with pd.ExcelWriter(summary_output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, index=False, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']
        # Format the Max_PDP column (column 4, i.e. column D) in scientific notation
        for row in range(2, len(df_summary) + 2):  # start from row 2 (skip the header)
            cell = worksheet.cell(row=row, column=4)
            cell.number_format = '0.00E+00'
    
    print(f"Summary saved: {summary_output}")
    print(df_summary.to_string(index=False))

    print(f"\n{'='*60}")
    print("All done! Files saved to:", OUTPUT_DIR)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
