"""
统计各时间窗口内每个 gene 的 PDP 平均值，导出为 Excel。

时间窗口:
  - Window 1: 7月13日 ~ 8月13日
  - Window 2: 8月13日 ~ 9月13日
  - Window 3: 9月13日 ~ 10月13日

输出: results/PDP/ 目录下，每个反应器类型 × 时间窗口 一个 Excel 文件。
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================
# 路径配置（与 results5 保持一致，仅替换数据源和输出目录）
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WQ_PATH = os.path.join(BASE_DIR, "data_WQ", "wq.xlsx")
PDP_BASE_DIR = os.path.join(BASE_DIR, "results", "PDP for trans")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "PDP")

# 时间窗口定义 (左闭右开)
TIME_WINDOWS = [
    ("2025-07-13", "2025-08-13", "Window1_0713-0813"),
    ("2025-08-13", "2025-09-13", "Window2_0813-0913"),
    ("2025-09-13", "2025-10-13", "Window3_0913-1013"),
]

# Step → WQ 行号的偏移量：dataset window_size = 40,
# 模型保存用 i+1 (= window_end), 所以 step_N 对应 wq.iloc[N+39)
WINDOW_SIZE = 40

MODEL_TYPES = ["ana_ano", "ox"]  # ★ 2block: ANA+ANO合并


# ============================================================
# 数据加载函数（复用 results5 的逻辑，替换为 PDP）
# ============================================================

def load_wq_time_series(wq_path):
    """从 wq.xlsx 读取 time 列"""
    print(f"Loading WQ time data from: {wq_path}")
    df = pd.read_excel(wq_path)
    if 'time' in df.columns:
        time_col = df['time']
    else:
        time_col = df.iloc[:, 0]
    print(f"  Loaded {len(time_col)} time records")
    return time_col


def discover_pdp_dirs(base_dir):
    """扫描 PDP for trans 目录，返回 [(step_num, dir_path)] 按编号升序排列"""
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
    """从第一个 PDP 结果中提取固定的基因顺序"""
    csv_path = os.path.join(first_pdp_dir, "PDP_feature_importance_ana_ano.csv")  # ★ 2block
    df = pd.read_csv(csv_path)
    gene_list = df['gene'].tolist()
    print(f"  Fixed gene order ({len(gene_list)} genes)")
    return gene_list


def load_and_build_matrix(pdp_dirs, model_type, fixed_genes):
    """构建 [n_genes x n_steps] PDP 矩阵"""
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
# 主逻辑
# ============================================================

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # 1. 加载时间序列
    dates = load_wq_time_series(WQ_PATH)
    # 转换为 datetime
    date_list = []
    for d in dates:
        if isinstance(d, (datetime, pd.Timestamp)):
            date_list.append(pd.to_datetime(d))
        else:
            date_list.append(pd.to_datetime(str(d)))

    # 2. 发现所有 PDP 目录
    pdp_dirs = discover_pdp_dirs(PDP_BASE_DIR)
    if not pdp_dirs:
        print("ERROR: No PDP directories found!")
        return

    # 3. 获取固定基因顺序
    first_dir = pdp_dirs[0][1]
    fixed_genes = get_fixed_gene_order(first_dir)

    # 4. 为每种反应器类型构建矩阵
    matrices = {}
    step_nums_all = {}
    for model_type in MODEL_TYPES:
        mat, steps = load_and_build_matrix(pdp_dirs, model_type, fixed_genes)
        matrices[model_type] = mat
        step_nums_all[model_type] = steps

    n_steps_total = len(pdp_dirs)

    # 4.5 为每个 step 计算对应的实际时间（与热力图脚本一致：step_N → wq.iloc[N+39]）
    step_times = []
    for step_num, _ in pdp_dirs:
        row_idx = step_num + WINDOW_SIZE - 1
        if row_idx < len(date_list):
            step_times.append(date_list[row_idx])
        else:
            step_times.append(None)
    print(f"\nStep time range: {step_times[0]} ~ {step_times[-1]}")

    # 5. 对每个时间窗口进行统计
    # 同时收集每个窗口×类型的最大gene信息，用于最终汇总
    summary_records = []  # [(time_window, model_type, top_gene, max_IMP)]

    for win_start_str, win_end_str, win_name in TIME_WINDOWS:
        win_start = pd.to_datetime(win_start_str)
        win_end = pd.to_datetime(win_end_str)

        print(f"\n{'='*60}")
        print(f"Time Window: {win_name} ({win_start_str} ~ {win_end_str})")
        print(f"{'='*60}")

        # 找出属于该时间窗口的 step 列索引（使用 step 对应的实际时间）
        window_indices = []
        for i, t in enumerate(step_times):
            if t is not None and win_start <= t < win_end:
                window_indices.append(i)

        print(f"  Found {len(window_indices)} time points in this window")

        if not window_indices:
            print("  WARNING: No data points in this window, skipping.")
            continue

        # 6. 对每种反应器类型统计平均值
        for model_type in MODEL_TYPES:
            mat = matrices[model_type]

            # 提取该时间窗口的列，计算每行(gene)的平均值
            window_data = mat[:, window_indices]  # shape: (n_genes, n_window_steps)
            gene_means = np.mean(window_data, axis=1)  # shape: (n_genes,)

            # 构建 DataFrame
            df_result = pd.DataFrame({
                'gene': fixed_genes,
                'mean_PDP': gene_means
            })

            # 按 mean_PDP 降序排列
            df_result = df_result.sort_values('mean_PDP', ascending=False).reset_index(drop=True)

            # 保存为 Excel
            output_file = os.path.join(OUTPUT_DIR, f"{model_type}_{win_name}.xlsx")
            df_result.to_excel(output_file, index=False)
            print(f"  [{model_type}] Saved: {output_file}  ({len(df_result)} genes)")

            # 收集该窗口×类型的最大gene信息
            top_gene = df_result.iloc[0]['gene']
            max_imp = df_result.iloc[0]['mean_PDP']
            summary_records.append((win_name, model_type, top_gene, max_imp))

    # 7. 生成汇总表：每个时间窗口×反应器类型 的 top gene 和 IMP
    print(f"\n{'='*60}")
    print("Generating summary table of top genes...")
    print(f"{'='*60}")

    df_summary = pd.DataFrame(summary_records, columns=[
        'Time_Window', 'Reactor_Type', 'Top_Gene', 'Max_PDP'
    ])
    summary_output = os.path.join(OUTPUT_DIR, "summary_top_genes.xlsx")
    
    # 使用 openpyxl 保存，设置 Max_PDP 列为科学计数法格式
    with pd.ExcelWriter(summary_output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, index=False, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']
        # 设置 Max_PDP 列（第4列，即D列）为科学计数法格式
        for row in range(2, len(df_summary) + 2):  # 从第2行开始（跳过表头）
            cell = worksheet.cell(row=row, column=4)
            cell.number_format = '0.00E+00'
    
    print(f"Summary saved: {summary_output}")
    print(df_summary.to_string(index=False))

    print(f"\n{'='*60}")
    print("All done! Files saved to:", OUTPUT_DIR)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
