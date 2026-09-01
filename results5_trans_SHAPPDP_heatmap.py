"""
迁移学习 Gene Model SHAP + PDP 敏感性分析热力图可视化
=========================================================
将 results/Shapely for trans/ 和 results/PDP for trans/ 下各时间步的结果汇总为热力图，
展示 21 个 n gene 的 SHAP/PDP 值随迁移学习过程的演变趋势。
- 3 行 2 列子图：左列 SHAP / 右列 PDP (ana_ano / ox) + 第3行余弦相似度曲线（★ 2block: ANA+ANO合并）
- 横坐标：WQ 实际时间（从 wq.xlsx 获取）
- 纵坐标：KEGG KO 基因名（从 gene_ref/Gene_info.xlsx 获取）
- 颜色：SHAP 用蓝色系（Blues），PDP 用橙色系（Oranges）
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
SHAP_BASE_DIR = os.path.join(ROOT_DIR, "results", "Shapely for trans")
PDP_BASE_DIR = os.path.join(ROOT_DIR, "results", "PDP for trans")
WQ_RAW_PATH = os.path.join(ROOT_DIR, "data_WQ", "wq.xlsx")
GENE_INFO_PATH = os.path.join(ROOT_DIR, "gene_ref", "Gene_info.xlsx")

# Step → WQ 行号的偏移量：dataset window_size = 40,
# 模型保存用 i+1 (= window_end), 所以 step_N 对应 wq.iloc[N + 39]
WINDOW_SIZE = 40

# ============================================================
# 归一化开关：True = 每个时刻列归一化, False = 使用原始值
# ============================================================
USE_NORMALIZE = False

# 输出文件名根据 USE_NORMALIZE 自动区分
_OUTPUT_SUFFIX = "_normalized" if USE_NORMALIZE else "_raw"
OUTPUT_PATH = os.path.join(ROOT_DIR, "results", f"shap_pdp_trans_heatmap{_OUTPUT_SUFFIX}.png")


def load_wq_time_series(wq_path):
    """从 wq.xlsx 读取 time 列，返回 pandas Series"""
    print(f"Loading WQ time data from: {wq_path}")
    df = pd.read_excel(wq_path)
    if 'time' in df.columns:
        time_col = df['time']
    else:
        # 兜底：取第一列
        time_col = df.iloc[:, 0]
    print(f"  Loaded {len(time_col)} time records")
    return time_col


def load_gene_name_mapping(gene_info_path):
    """
    从 Gene_info.xlsx 读取基因编号→名字的映射关系。
    返回 dict: {KO编号: 显示名称}
    """
    print(f"Loading gene info from: {gene_info_path}")
    df = pd.read_excel(gene_info_path)
    
    # Gene_info.xlsx 列: KO, L1, L2, Pathway, Description
    # 使用 Description 列作为显示名称，如果没有则使用 KO 编号
    gene_id_to_name = {}
    for _, row in df.iterrows():
        ko_id = row['KO']
        # Description 格式如: "K00844  HK; hexokinase [EC:2.7.1.1]"
        # 提取简短名称（取第一个分号前的部分或直接用 Description）
        desc = str(row.get('Description', ''))
        if desc and desc.strip():
            # 尝试提取有意义的名字部分
            # 通常格式: "K0xxxx  Name; description [EC:x.x.x.x]"
            name_part = desc.split(';')[0] if ';' in desc else desc
            # 去掉开头的 KO 编号重复
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
    扫描结果目录，返回 [(step_num, dir_path)] 按编号升序排列
    文件夹名格式: step_XXXXX
    prefix: 用于区分 SHAP 或 PDP 文件
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
            # 确认文件夹存在且含对应 CSV
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
    从第一个 SHAP 结果中提取固定的基因顺序（KO 编号列表），
    用于后续所有时间步的矩阵对齐。
    """
    csv_path = os.path.join(first_shap_dir, "SHAP_feature_importance_ana_ano.csv")  # ★ 2block
    df = pd.read_csv(csv_path)
    gene_list = df['gene'].tolist()
    print(f"  Fixed gene order ({len(gene_list)} genes): {gene_list}")
    return gene_list


def load_and_build_matrix(result_dirs, model_type, fixed_genes, value_column, prefix="SHAP"):
    """
    构建指定反应器类型的 [n_genes × n_steps] 矩阵。

    Args:
        result_dirs: discover_result_dirs() 返回的 [(step_num, path)] 列表
        model_type: 'ana' | 'ano' | 'ox'
        fixed_genes: 固定的基因名列表（顺序一致）
        value_column: CSV 中要读取的值列名 (mean_abs_SHAP 或 mean_PDP)
        prefix: "SHAP" 或 "PDP"

    Returns:
        np.ndarray: shape (n_genes, n_steps)，以及对应的 step_nums 列表
    """
    n_genes = len(fixed_genes)
    n_steps = len(result_dirs)
    matrix = np.zeros((n_genes, n_steps))
    step_nums = []

    # 构建 gene → row_index 的映射
    gene_to_idx = {g: i for i, g in enumerate(fixed_genes)}

    for col_idx, (step_num, dir_path) in enumerate(result_dirs):
        csv_name = f"{prefix}_feature_importance_{model_type}.csv"
        csv_path = os.path.join(dir_path, csv_name)

        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, filling with zeros")
            continue

        df = pd.read_csv(csv_path)
        # 用 gene 名映射到固定行索引（CSV 内部顺序可能不同）
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
    对每个时刻（每列）的值进行 min-max 归一化到 [0, 1]。
    归一化按列独立进行，使每个时刻内部各基因的相对重要性可比。

    Args:
        matrices_dict: {'ana': ndarray, 'ano': ndarray, 'ox': ndarray}
                        每个 shape=(n_genes, n_steps)

    Returns:
        dict: 归一化后的矩阵，结构同输入
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
                norm_mat[:, col] = 0.0  # 全列值相同则置0
        normalized[mtype] = norm_mat
        print(f"  Normalized per-step for '{mtype}'")
    return normalized


def compute_cosine_similarity(vec_a, vec_b):
    """
    计算两个向量的余弦相似度。

    Args:
        vec_a: np.ndarray, 向量A
        vec_b: np.ndarray, 向量B（与vec_a等长）

    Returns:
        float: 余弦相似度，范围 [-1, 1]
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
    绘制 4×2 热力图（左列 SHAP Blues / 右列 PDP Oranges）+ 第4行余弦相似度曲线并保存。

    Args:
        shap_matrices: {'ana': ndarray, 'ano': ndarray, 'ox': ndarray} SHAP数据
        pdp_matrices: {'ana': ndarray, 'ano': ndarray, 'ox': ndarray} PDP数据
        time_labels: list of datetime/str, 每个步骤对应的时间标签
        gene_display_names: list of str, 基因显示名称（从 Gene_info 获取）
        output_path: 输出图片路径
        use_normalize: bool, 是否使用了列归一化（影响色阶和标签）
    """
    # ★ 2block: ANA+ANO合并
    types = ['ana_ano', 'ox']
    type_labels = {
        'ana_ano': 'Anaerobic & Anoxic zone gene',
        'ox': 'Oxic zone gene'
    }
    n_types = len(types)
    n_genes = len(gene_display_names)
    n_steps = len(time_labels)

    # ---- 图幅设置 ----
    width_per_step = 0.06 if n_steps > 150 else (0.08 if n_steps > 80 else 0.10)
    max_fig_width = 32  # 最大宽度限制（两列）
    fig_width = min(max(16, n_steps * width_per_step * 2), max_fig_width)
    
    # ---- 创建 Figure 和 GridSpec ----
    fig = plt.figure(
        figsize=(fig_width, n_genes * 0.55 + 10),  # 放大整体图幅
        constrained_layout=False
    )
    
    # 定义 GridSpec: 3行(2行热力图+1行曲线) × 2列（★ 2block: 从4行改为3行）
    gs = fig.add_gridspec(
        n_types + 1, 2,
        height_ratios=[1, 1, 1.0],  # 第3行高度增加（余弦相似度）
        hspace=0.45,
        wspace=0.08
    )
    
    # 存储所有 axes 用于后续统一设置 xtick
    all_axes = []

    # ---- 色阶设置 ----
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
        # ===== 左边: SHAP 热力图 =====
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

        # Y轴：基因显示名称（只在左列显示）
        ax_shap.set_yticks(range(n_genes))
        ax_shap.set_yticklabels(gene_display_names, fontsize=13)

        # 标题
        ax_shap.set_title(f"{type_labels[t]} - SHAP values",
                         fontsize=22, fontweight='bold', pad=10)

        # Colorbar（★ 紧贴热力图）
        cbar_shap = plt.colorbar(im_shap, ax=ax_shap, fraction=0.03, pad=0.00)
        cbar_shap.ax.tick_params(labelsize=18)

        ax_shap.grid(False)
        
        # 在第一行左上角外侧添加标签 (a)
        if idx == 0:
            pos = ax_shap.get_position()
            fig.text(pos.x0 + 0.02, pos.y1 + 0.06, '(a)',  # ★ 右移贴近图
                    transform=fig.transFigure,
                    fontsize=22, fontweight='bold', va='bottom', ha='left')

        # ===== 右边: PDP 热力图 =====
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

        # Y轴：与左边相同的基因顺序（不重复显示标签以节省空间）
        ax_pdp.set_yticks(range(n_genes))
        ax_pdp.set_yticklabels([''] * n_genes)  # 隐藏右侧 y 轴标签

        # 标题
        ax_pdp.set_title(f"{type_labels[t]} - PDP values",
                         fontsize=22, fontweight='bold', pad=10)

        # Colorbar（★ 紧贴热力图）
        cbar_pdp = plt.colorbar(im_pdp, ax=ax_pdp, fraction=0.03, pad=0.00)
        cbar_pdp.ax.tick_params(labelsize=18)

        ax_pdp.grid(False)
        
        # 在第一行右列左上角外侧添加标签 (b)
        if idx == 0:
            pos = ax_pdp.get_position()
            fig.text(pos.x0 + 0.02, pos.y1 + 0.06, '(b)',  # ★ 右移贴近图
                    transform=fig.transFigure,
                    fontsize=22, fontweight='bold', va='bottom', ha='left')

    # ===== 第4行：SHAP vs PDP 余弦相似度曲线（跨列合并） =====
    # 曲线样式配置：颜色 + 线型 + 标记（★ 2block: 从3条改为2条）
    line_styles = {
        'ana_ano': {'color': '#1f77b4', 'linestyle': '-',  'marker': 'o',  'markersize': 6, 'linewidth': 2.8, 'label': 'Anaerobic & Anoxic', 'markevery': max(1, n_steps // 15)},
        'ox':       {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^',  'markersize': 8, 'linewidth': 3.0, 'label': 'Oxic',             'markevery': max(1, n_steps // 10)}
    }
    
    x_positions = range(n_steps)
    
    # ---- 使用 GridSpec 创建跨列子图 ----
    ax_cosine = fig.add_subplot(gs[n_types, :])
    all_axes.append(ax_cosine)
    
    for t in types:
        shap_mat = shap_matrices[t]  # shape: (n_genes, n_steps)
        pdp_mat = pdp_matrices[t]   # shape: (n_genes, n_steps)
        
        # 计算每个时间步的余弦相似度
        cosine_sims = np.zeros(n_steps)
        for step in range(n_steps):
            shap_vec = shap_mat[:, step]
            pdp_vec = pdp_mat[:, step]
            cosine_sims[step] = compute_cosine_similarity(shap_vec, pdp_vec)
        
        # 使用不同画图方式绘制
        style = line_styles[t]
        ax_cosine.plot(x_positions, cosine_sims,
                       color=style['color'],
                       linestyle=style['linestyle'],
                       linewidth=style['linewidth'],
                       marker=style['marker'],
                       markersize=style['markersize'],
                       label=style['label'],
                       markevery=style['markevery'])
    
    ax_cosine.set_ylabel('Cosine Similarity', fontsize=20)  # ★ 放大 ylabel
    ax_cosine.tick_params(axis='y', labelsize=16)  # ★ 放大 ytick
    ax_cosine.set_title('SHAP vs PDP Cosine Similarity (All Reactor Types)',
                        fontsize=22, fontweight='bold', pad=10)
    ax_cosine.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_cosine.axhline(y=1, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_cosine.axhline(y=-1, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_cosine.set_ylim(0.5, 1.5)
    ax_cosine.legend(loc='upper right', fontsize=13, framealpha=0.9)
    ax_cosine.grid(True, alpha=0.3)
    
    # 在最后一张子图左上角外侧添加标签 (c)
    pos = ax_cosine.get_position()
    fig.text(pos.x0 + 0.02, pos.y1 + 0.005, '(c)',  # ★ 右移贴近图
             transform=fig.transFigure,
             fontsize=22, fontweight='bold', va='bottom', ha='right')

    # X轴：时间标签——固定显示5个关键位置
    tick_indices = sorted(set([0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]))

    x_tick_labels = []
    for ti in tick_indices:
        tl = time_labels[ti]
        if isinstance(tl, (datetime, pd.Timestamp)):
            x_tick_labels.append(tl.strftime('%Y-%m-%d'))
        else:
            x_tick_labels.append(str(tl)[:10])

    # X轴：时间标签——只有第2、3行显示xtick（字体缩小），其他子图去掉xtick（★ 2block: 从4行改为3行）
    n_total_axes = len(all_axes)
    # 第2行是最后第3和第2个axes，第3行是最后1个axes
    show_xtick_indices = {n_total_axes - 3, n_total_axes - 2, n_total_axes - 1}  # Row2左, Row2右, Row3
    
    for ax_idx, ax in enumerate(all_axes):
        if ax_idx in show_xtick_indices:
            # 第3、4行显示xtick，字体缩小
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(x_tick_labels, rotation=30, ha='right', fontsize=12)
        else:
            # 其他子图去掉xtick
            ax.set_xticks([])
            ax.set_xticklabels([])

    fig.subplots_adjust(left=0.15, right=0.92, top=0.94, bottom=0.10, hspace=0.35, wspace=0.15)

    # ★ 手动缩窄前两行热力图（Row 0 和 Row 1）
    heatmap_axes = all_axes[:4]  # 前4个 axes: Row0(SHAP,PDP), Row1(SHAP,PDP)
    shrink_factor = 0.85  # 缩窄到原来的85%
    for ax in heatmap_axes:
        pos = ax.get_position()
        new_width = pos.width * shrink_factor
        # 居中缩窄：左右各缩进一半
        new_x0 = pos.x0 + (pos.width - new_width) / 2
        ax.set_position([new_x0, pos.y0, new_width, pos.height])

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nHeatmap saved to: {output_path}")


def main():
    """主流程"""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print("=" * 60)
    print("Trans-Learning Gene SHAP + PDP Heatmap Visualization")
    print("=" * 60)

    # 1. 加载 WQ 时间序列
    wq_time = load_wq_time_series(WQ_RAW_PATH)

    # 2. 加载基因名称映射
    gene_name_map = load_gene_name_mapping(GENE_INFO_PATH)

    # 3. 发现所有 SHAP 和 PDP 结果目录
    shap_dirs = discover_result_dirs(SHAP_BASE_DIR, prefix="SHAP")
    pdp_dirs = discover_result_dirs(PDP_BASE_DIR, prefix="PDP")
    
    if not shap_dirs:
        print("ERROR: No SHAP result directories found!")
        return
    
    if not pdp_dirs:
        print("WARNING: No PDP result directories found! Only SHAP will be plotted.")
        pdp_dirs = shap_dirs  # 兜底：使用 SHAP 的目录结构

    # 4. 获取共同的 step 列表（以 SHAP 为基准，取交集）
    shap_step_nums = set(s[0] for s in shap_dirs)
    pdp_step_nums = set(p[0] for p in pdp_dirs)
    common_step_nums = sorted(shap_step_nums & pdp_step_nums)
    
    if not common_step_nums:
        print("WARNING: No common steps between SHAP and PDP! Using all SHAP steps.")
        common_step_nums = [s[0] for s in shap_dirs]
    
    # 过滤出共同 step 的目录
    shap_dirs_common = [(s, p) for s, p in shap_dirs if s in common_step_nums]
    pdp_dirs_common = [(s, p) for s, p in pdp_dirs if s in common_step_nums]
    
    print(f"\nCommon steps: {len(common_step_nums)} (step_{common_step_nums[0]} ~ step_{common_step_nums[-1]})")

    # 5. 获取固定的基因顺序（从第一个 SHAP 结果）
    fixed_genes = get_fixed_gene_order(shap_dirs_common[0][1])
    
    # 将基因编号转换为显示名称
    gene_display_names = [gene_name_map.get(g, g) for g in fixed_genes]
    print(f"\nGene display names (first 5): {gene_display_names[:5]}")

    # 6. 为每个 step 提取对应的时间标签
    time_labels = []
    for step_num, _ in shap_dirs_common:
        row_idx = step_num + WINDOW_SIZE - 1  # step_N → wq.iloc[N+39]
        if row_idx < len(wq_time):
            time_labels.append(wq_time.iloc[row_idx])
        else:
            time_labels.append(f"N/A (idx={row_idx})")
    print(f"\nTime range: {time_labels[0]} ~ {time_labels[-1]}")

    # 7. 构建 SHAP 矩阵（★ 2block: ana_ano/ox）
    shap_matrices = {}
    for mtype in ['ana_ano', 'ox']:
        print(f"\nBuilding SHAP matrix for '{mtype}'...")
        mat, _ = load_and_build_matrix(shap_dirs_common, mtype, fixed_genes,
                                       value_column='mean_abs_SHAP', prefix="SHAP")
        shap_matrices[mtype] = mat

    # 8. 构建 PDP 矩阵（★ 2block: ana_ano/ox）
    pdp_matrices = {}
    for mtype in ['ana_ano', 'ox']:
        print(f"\nBuilding PDP matrix for '{mtype}'...")
        mat, _ = load_and_build_matrix(pdp_dirs_common, mtype, fixed_genes,
                                       value_column='mean_PDP', prefix="PDP")
        pdp_matrices[mtype] = mat

    # 9. 根据 USE_NORMALIZE 开关决定是否列归一化
    if USE_NORMALIZE:
        print("\nNormalizing values per time step...")
        shap_matrices = normalize_per_step(shap_matrices)
        pdp_matrices = normalize_per_step(pdp_matrices)
    else:
        print("\nSkipping per-step normalization (USE_NORMALIZE = False)")

    # 10. 绘制热力图
    plot_shap_pdp_heatmap(shap_matrices, pdp_matrices, time_labels, gene_display_names,
                          OUTPUT_PATH, use_normalize=USE_NORMALIZE)
    print("\nDone!")


if __name__ == "__main__":
    main()
