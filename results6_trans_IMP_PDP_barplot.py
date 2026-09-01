"""
迁移学习 Gene Model IMP + PDP 柱状图可视化
=========================================================
将 results/IMP/ 和 results/PDP/ 下两个时间窗口的结果绘制为横向柱状图，
展示 21 个 n gene 的 IMP/PDP 值对比。
- 3 行 4 列子图：
  - 第1列: 0813-0913 IMP值 (ana_ano/ox)
  - 第2列: 0913-1013 IMP值 (ana_ano/ox)
  - 第3列: 0813-0913 PDP值 (ana_ano/ox)
  - 第4列: 0913-1013 PDP值 (ana_ano/ox)
  - 第3行: n_gene_dynamic 数据 (9月/10月)
- 横向柱状图，颜色: IMP用蓝色系，PDP用橙色系，dynamic用绿色系
- Y轴: 基因显示名称（从 Gene_info.xlsx 获取），顺序一致
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
IMP_BASE_DIR = os.path.join(ROOT_DIR, "results", "IMP")
PDP_BASE_DIR = os.path.join(ROOT_DIR, "results", "PDP")
GENE_INFO_PATH = os.path.join(ROOT_DIR, "gene_ref", "Gene_info.xlsx")
DYNAMIC_DATA_PATH = os.path.join(ROOT_DIR, "data_GeneM", "n_gene_dynamic.xlsx")

# 时间窗口配置
WINDOWS = {
    '0813-0913': 'Window2',
    '0913-1013': 'Window3'
}

OUTPUT_PATH = os.path.join(ROOT_DIR, "results", "imp_pdp_barplot_comparison.png")


def load_gene_name_mapping(gene_info_path):
    """
    从 Gene_info.xlsx 读取基因编号→名字的映射关系。
    返回 dict: {KO编号: 显示名称}
    """
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


def load_dynamic_data(dynamic_path, sheet_name):
    """
    加载 n_gene_dynamic 数据（指定sheet）。
    
    Args:
        dynamic_path: n_gene_dynamic.xlsx 文件路径
        sheet_name: sheet名称 ('ana_ano' 或 'ox')
    
    Returns:
        DataFrame: 包含 gene, Sep, Oct 列
    """
    print(f"Loading dynamic data from: {dynamic_path} (sheet: {sheet_name})")
    df = pd.read_excel(dynamic_path, sheet_name=sheet_name)
    
    # 处理列名：第3列是基因ID
    df.columns = ['Sep', 'Oct', 'gene']
    
    # 转换为数值类型
    df['Sep'] = pd.to_numeric(df['Sep'], errors='coerce')
    df['Oct'] = pd.to_numeric(df['Oct'], errors='coerce')
    
    print(f"  Loaded {len(df)} genes with Sep/Oct values")
    return df


def load_data(base_dir, model_type, window_key, data_type):
    """
    加载指定类型的IMP或PDP数据。
    
    Args:
        base_dir: IMP或PDP的基目录
        model_type: 'ana' | 'ano' | 'ox'
        window_key: '0813-0913' | '0913-1013'
        data_type: 'IMP' | 'PDP'
    
    Returns:
        DataFrame: 包含 gene 和 value 列
    """
    window_name = WINDOWS[window_key]
    filename = f"{model_type}_{window_name}_{window_key}.xlsx"
    filepath = os.path.join(base_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found!")
        return None
    
    df = pd.read_excel(filepath)
    
    # 统一列名
    value_col = 'mean_SHAP' if data_type == 'IMP' else 'mean_PDP'
    if value_col not in df.columns:
        value_col = df.columns[1]  # 兜底取第二列
    
    return df[['gene', value_col]].rename(columns={value_col: 'value'})


def plot_imp_pdp_barchart(output_path):
    """
    绘制 4×4 横向柱状图并保存（含 n_gene_dynamic 数据行）。
    """
    # 加载基因名称映射
    gene_name_map = load_gene_name_mapping(GENE_INFO_PATH)
    
    # 反应器类型和标签（★ 2block: ANA+ANO合并）
    types = ['ana_ano', 'ox']
    type_labels = {
        'ana_ano': 'Anaerobic & Anoxic zone',
        'ox': 'Oxic zone'
    }
    
    # 数据类型配置
    data_configs = [
        ('IMP', '0813-0913', IMP_BASE_DIR, 'IMP'),
        ('IMP', '0913-1013', IMP_BASE_DIR, 'IMP'),
        ('LEF', '0813-0913', PDP_BASE_DIR, 'LEF'),
        ('LEF', '0913-1013', PDP_BASE_DIR, 'LEF'),
    ]
    
    # 颜色配置（参考results5）
    imp_color = '#2166ac'  # 蓝色系（类似Blues）
    pdp_color = '#d6604d'  # 橙红色系（类似Oranges）
    dynamic_color = '#1b7837'  # 绿色系（用于dynamic数据）
    
    # 列标题
    col_titles = [
        'IMP (August 13-September 13)',
        'IMP (September 13-October 13)',
        'LEF (August 13-September 13)',
        'LEF (September 13-October 13)'
    ]
    
    # ---- 首先加载数据以获取基因顺序 ----
    # 使用第一个数据文件确定基因顺序（★ 2block: ana_ano）
    first_df = load_data(IMP_BASE_DIR, 'ana_ano', '0813-0913', 'IMP')
    if first_df is None:
        print("ERROR: Cannot load reference data!")
        return
    
    fixed_genes = first_df['gene'].tolist()
    gene_display_names = [gene_name_map.get(g, g) for g in fixed_genes]
    n_genes = len(fixed_genes)
    
    print(f"\nFixed gene order ({n_genes} genes)")
    print(f"Gene display names: {gene_display_names}")
    
    # ---- 加载 dynamic 数据 (两个sheet) ----
    dynamic_ana_ano = load_dynamic_data(DYNAMIC_DATA_PATH, 'ana_ano')
    dynamic_ox = load_dynamic_data(DYNAMIC_DATA_PATH, 'ox')
    
    # ---- 创建图形 (3行4列，★ 2block: ANA+ANO合并后减少一行) ----
    fig, axes = plt.subplots(3, 4, figsize=(16, 14))
    
    for row_idx, mtype in enumerate(types):
        for col_idx, (data_type, window_key, base_dir, _) in enumerate(data_configs):
            ax = axes[row_idx, col_idx]
            
            # 加载数据
            df = load_data(base_dir, mtype, window_key, data_type)
            
            if df is not None:
                # 按固定基因顺序对齐数据
                df_sorted = df.set_index('gene').loc[fixed_genes].reset_index()
                values = df_sorted['value'].values
                
                # 选择颜色
                color = imp_color if data_type == 'IMP' else pdp_color
                
                # 绘制横向柱状图
                y_pos = range(n_genes)
                ax.barh(y_pos, values, color=color, edgecolor='white', linewidth=0.5, height=0.7, alpha=0.75)
                
                # 设置Y轴
                ax.set_yticks(y_pos)
                if col_idx == 0:
                    ax.set_yticklabels(gene_display_names, fontsize=10)
                else:
                    ax.set_yticklabels([''] * n_genes)
            
            # 设置标题：每行都显示，格式为 "Anaerobic zone IMP (August 13-September 13)"
            date_labels = {
                '0813-0913': 'August 13-September 13',
                '0913-1013': 'September 13-October 13'
            }
            data_type_label = data_type  # 'IMP' 或 'LEF'
            date_label = date_labels.get(window_key, window_key)
            ax.set_title(f'{type_labels[mtype]} {data_type_label}\n({date_label})', fontsize=12, fontweight='bold', pad=8)
            
            # 网格线
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
    
    # ---- 第3行: n_gene_dynamic 数据 (两个sheet，★ 2block: 从第4行改为第3行) ----
    y_pos = range(n_genes)
    
    # Sheet 1: ana_ano (第1、2列)
    if dynamic_ana_ano is not None:
        dynamic_aa_sorted = dynamic_ana_ano.set_index('gene').loc[fixed_genes].reset_index()
        
        # 第1列: ana_ano Sep
        ax_aa_sep = axes[2, 0]
        ax_aa_sep.barh(y_pos, dynamic_aa_sorted['Sep'].values, color=dynamic_color, 
                       edgecolor='white', linewidth=0.5, height=0.7, alpha=0.75)
        ax_aa_sep.set_yticks(y_pos)
        ax_aa_sep.set_yticklabels(gene_display_names, fontsize=10)
        ax_aa_sep.set_title('MT: Ana&ano Sep', fontsize=12, fontweight='bold', pad=8)
        ax_aa_sep.grid(axis='x', alpha=0.3, linestyle='--')
        ax_aa_sep.set_axisbelow(True)
        
        # 第2列: ana_ano Oct
        ax_aa_oct = axes[2, 1]
        ax_aa_oct.barh(y_pos, dynamic_aa_sorted['Oct'].values, color=dynamic_color,
                       edgecolor='white', linewidth=0.5, height=0.7, alpha=0.75)
        ax_aa_oct.set_yticks(y_pos)
        ax_aa_oct.set_yticklabels([''] * n_genes)
        ax_aa_oct.set_title('MT: Ana&ano Oct', fontsize=12, fontweight='bold', pad=8)
        ax_aa_oct.grid(axis='x', alpha=0.3, linestyle='--')
        ax_aa_oct.set_axisbelow(True)
    
    # Sheet 2: ox (第3、4列)
    if dynamic_ox is not None:
        dynamic_ox_sorted = dynamic_ox.set_index('gene').loc[fixed_genes].reset_index()
        
        # 第3列: ox Sep
        ax_ox_sep = axes[2, 2]
        ax_ox_sep.barh(y_pos, dynamic_ox_sorted['Sep'].values, color=dynamic_color,
                       edgecolor='white', linewidth=0.5, height=0.7, alpha=0.75)
        ax_ox_sep.set_yticks(y_pos)
        ax_ox_sep.set_yticklabels([''] * n_genes)
        ax_ox_sep.set_title('MT: Ox Sep', fontsize=12, fontweight='bold', pad=8)
        ax_ox_sep.grid(axis='x', alpha=0.3, linestyle='--')
        ax_ox_sep.set_axisbelow(True)
        
        # 第4列: ox Oct
        ax_ox_oct = axes[2, 3]
        ax_ox_oct.barh(y_pos, dynamic_ox_sorted['Oct'].values, color=dynamic_color,
                       edgecolor='white', linewidth=0.5, height=0.7, alpha=0.75)
        ax_ox_oct.set_yticks(y_pos)
        ax_ox_oct.set_yticklabels([''] * n_genes)
        ax_ox_oct.set_title('MT: Ox Oct', fontsize=12, fontweight='bold', pad=8)
        ax_ox_oct.grid(axis='x', alpha=0.3, linestyle='--')
        ax_ox_oct.set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 添加子图标签 (a) 和 (b) - 在子图外面左上角
    # (a) 在第1行第1列左上角外侧
    pos_a = axes[0, 0].get_position()
    fig.text(pos_a.x0 - 0.01, pos_a.y1 + 0.01, '(a)',
             transform=fig.transFigure, fontsize=14, fontweight='bold', va='bottom', ha='right')
    
    # (b) 在第1行第3列左上角外侧
    pos_b = axes[0, 2].get_position()
    fig.text(pos_b.x0 - 0.01, pos_b.y1 + 0.01, '(b)',
             transform=fig.transFigure, fontsize=14, fontweight='bold', va='bottom', ha='right')
    
    # (c) 在第3行第1列左上角外侧（★ 2block: 从第4行改为第3行）
    pos_c = axes[2, 0].get_position()
    fig.text(pos_c.x0 - 0.01, pos_c.y1 + 0.01, '(c)',
             transform=fig.transFigure, fontsize=14, fontweight='bold', va='bottom', ha='right')
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nBar chart saved to: {output_path}")


def extract_top10_genes(output_path):
    """
    提取IMP/PDP/MT各时间段和zone的前10个重要基因，保存为Excel。
    """
    print("\n" + "=" * 60)
    print("Extracting Top 10 Genes for IMP / PDP / MT")
    print("=" * 60)
    
    # 加载基因名称映射
    gene_name_map = load_gene_name_mapping(GENE_INFO_PATH)
    
    # 配置（★ 2block: ANA+ANO合并）
    types = ['ana_ano', 'ox']
    type_labels = {'ana_ano': 'Anaerobic & Anoxic', 'ox': 'Oxic'}
    windows = ['0813-0913', '0913-1013']
    
    # 存储所有结果
    all_results = []
    
    # ---- 1. 提取 IMP 前10基因 ----
    print("\n--- IMP Top 10 Genes ---")
    for window in windows:
        for mtype in types:
            df = load_data(IMP_BASE_DIR, mtype, window, 'IMP')
            if df is not None:
                df_sorted = df.sort_values('value', ascending=False).head(10)
                for _, row in df_sorted.iterrows():
                    gene_id = row['gene']
                    all_results.append({
                        'Metric': 'IMP',
                        'Time Window': window,
                        'Zone': type_labels[mtype],
                        'Rank': len([r for r in all_results if r['Metric']=='IMP' and r['Time Window']==window and r['Zone']==type_labels[mtype]]) + 1,
                        'Gene ID': gene_id,
                        'Gene Name': gene_name_map.get(gene_id, gene_id),
                        'Value': round(row['value'], 6)
                    })
                print(f"  {window} {mtype}: extracted {len(df_sorted)} genes")
    
    # ---- 2. 提取 PDP (LEF) 前10基因 ----
    print("\n--- PDP (LEF) Top 10 Genes ---")
    for window in windows:
        for mtype in types:
            df = load_data(PDP_BASE_DIR, mtype, window, 'LEF')
            if df is not None:
                df_sorted = df.sort_values('value', ascending=False).head(10)
                for _, row in df_sorted.iterrows():
                    gene_id = row['gene']
                    all_results.append({
                        'Metric': 'PDP',
                        'Time Window': window,
                        'Zone': type_labels[mtype],
                        'Rank': len([r for r in all_results if r['Metric']=='PDP' and r['Time Window']==window and r['Zone']==type_labels[mtype]]) + 1,
                        'Gene ID': gene_id,
                        'Gene Name': gene_name_map.get(gene_id, gene_id),
                        'Value': round(row['value'], 6)
                    })
                print(f"  {window} {mtype}: extracted {len(df_sorted)} genes")
    
    # ---- 3. 提取 MT (Dynamic) 前10基因 ----
    print("\n--- MT (Dynamic) Top 10 Genes ---")
    dynamic_ana_ano = load_dynamic_data(DYNAMIC_DATA_PATH, 'ana_ano')
    dynamic_ox = load_dynamic_data(DYNAMIC_DATA_PATH, 'ox')
    
    # ana_ano zone: Sep 和 Oct
    for month, col in [('Sep', 'Sep'), ('Oct', 'Oct')]:
        if dynamic_ana_ano is not None:
            df_sorted = dynamic_ana_ano.sort_values(col, ascending=False).head(10)
            for _, row in df_sorted.iterrows():
                gene_id = row['gene']
                all_results.append({
                    'Metric': 'MT',
                    'Time Window': f'{month} (Ana&Ano)',
                    'Zone': 'Anaerobic & Anoxic',
                    'Rank': len([r for r in all_results if r['Metric']=='MT' and r['Time Window']==f'{month} (Ana&Ano)']) + 1,
                    'Gene ID': gene_id,
                    'Gene Name': gene_name_map.get(gene_id, gene_id),
                    'Value': round(row[col], 6)
                })
            print(f"  ana_ano {month}: extracted {len(df_sorted)} genes")
    
    # ox zone: Sep 和 Oct
    for month, col in [('Sep', 'Sep'), ('Oct', 'Oct')]:
        if dynamic_ox is not None:
            df_sorted = dynamic_ox.sort_values(col, ascending=False).head(10)
            for _, row in df_sorted.iterrows():
                gene_id = row['gene']
                all_results.append({
                    'Metric': 'MT',
                    'Time Window': f'{month} (Ox)',
                    'Zone': 'Oxic',
                    'Rank': len([r for r in all_results if r['Metric']=='MT' and r['Time Window']==f'{month} (Ox)']) + 1,
                    'Gene ID': gene_id,
                    'Gene Name': gene_name_map.get(gene_id, gene_id),
                    'Value': round(row[col], 6)
                })
            print(f"  ox {month}: extracted {len(df_sorted)} genes")
    
    # ---- 4. 保存为 Excel ----
    result_df = pd.DataFrame(all_results)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_excel(output_path, index=False, sheet_name='Top10Genes')
    print(f"\n[OK] Top 10 genes saved to: {output_path}")
    print(f"  Total records: {len(result_df)}")
    
    return result_df


def main():
    """主流程"""
    import warnings
    warnings.filterwarnings('ignore')
    
    print("=" * 60)
    print("Trans-Learning Gene IMP + PDP Bar Chart Visualization")
    print("=" * 60)
    
    plot_imp_pdp_barchart(OUTPUT_PATH)
    
    # 提取并保存Top10基因
    TOP10_OUTPUT_PATH = os.path.join(ROOT_DIR, "results", "top10gene.xlsx")
    extract_top10_genes(TOP10_OUTPUT_PATH)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
