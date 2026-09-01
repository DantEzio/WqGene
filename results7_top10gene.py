"""
results7_top10gene.py
=====================
功能：整合 9月(Sep)/10月(Oct) 的 geneM(MT)、IMP、LEF(PDP) 数据，
提取每个 Period × DO × Index 组合下的 Top10 基因，
并找出同一 DO 下三个 Index 共同识别的重要基因。

输出格式：
  Period | DO       | Index | Gene name                              | Commonly identified important genes
  -------|----------|-------|----------------------------------------|--------------------------------------
  Sep    | Ana&ano  | MT    | K00366, K00368, ...                    | K00371, K04561, ...
  Sep    | Ana&ano  | IMP   | K04561, K03385, ...                    | K00371, K04561, ...
  Sep    | Ana&ano  | LEF   | K04561, K02586, ...                    | K00371, K04561, ...
  ...

数据源：
  - MT (geneM):     ./data_GeneM/n_gene_dynamic.xlsx  (sheet: ana_ano, ox)
  - IMP:            ./results/IMP/*_Window2_0813-0913.xlsx (Sep)
                             *_Window3_0913-1013.xlsx (Oct)
  - LEF (PDP):      ./results/PDP/*_Window2_0813-0913.xlsx (Sep)
                             *_Window3_0913-1013.xlsx (Oct)
  - Gene 名称映射： ./gene_ref/Gene_info.xlsx
"""

import os
import pandas as pd


# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GENEM_PATH = os.path.join(BASE_DIR, 'data_GeneM', 'n_gene_dynamic.xlsx')
IMP_DIR = os.path.join(BASE_DIR, 'results', 'IMP')
PDP_DIR = os.path.join(BASE_DIR, 'results', 'PDP')  # LEF data
GENE_INFO_PATH = os.path.join(BASE_DIR, 'gene_ref', 'Gene_info.xlsx')

OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'top10gene.xlsx')

TOP_N = 10  # 提取 Top N 基因

# DO 类型映射
DO_TYPES = {
    'ana_ano': 'Ana&ano',
    'ox': 'Ox',
}

# 月份映射（基于文件名中的日期范围）
PERIOD_MAP = {
    'Window2_0813-0913': 'Sep',
    'Window3_0913-1013': 'Oct',
}


# ============================================================
# 1. 加载 Gene 名称映射表 (Kxxxxx → 描述名)
# ============================================================
def load_gene_info():
    """加载 Gene_info，返回 KO → Description 的映射字典"""
    if not os.path.exists(GENE_INFO_PATH):
        print(f"[警告] Gene_info 不存在: {GENE_INFO_PATH}")
        return {}
    
    df = pd.read_excel(GENE_INFO_PATH)
    # 取每個 KO 的第一條描述
    gene_map = dict(zip(df['KO'], df['Description']))
    print(f"[Gene_info] 已加载 {len(gene_map)} 条基因映射")
    return gene_map


def resolve_gene_name(ko_id, gene_map):
    """
    解析基因名称：从 Gene_info Description 中提取基因名（保留原始格式）。
    Description 格式: 
      - "K00366  nirA; ferredoxin-nitrite reductase [EC:1.7.7.1]" → "nirA"
      - "K00371  narH, narY, nxrB; nitrate reductase ..." → "narH, narY, nxrB"
    返回单个 KO 对应的基因名（多个则保留逗号）。
    """
    if not isinstance(ko_id, str):
        ko_id = str(ko_id)
    
    ko_id = ko_id.strip()
    
    if ko_id.upper().startswith('K'):
        desc = gene_map.get(ko_id.upper(), '')
        if desc:
            # 提取 ; 前面的部分（基因名区域）
            if ';' in desc:
                gene_part = desc.split(';')[0].strip()
            else:
                gene_part = desc.strip()
            
            # 去掉可能的 KO ID 前缀 (如 "K15864  nirS" → "nirS")
            parts = gene_part.split()
            if len(parts) > 1 and parts[0].upper().startswith('K'):
                gene_part = ' '.join(parts[1:])
            
            return gene_part  # ★ 保留原始逗号格式: "narH, narY, nxrB"
        return ko_id
    return ko_id


# ============================================================
# 2. 读取 MT (geneM) 数据
# ============================================================
def read_genem_data():
    """
    读取 n_gene_dynamic.xlsx，按 DO 类型提取 Top N 基因。
    返回: {do_type: {period: [gene_list]}}
    """
    result = {}
    
    if not os.path.exists(GENEM_PATH):
        print(f"[错误] geneM 文件不存在: {GENEM_PATH}")
        return result
    
    # 两个 sheet: ana_ano, ox
    for do_type in ['ana_ano', 'ox']:
        try:
            df = pd.read_excel(GENEM_PATH, sheet_name=do_type)
            print(f"\n[geneM] Sheet [{do_type}]: shape={df.shape}, cols={list(df.columns)}")
            
            # 列结构: [Sep_col, Oct_col, gene_name_col]
            cols = list(df.columns)
            gene_col = cols[-1]  # 基因名列通常是最后一列
            
            result[do_type] = {}
            
            # Sep 对应第1列 (index 0), Oct 对应第2列 (index 1)
            period_cols = {
                'Sep': cols[0],
                'Oct': cols[1],
            }
            
            for period, col_name in period_cols.items():
                # 按值降序排列，取 Top N
                df_sorted = df.sort_values(by=col_name, ascending=False)
                top_genes = df_sorted[gene_col].head(TOP_N).tolist()
                
                # 清理基因名称
                top_genes = [str(g).strip() for g in top_genes if pd.notna(g)]
                
                result[do_type][period] = top_genes
                print(f"  {period} Top{TOP_N}: {top_genes[:3]}... (共{len(top_genes)}个)")
                
        except Exception as e:
            print(f"[错误] 读取 geneM sheet [{do_type}] 失败: {e}")
    
    return result


# ============================================================
# 3. 读取 IMP / LEF (PDP) 数据
# ============================================================
def read_index_data(data_dir, index_name):
    """
    读取 IMP 或 PDP 目录下的数据，按 DO 类型和月份分类。
    
    Args:
        data_dir: 数据目录 (IMP_DIR 或 PDP_DIR)
        index_name: 'IMP' 或 'LEF'
    
    Returns:
        {do_type: {period: [gene_list]}}
    """
    result = {}
    
    if not os.path.exists(data_dir):
        print(f"[错误] 目录不存在: {data_dir}")
        return result
    
    for do_type in ['ana_ano', 'ox']:
        result[do_type] = {}
        
        for window_key, period in PERIOD_MAP.items():
            # 构建文件名: {do_type}_{window_key}.xlsx
            filename = f"{do_type}_{window_key}.xlsx"
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"[警告] {index_name} 文件不存在: {filename}")
                result[do_type][period] = []
                continue
            
            try:
                df = pd.read_excel(filepath)
                # 取 Top N 基因 (已按重要性排序)
                top_genes = df['gene'].head(TOP_N).tolist()
                top_genes = [str(g).strip() for g in top_genes if pd.notna(g)]
                
                result[do_type][period] = top_genes
                print(f"[{index_name}] {filename}: Top{TOP_N} = {top_genes[:3]}... (共{len(top_genes)}个)")
                
            except Exception as e:
                print(f"[错误] 读取 {index_name} 文件失败 {filename}: {e}")
                result[do_type][period] = []
    
    return result


# ============================================================
# 4. 构建 Top10Gene 表格
# ============================================================
def build_top10gene_table(genem_data, imp_data, lef_data, gene_map):
    """
    整合所有数据源，构建最终表格。
    
    表格结构:
      Period | DO | Index | Gene name | Common genes
    """
    rows = []
    
    # 遍历所有 Period × DO × Index 组合
    for period in ['Sep', 'Oct']:
        for do_type in ['ana_ano', 'ox']:
            do_display = DO_TYPES[do_type]
            
            # 收集该 (Period, DO) 下三个 Index 的 Top10 基因列表
            mt_genes = genem_data.get(do_type, {}).get(period, [])
            imp_genes = imp_data.get(do_type, {}).get(period, [])
            lef_genes = lef_data.get(do_type, {}).get(period, [])
            
            # 解析基因名称 (添加描述)
            mt_resolved = [resolve_gene_name(g, gene_map) for g in mt_genes]
            imp_resolved = [resolve_gene_name(g, gene_map) for g in imp_genes]
            lef_resolved = [resolve_gene_name(g, gene_map) for g in lef_genes]
            
            # 找出共同基因 (用原始 ID 匹配，不包含描述)
            mt_set = set(mt_genes)
            imp_set = set(imp_genes)
            lef_set = set(lef_genes)
            common_genes = mt_set & imp_set & lef_set
            
            # 共同基因也解析名称
            common_resolved = [resolve_gene_name(g, gene_map) for g in sorted(common_genes)]
            # ★ 不同 KO ID 之间用分号隔开
            common_str = '; '.join(common_resolved) if common_resolved else '-'
            
            # 构建三行 (MT, IMP, LEF)
            index_data = [
                ('MT', mt_resolved),
                ('IMP', imp_resolved),
                ('LEF', lef_resolved),
            ]
            
            for idx_name, gene_list in index_data:
                # ★ 不同 KO ID 之间用分号隔开
                gene_str = '; '.join(gene_list) if gene_list else '-'
                
                rows.append({
                    'Period': period,
                    'DO': do_display,
                    'Index': idx_name,
                    'Gene name': gene_str,
                    'Commonly identified important genes': common_str,
                })
    
    return pd.DataFrame(rows)


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("Results7: Top10 Gene Integration (Sep/Oct)")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载基因名称映射
    print("\n--- Step 1: 加载 Gene Info ---")
    gene_map = load_gene_info()
    
    # 2. 读取 MT (geneM) 数据
    print("\n--- Step 2: 读取 MT (geneM) 数据 ---")
    genem_data = read_genem_data()
    
    # 3. 读取 IMP 数据
    print("\n--- Step 3: 读取 IMP 数据 ---")
    imp_data = read_index_data(IMP_DIR, 'IMP')
    
    # 4. 读取 LEF (PDP) 数据
    print("\n--- Step 4: 读取 LEF (PDP) 数据 ---")
    lef_data = read_index_data(PDP_DIR, 'LEF')
    
    # 5. 构建表格
    print("\n--- Step 5: 构建 Top10Gene 表格 ---")
    df_result = build_top10gene_table(genem_data, imp_data, lef_data, gene_map)
    
    # 6. 保存结果
    print("\n--- Step 6: 保存结果 ---")
    df_result.to_excel(OUTPUT_FILE, index=False)
    print(f"已保存: {OUTPUT_FILE}")
    print(f"表格形状: {df_result.shape}")
    print("\n预览:")
    print(df_result.to_string())
    
    return df_result


if __name__ == '__main__':
    main()
