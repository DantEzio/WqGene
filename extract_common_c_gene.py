"""
提取 data_Gene 和 data_GeneM 中 c_gene 的共同 KEGG 行数据。
与 extract_common_n_gene.py 逻辑相同，仅文件名和输出路径不同。
"""
import pandas as pd
import os
import re

# ============================================================
# 配置：输入/输出路径
# ============================================================
BASE_DIR = r"d:\Github\AI-ML\AI-Bio"

GENE_PATH = os.path.join(BASE_DIR, "data_Gene", "c_gene_dynamic.xls")
GENEM_PATH = os.path.join(BASE_DIR, "data_GeneM", "c_gene_dynamic.xlsx")

GENE_OUT = os.path.join(BASE_DIR, "data_Gene", "c_gene_dynamic_common.xlsx")
GENEM_OUT = os.path.join(BASE_DIR, "data_GeneM", "c_gene_dynamic_common.xlsx")

SHEET_NAMES = ["ana_ano", "ox"]


def extract_kegg_ids(df):
    """从 DataFrame 最后一列提取 KEGG ID 列表（保留原始顺序）"""
    last_col = df.iloc[:, -1].astype(str)
    kegg_ids = last_col.str.extract(r"(K\d+)")[0]
    return kegg_ids.dropna().tolist()


def process_sheet(gene_df, genem_df, sheet_name):
    """
    处理单个 sheet:
    1. 提取两个文件的 KEGG ID
    2. 找出交集
    3. 按交集过滤两边的行
    4. 返回过滤后的 DataFrame
    """
    gene_keggs = extract_kegg_ids(gene_df)
    genem_keggs = extract_kegg_ids(genem_df)

    gene_kegg_set = set(gene_keggs)
    genem_kegg_set = set(genem_keggs)
    common_keggs = gene_kegg_set & genem_kegg_set

    print(f"\n{'='*60}")
    print(f"Sheet: {sheet_name}")
    print(f"  data_Gene 原始行数:   {len(gene_df)}, KEGG 数: {len(gene_kegg_set)}")
    print(f"  data_GeneM 原始行数:  {len(genem_df)}, KEGG 数: {len(genem_kegg_set)}")
    print(f"  共同 KEGG 数量:      {len(common_keggs)}")
    
    if gene_kegg_set - common_keggs:
        print(f"  data_Gene 独有 ({len(gene_kegg_set - common_keggs)}): {sorted(gene_kegg_set - common_keggs)[:10]}{'...' if len(gene_kegg_set - common_keggs) > 10 else ''}")
    if genem_kegg_set - common_keggs:
        print(f"  data_GeneM 独有 ({len(genem_kegg_set - common_keggs)}): {sorted(genem_kegg_set - common_keggs)[:10]}{'...' if len(genem_kegg_set - common_keggs) > 10 else ''}")

    # 构建映射: KEGG ID -> 原始行索引（保留首次出现的行）
    def build_kegg_row_map(df, kegg_list):
        last_col = df.iloc[:, -1].astype(str)
        extracted = last_col.str.extract(r"(K\d+)")[0]
        mapping = {}
        for idx, kegg in extracted.items():
            if pd.notna(kegg) and kegg not in mapping:
                mapping[kegg] = idx
        return mapping

    gene_map = build_kegg_row_map(gene_df, gene_keggs)
    genem_map = build_kegg_row_map(genem_df, genem_keggs)

    # 按共同 KEGG 在 data_Gene 中的原始顺序排序
    sorted_common = [k for k in gene_keggs if k in common_keggs]

    gene_rows = [gene_map[k] for k in sorted_common if k in gene_map]
    genem_rows = [genem_map[k] for k in sorted_common if k in genem_map]

    gene_filtered = gene_df.loc[gene_rows].reset_index(drop=True)
    genem_filtered = genem_df.loc[genem_rows].reset_index(drop=True)

    print(f"  过滤后 data_Gene:     {gene_filtered.shape[0]} 行 x {gene_filtered.shape[1]} 列")
    print(f"  过滤后 data_GeneM:    {genem_filtered.shape[0]} 行 x {genem_filtered.shape[1]} 列")

    return gene_filtered, genem_filtered, common_keggs


def main():
    print("=" * 60)
    print("c_gene 共同行提取工具")
    print("=" * 60)
    print(f"data_Gene 输入:   {GENE_PATH}")
    print(f"data_GeneM 输入:  {GENEM_PATH}")
    print(f"data_Gene 输出:   {GENE_OUT}")
    print(f"data_GeneM 输出:  {GENEM_OUT}")

    # 读取所有 sheets
    gene_sheets = pd.read_excel(GENE_PATH, sheet_name=None, engine="xlrd")
    genem_sheets = pd.read_excel(GENEM_PATH, sheet_name=None, engine="openpyxl")

    all_common_keggs = {}

    with pd.ExcelWriter(GENE_OUT, engine="openpyxl") as gene_writer, \
         pd.ExcelWriter(GENEM_OUT, engine="openpyxl") as genem_writer:

        for sheet_name in SHEET_NAMES:
            if sheet_name not in gene_sheets or sheet_name not in genem_sheets:
                print(f"\n警告: Sheet '{sheet_name}' 不存在于某个文件中，跳过")
                continue

            gene_f, genem_f, common = process_sheet(
                gene_sheets[sheet_name], genem_sheets[sheet_name], sheet_name
            )
            all_common_keggs[sheet_name] = common

            gene_f.to_excel(gene_writer, sheet_name=sheet_name, index=False)
            genem_f.to_excel(genem_writer, sheet_name=sheet_name, index=False)

    print(f"\n{'='*60}")
    print("完成! 输出文件:")
    print(f"  data_Gene:   {GENE_OUT}")
    print(f"  data_GeneM:  {GENEM_OUT}")


if __name__ == "__main__":
    main()
