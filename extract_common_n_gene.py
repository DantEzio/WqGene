"""提取 data_Gene 和 data_GeneM 中 n_gene 数据的共同 KEGG 行。

读取两个 n_gene Excel 文件（ana_ano 和 ox 两个 sheet），
按 KEGG ID 取交集，分别过滤并保存为新的 xlsx 文件。
"""
import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENE_PATH = os.path.join(BASE_DIR, 'data_Gene', 'n_gene_dynamic.xls')
GENEM_PATH = os.path.join(BASE_DIR, 'data_GeneM', 'n_gene_dynamic.xlsx')
GENE_OUT = os.path.join(BASE_DIR, 'data_Gene', 'n_gene_dynamic_common.xlsx')
GENEM_OUT = os.path.join(BASE_DIR, 'data_GeneM', 'n_gene_dynamic_common.xlsx')

SHEET_NAMES = ['ana_ano', 'ox']


def load_gene_sheets(file_path):
    """读取 Excel 文件的 ana_ano 和 ox 两个 sheet，返回原始 DataFrame dict。"""
    sheets = {}
    for name in SHEET_NAMES:
        df = pd.read_excel(file_path, sheet_name=name)
        sheets[name] = df
    return sheets


def extract_kegg_ids(df):
    """从 DataFrame 最后一列提取 KEGG ID，返回与 df 行对齐的 Series。"""
    last_col = df.iloc[:, -1].astype(str)
    return last_col.str.extract(r'(K\d+)')[0]


def filter_by_kegg(df, kegg_ids):
    """按 KEGG ID 过滤行，保留原始列结构。返回过滤后的 DataFrame。"""
    kegg_series = extract_kegg_ids(df)
    mask = kegg_series.isin(kegg_ids) & kegg_series.notna()
    return df[mask].reset_index(drop=True)


def main():
    print("=" * 70)
    print("提取 data_Gene 和 data_GeneM 中 n_gene 的共同 KEGG 行")
    print("=" * 70)

    print(f"\n读取文件:")
    print(f"  - data_Gene:   {GENE_PATH}")
    print(f"  - data_GeneM:  {GENEM_PATH}")

    gene_sheets = load_gene_sheets(GENE_PATH)
    genem_sheets = load_gene_sheets(GENEM_PATH)

    gene_filtered = {}
    genem_filtered = {}

    for sheet_name in SHEET_NAMES:
        print(f"\n{'-' * 50}")
        print(f"处理 sheet: {sheet_name}")
        print(f"{'-' * 50}")

        df_g = gene_sheets[sheet_name]
        df_m = genem_sheets[sheet_name]

        kegg_g = extract_kegg_ids(df_g).dropna()
        kegg_m = extract_kegg_ids(df_m).dropna()

        set_g = set(kegg_g.tolist())
        set_m = set(kegg_m.tolist())
        common = set_g & set_m

        print(f"  data_Gene  原始行数: {len(df_g)}, KEGG ID 数: {len(set_g)}")
        print(f"  data_GeneM 原始行数: {len(df_m)}, KEGG ID 数: {len(set_m)}")
        print(f"  共同 KEGG ID 数: {len(common)}")

        only_g = sorted(set_g - set_m)
        only_m = sorted(set_m - set_g)
        if only_g:
            print(f"  data_Gene 独有: {only_g}")
        if only_m:
            print(f"  data_GeneM 独有: {only_m}")
        print(f"  共同 KEGG ID: {sorted(common)}")

        gene_filtered[sheet_name] = filter_by_kegg(df_g, common)
        genem_filtered[sheet_name] = filter_by_kegg(df_m, common)

        print(f"  过滤后 data_Gene  行数: {len(gene_filtered[sheet_name])}")
        print(f"  过滤后 data_GeneM 行数: {len(genem_filtered[sheet_name])}")

    # 保存为 xlsx，保留两个 sheet
    with pd.ExcelWriter(GENE_OUT, engine='openpyxl') as writer:
        for sheet_name in SHEET_NAMES:
            gene_filtered[sheet_name].to_excel(
                writer, sheet_name=sheet_name, index=False)

    with pd.ExcelWriter(GENEM_OUT, engine='openpyxl') as writer:
        for sheet_name in SHEET_NAMES:
            genem_filtered[sheet_name].to_excel(
                writer, sheet_name=sheet_name, index=False)

    print(f"\n{'=' * 70}")
    print("保存完成:")
    print(f"  - data_Gene  -> {GENE_OUT}")
    print(f"  - data_GeneM -> {GENEM_OUT}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
