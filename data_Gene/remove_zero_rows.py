"""
以 data_GeneM/c_gene_dynamic_common.xlsx 中的 KO（KEGG ID）列为基准，
从 data_Gene/c_gene_dynamic.xlsx 中提取同名行，
保存为 data_Gene/c_gene_dynamic_common.xlsx。

处理 ana_ano 和 ox 两个 sheet：
1. 从 data_GeneM 的 common 文件提取 KO 列表（保留顺序、去重）
2. 从 data_Gene 的 c_gene_dynamic.xlsx 提取每行 KO
3. 按 data_GeneM 的 KO 顺序，从 data_Gene 中取出对应行
4. 写入 data_Gene/c_gene_dynamic_common.xlsx
"""
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # data_Gene
PROJECT_DIR = os.path.dirname(BASE_DIR)                     # AI-Bio

# 参考文件（KO 基准）
REF_FILE = os.path.join(PROJECT_DIR, 'data_GeneM', 'c_gene_dynamic_common.xlsx')
# 源数据文件
SRC_FILE = os.path.join(BASE_DIR, 'c_gene_dynamic.xlsx')
# 输出文件
OUT_FILE = os.path.join(BASE_DIR, 'c_gene_dynamic_common.xlsx')

SHEET_NAMES = ['ana_ano', 'ox']


def extract_ko_series(df):
    """提取每行的 KO（K+数字）。
    优先使用名为 'KO' 的列；否则从最后一列用正则提取。
    返回与 df 行对齐的 Series。
    """
    for c in df.columns:
        if str(c).strip().upper() == 'KO':
            return df[c].astype(str).str.extract(r'(K\d+)')[0]
    return df.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]


def main():
    print('=' * 60)
    print('以 data_GeneM common 的 KO 为基准，提取 data_Gene 同名行')
    print('=' * 60)
    print(f'参考文件(KO基准): {REF_FILE}')
    print(f'源数据文件:       {SRC_FILE}')
    print(f'输出文件:         {OUT_FILE}')

    ref_sheets = pd.read_excel(REF_FILE, sheet_name=SHEET_NAMES, engine='openpyxl')
    src_sheets = pd.read_excel(SRC_FILE, sheet_name=SHEET_NAMES, engine='openpyxl')

    with pd.ExcelWriter(OUT_FILE, engine='openpyxl') as writer:
        for name in SHEET_NAMES:
            print(f"\n{'-' * 50}")
            print(f'Sheet: {name}')

            ref_df = ref_sheets[name]
            src_df = src_sheets[name]

            # 1. 基准 KO 列表（按 data_GeneM 出现顺序，去重）
            ref_ko = extract_ko_series(ref_df)
            ref_order, seen = [], set()
            for ko in ref_ko:
                if pd.notna(ko) and ko not in seen:
                    seen.add(ko)
                    ref_order.append(ko)
            print(f'  基准 KO 数量(data_GeneM): {len(ref_order)}')

            # 2. 源数据 KO -> 行索引（首次出现）
            src_ko = extract_ko_series(src_df)
            src_map = {}
            for idx, ko in src_ko.items():
                if pd.notna(ko) and ko not in src_map:
                    src_map[ko] = idx
            print(f'  源数据 KO 数量(data_Gene): {len(src_map)}')

            # 3. 按基准顺序取行；缺失的 KO 跳过并提示
            rows, missing = [], []
            for ko in ref_order:
                if ko in src_map:
                    rows.append(src_map[ko])
                else:
                    missing.append(ko)

            if missing:
                print(f'  警告: {len(missing)} 个基准 KO 在源数据中不存在，已跳过: '
                      f'{missing[:10]}{"..." if len(missing) > 10 else ""}')

            out_df = src_df.loc[rows].reset_index(drop=True)
            print(f'  输出行数: {len(out_df)} x {out_df.shape[1]} 列')

            out_df.to_excel(writer, sheet_name=name, index=False)

    print(f"\n{'=' * 60}")
    print(f'保存完成: {OUT_FILE}')
    print('=' * 60)


if __name__ == '__main__':
    main()
