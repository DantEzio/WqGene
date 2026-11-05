"""
extract_common_n_gene.py - extract the shared (common KEGG) rows of n_gene
=========================================================
Purpose:
    read the n_gene dynamic expression data of data_Gene and data_GeneM (ana_ano / ox sheets),
    take the intersection of the KEGG IDs at the end of each row (e.g. K00001) and keep only rows present on both sides,
    and re-order them following the original row order of data_Gene before saving the *_common.xlsx files.
    the goal is to fully align the gene feature dimensions of the source domain (data_Gene) and the target domain (data_GeneM),
    so the input space stays identical during transfer learning.
Input:
    data_Gene/n_gene_dynamic.xls      (xls format, requires the xlrd engine)
    data_GeneM/n_gene_dynamic.xlsx    (xlsx format, openpyxl engine)
Output:
    data_Gene/n_gene_dynamic_common.xlsx
    data_GeneM/n_gene_dynamic_common.xlsx
Usage:
    python extract_common_n_gene.py
Note:
    identical logic to extract_common_c_gene.py, only the filenames and output paths differ
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
    """Read the ana_ano and ox sheets of an Excel file and return a dict of raw DataFrames."""
    sheets = {}
    for name in SHEET_NAMES:
        df = pd.read_excel(file_path, sheet_name=name)
        sheets[name] = df
    return sheets


def extract_kegg_ids(df):
    """Extract KEGG IDs from the last DataFrame column and return a Series aligned with the df rows."""
    last_col = df.iloc[:, -1].astype(str)
    return last_col.str.extract(r'(K\d+)')[0]


def filter_by_kegg(df, kegg_ids):
    """Filter rows by KEGG ID while keeping the original column structure. Returns the filtered DataFrame."""
    kegg_series = extract_kegg_ids(df)
    mask = kegg_series.isin(kegg_ids) & kegg_series.notna()
    return df[mask].reset_index(drop=True)


def main():
    print("=" * 70)
    print("Extract the shared KEGG rows of n_gene from data_Gene and data_GeneM")
    print("=" * 70)

    print(f"\nReading files:")
    print(f"  - data_Gene:   {GENE_PATH}")
    print(f"  - data_GeneM:  {GENEM_PATH}")

    gene_sheets = load_gene_sheets(GENE_PATH)
    genem_sheets = load_gene_sheets(GENEM_PATH)

    gene_filtered = {}
    genem_filtered = {}

    for sheet_name in SHEET_NAMES:
        print(f"\n{'-' * 50}")
        print(f"Processing sheet: {sheet_name}")
        print(f"{'-' * 50}")

        df_g = gene_sheets[sheet_name]
        df_m = genem_sheets[sheet_name]

        kegg_g = extract_kegg_ids(df_g).dropna()
        kegg_m = extract_kegg_ids(df_m).dropna()

        set_g = set(kegg_g.tolist())
        set_m = set(kegg_m.tolist())
        common = set_g & set_m

        print(f"  data_Gene  raw rows: {len(df_g)}, KEGG ID count: {len(set_g)}")
        print(f"  data_GeneM raw rows: {len(df_m)}, KEGG ID count: {len(set_m)}")
        print(f"  Common KEGG ID count: {len(common)}")

        only_g = sorted(set_g - set_m)
        only_m = sorted(set_m - set_g)
        if only_g:
            print(f"  data_Gene unique: {only_g}")
        if only_m:
            print(f"  data_GeneM unique: {only_m}")
        print(f"  Common KEGG IDs: {sorted(common)}")

        gene_filtered[sheet_name] = filter_by_kegg(df_g, common)
        genem_filtered[sheet_name] = filter_by_kegg(df_m, common)

        print(f"  Filtered data_Gene  rows: {len(gene_filtered[sheet_name])}")
        print(f"  Filtered data_GeneM rows: {len(genem_filtered[sheet_name])}")

    # Save as xlsx, keeping both sheets
    with pd.ExcelWriter(GENE_OUT, engine='openpyxl') as writer:
        for sheet_name in SHEET_NAMES:
            gene_filtered[sheet_name].to_excel(
                writer, sheet_name=sheet_name, index=False)

    with pd.ExcelWriter(GENEM_OUT, engine='openpyxl') as writer:
        for sheet_name in SHEET_NAMES:
            genem_filtered[sheet_name].to_excel(
                writer, sheet_name=sheet_name, index=False)

    print(f"\n{'=' * 70}")
    print("Save complete:")
    print(f"  - data_Gene  -> {GENE_OUT}")
    print(f"  - data_GeneM -> {GENEM_OUT}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
