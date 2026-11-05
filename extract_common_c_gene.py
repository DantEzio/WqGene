"""
extract_common_c_gene.py - extract the shared (common KEGG) rows of c_gene
=========================================================
Purpose:
    read the c_gene dynamic expression data of data_Gene and data_GeneM (ana_ano / ox sheets),
    take the intersection of the KEGG IDs at the end of each row (e.g. K00001) and keep only rows present on both sides,
    and re-order them following the original row order of data_Gene before saving the *_common.xlsx files.
    the goal is to fully align the gene feature dimensions of the source domain (data_Gene) and the target domain (data_GeneM),
    so the input space stays identical during transfer learning.
Input:
    data_Gene/c_gene_dynamic.xls      (xls format, requires the xlrd engine)
    data_GeneM/c_gene_dynamic.xlsx    (xlsx format, openpyxl engine)
Output:
    data_Gene/c_gene_dynamic_common.xlsx
    data_GeneM/c_gene_dynamic_common.xlsx
Usage:
    python extract_common_c_gene.py
Note:
    identical logic to extract_common_n_gene.py, only the filenames and output paths differ
"""
import pandas as pd
import os
import re

# ============================================================
# Configuration: input/output paths (the script directory is the project root)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GENE_PATH = os.path.join(BASE_DIR, "data_Gene", "c_gene_dynamic.xls")
GENEM_PATH = os.path.join(BASE_DIR, "data_GeneM", "c_gene_dynamic.xlsx")

GENE_OUT = os.path.join(BASE_DIR, "data_Gene", "c_gene_dynamic_common.xlsx")
GENEM_OUT = os.path.join(BASE_DIR, "data_GeneM", "c_gene_dynamic_common.xlsx")

SHEET_NAMES = ["ana_ano", "ox"]


def extract_kegg_ids(df):
    """Extract KEGG IDs from the last DataFrame column (keeping the original order)"""
    last_col = df.iloc[:, -1].astype(str)
    kegg_ids = last_col.str.extract(r"(K\d+)")[0]
    return kegg_ids.dropna().tolist()


def process_sheet(gene_df, genem_df, sheet_name):
    """
    Handle a single sheet:
    1. Extract the KEGG IDs of both files
    2. Compute the intersection
    3. Filter both sides by the intersection
    4. Return the filtered DataFrame
    """
    gene_keggs = extract_kegg_ids(gene_df)
    genem_keggs = extract_kegg_ids(genem_df)

    gene_kegg_set = set(gene_keggs)
    genem_kegg_set = set(genem_keggs)
    common_keggs = gene_kegg_set & genem_kegg_set

    print(f"\n{'='*60}")
    print(f"Sheet: {sheet_name}")
    print(f"  data_Gene raw rows:   {len(gene_df)}, KEGG count: {len(gene_kegg_set)}")
    print(f"  data_GeneM raw rows:  {len(genem_df)}, KEGG count: {len(genem_kegg_set)}")
    print(f"  Common KEGG count:      {len(common_keggs)}")
    
    if gene_kegg_set - common_keggs:
        print(f"  data_Gene unique ({len(gene_kegg_set - common_keggs)}): {sorted(gene_kegg_set - common_keggs)[:10]}{'...' if len(gene_kegg_set - common_keggs) > 10 else ''}")
    if genem_kegg_set - common_keggs:
        print(f"  data_GeneM unique ({len(genem_kegg_set - common_keggs)}): {sorted(genem_kegg_set - common_keggs)[:10]}{'...' if len(genem_kegg_set - common_keggs) > 10 else ''}")

    # Build the mapping: KEGG ID -> original row index (first occurrence kept)
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

    # Order by the original order of the common KEGG ids in data_Gene
    sorted_common = [k for k in gene_keggs if k in common_keggs]

    gene_rows = [gene_map[k] for k in sorted_common if k in gene_map]
    genem_rows = [genem_map[k] for k in sorted_common if k in genem_map]

    gene_filtered = gene_df.loc[gene_rows].reset_index(drop=True)
    genem_filtered = genem_df.loc[genem_rows].reset_index(drop=True)

    print(f"  Filtered data_Gene:     {gene_filtered.shape[0]} rows x {gene_filtered.shape[1]} cols")
    print(f"  Filtered data_GeneM:    {genem_filtered.shape[0]} rows x {genem_filtered.shape[1]} cols")

    return gene_filtered, genem_filtered, common_keggs


def main():
    print("=" * 60)
    print("c_gene common-row extraction tool")
    print("=" * 60)
    print(f"data_Gene input:   {GENE_PATH}")
    print(f"data_GeneM input:  {GENEM_PATH}")
    print(f"data_Gene output:   {GENE_OUT}")
    print(f"data_GeneM output:  {GENEM_OUT}")

    # Read all sheets
    gene_sheets = pd.read_excel(GENE_PATH, sheet_name=None, engine="xlrd")
    genem_sheets = pd.read_excel(GENEM_PATH, sheet_name=None, engine="openpyxl")

    all_common_keggs = {}

    with pd.ExcelWriter(GENE_OUT, engine="openpyxl") as gene_writer, \
         pd.ExcelWriter(GENEM_OUT, engine="openpyxl") as genem_writer:

        for sheet_name in SHEET_NAMES:
            if sheet_name not in gene_sheets or sheet_name not in genem_sheets:
                print(f"\nWarning: sheet '{sheet_name}' is missing from one of the files, skipping")
                continue

            gene_f, genem_f, common = process_sheet(
                gene_sheets[sheet_name], genem_sheets[sheet_name], sheet_name
            )
            all_common_keggs[sheet_name] = common

            gene_f.to_excel(gene_writer, sheet_name=sheet_name, index=False)
            genem_f.to_excel(genem_writer, sheet_name=sheet_name, index=False)

    print(f"\n{'='*60}")
    print("Done! Output files:")
    print(f"  data_Gene:   {GENE_OUT}")
    print(f"  data_GeneM:  {GENEM_OUT}")


if __name__ == "__main__":
    main()
