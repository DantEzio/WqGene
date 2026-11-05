"""
results7_top10gene.py
=====================
Purpose: integrate the geneM (MT), IMP and LEF (PDP) data of Sep and Oct,
extract the Top10 genes of each Period x DO x Index combination,
and find the important genes shared by the three indices of the same DO.

Output format:
  Period | DO       | Index | Gene name                              | Commonly identified important genes
  -------|----------|-------|----------------------------------------|--------------------------------------
  Sep    | Ana&ano  | MT    | K00366, K00368, ...                    | K00371, K04561, ...
  Sep    | Ana&ano  | IMP   | K04561, K03385, ...                    | K00371, K04561, ...
  Sep    | Ana&ano  | LEF   | K04561, K02586, ...                    | K00371, K04561, ...
  ...

Data sources:
  - MT (geneM):     ./data_GeneM/n_gene_dynamic.xlsx  (sheet: ana_ano, ox)
  - IMP:            ./results/IMP/*_Window2_0813-0913.xlsx (Sep)
                             *_Window3_0913-1013.xlsx (Oct)
  - LEF (PDP):      ./results/PDP/*_Window2_0813-0913.xlsx (Sep)
                             *_Window3_0913-1013.xlsx (Oct)
  - Gene name mapping: ./gene_ref/Gene_info.xlsx

Output:
    results/top10gene.xlsx (the Top10 genes and shared genes of each Period x DO x Index combination)

Usage:
    python results7_top10gene.py                 # run from the project root
"""

import os
import pandas as pd


# ============================================================
# Path configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GENEM_PATH = os.path.join(BASE_DIR, 'data_GeneM', 'n_gene_dynamic.xlsx')
IMP_DIR = os.path.join(BASE_DIR, 'results', 'IMP')
PDP_DIR = os.path.join(BASE_DIR, 'results', 'PDP')  # LEF data
GENE_INFO_PATH = os.path.join(BASE_DIR, 'gene_ref', 'Gene_info.xlsx')

OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'top10gene.xlsx')

TOP_N = 10  # number of top genes to extract

# DO type mapping
DO_TYPES = {
    'ana_ano': 'Ana&ano',
    'ox': 'Ox',
}

# Month mapping (derived from the date range in the filename)
PERIOD_MAP = {
    'Window2_0813-0913': 'Sep',
    'Window3_0913-1013': 'Oct',
}


# ============================================================
# 1. Load the gene name mapping table (Kxxxxx -> description)
# ============================================================
def load_gene_info():
    """Load Gene_info and return a KO -> Description mapping dict"""
    if not os.path.exists(GENE_INFO_PATH):
        print(f"[warning] Gene_info not found: {GENE_INFO_PATH}")
        return {}
    
    df = pd.read_excel(GENE_INFO_PATH)
    # Take the first description of each KO
    gene_map = dict(zip(df['KO'], df['Description']))
    print(f"[Gene_info] loaded {len(gene_map)} gene mappings")
    return gene_map


def resolve_gene_name(ko_id, gene_map):
    """
    resolve the gene names: extract the gene name from the Gene_info Description (keeping the original format).
    Description format:
      - "K00366  nirA; ferredoxin-nitrite reductase [EC:1.7.7.1]" -> "nirA"
      - "K00371  narH, narY, nxrB; nitrate reductase ..." -> "narH, narY, nxrB"
    returns the gene name of a single KO (comma-separated when there are several).
    """
    if not isinstance(ko_id, str):
        ko_id = str(ko_id)
    
    ko_id = ko_id.strip()
    
    if ko_id.upper().startswith('K'):
        desc = gene_map.get(ko_id.upper(), '')
        if desc:
            # Take the part before the semicolon (the gene name field)
            if ';' in desc:
                gene_part = desc.split(';')[0].strip()
            else:
                gene_part = desc.strip()
            
            # Strip a possible KO ID prefix (e.g. "K15864  nirS" -> "nirS")
            parts = gene_part.split()
            if len(parts) > 1 and parts[0].upper().startswith('K'):
                gene_part = ' '.join(parts[1:])
            
            return gene_part  # keep the original comma format: "narH, narY, nxrB"
        return ko_id
    return ko_id


# ============================================================
# 2. Read the MT (geneM) data
# ============================================================
def read_genem_data():
    """
    read n_gene_dynamic.xlsx and extract the top N genes for each DO type.
    returns: {do_type: {period: [gene_list]}}
    """
    result = {}
    
    if not os.path.exists(GENEM_PATH):
        print(f"[error] geneM file not found: {GENEM_PATH}")
        return result
    
    # Two sheets: ana_ano, ox
    for do_type in ['ana_ano', 'ox']:
        try:
            df = pd.read_excel(GENEM_PATH, sheet_name=do_type)
            print(f"\n[geneM] Sheet [{do_type}]: shape={df.shape}, cols={list(df.columns)}")
            
            # Column layout: [Sep_col, Oct_col, gene_name_col]
            cols = list(df.columns)
            gene_col = cols[-1]  # the gene name column is usually the last one
            
            result[do_type] = {}
            
            # Sep maps to column 1 (index 0), Oct to column 2 (index 1)
            period_cols = {
                'Sep': cols[0],
                'Oct': cols[1],
            }
            
            for period, col_name in period_cols.items():
                # Sort by value descending and take the top N
                df_sorted = df.sort_values(by=col_name, ascending=False)
                top_genes = df_sorted[gene_col].head(TOP_N).tolist()
                
                # Clean up the gene names
                top_genes = [str(g).strip() for g in top_genes if pd.notna(g)]
                
                result[do_type][period] = top_genes
                print(f"  {period} Top{TOP_N}: {top_genes[:3]}... ({len(top_genes)} genes in total)")
                
        except Exception as e:
            print(f"[error] failed to read geneM sheet [{do_type}]: {e}")
    
    return result


# ============================================================
# 3. Read the IMP / LEF (PDP) data
# ============================================================
def read_index_data(data_dir, index_name):
    """
    read the data under the IMP or PDP directory, grouped by DO type and month.
    
    Args:
        data_dir: data directory (IMP_DIR or PDP_DIR)
        index_name: 'IMP' or 'LEF'
    
    Returns:
        {do_type: {period: [gene_list]}}
    """
    result = {}
    
    if not os.path.exists(data_dir):
        print(f"[error] directory not found: {data_dir}")
        return result
    
    for do_type in ['ana_ano', 'ox']:
        result[do_type] = {}
        
        for window_key, period in PERIOD_MAP.items():
            # Build the filename: {do_type}_{window_key}.xlsx
            filename = f"{do_type}_{window_key}.xlsx"
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"[warning] {index_name} file not found: {filename}")
                result[do_type][period] = []
                continue
            
            try:
                df = pd.read_excel(filepath)
                # Take the top N genes (already sorted by importance)
                top_genes = df['gene'].head(TOP_N).tolist()
                top_genes = [str(g).strip() for g in top_genes if pd.notna(g)]
                
                result[do_type][period] = top_genes
                print(f"[{index_name}] {filename}: Top{TOP_N} = {top_genes[:3]}... ({len(top_genes)} genes in total)")
                
            except Exception as e:
                print(f"[error] failed to read the {index_name} file {filename}: {e}")
                result[do_type][period] = []
    
    return result


# ============================================================
# 4. Build the Top10Gene table
# ============================================================
def build_top10gene_table(genem_data, imp_data, lef_data, gene_map):
    """
    integrate all data sources and build the final table.
    
    Table layout:
      Period | DO | Index | Gene name | Common genes
    """
    rows = []
    
    # Loop over every Period x DO x Index combination
    for period in ['Sep', 'Oct']:
        for do_type in ['ana_ano', 'ox']:
            do_display = DO_TYPES[do_type]
            
            # Collect the Top10 gene lists of the three indices for this (Period, DO)
            mt_genes = genem_data.get(do_type, {}).get(period, [])
            imp_genes = imp_data.get(do_type, {}).get(period, [])
            lef_genes = lef_data.get(do_type, {}).get(period, [])
            
            # Resolve the gene names (adding the description)
            mt_resolved = [resolve_gene_name(g, gene_map) for g in mt_genes]
            imp_resolved = [resolve_gene_name(g, gene_map) for g in imp_genes]
            lef_resolved = [resolve_gene_name(g, gene_map) for g in lef_genes]
            
            # Find the common genes (matched by raw ID, excluding descriptions)
            mt_set = set(mt_genes)
            imp_set = set(imp_genes)
            lef_set = set(lef_genes)
            common_genes = mt_set & imp_set & lef_set
            
            # Resolve the names of the common genes as well
            common_resolved = [resolve_gene_name(g, gene_map) for g in sorted(common_genes)]
            # KO IDs are separated by semicolons
            common_str = '; '.join(common_resolved) if common_resolved else '-'
            
            # Build three rows (MT, IMP, LEF)
            index_data = [
                ('MT', mt_resolved),
                ('IMP', imp_resolved),
                ('LEF', lef_resolved),
            ]
            
            for idx_name, gene_list in index_data:
                # KO IDs are separated by semicolons
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
# Main function
# ============================================================
def main():
    print("=" * 60)
    print("Results7: Top10 Gene Integration (Sep/Oct)")
    print("=" * 60)
    
    # Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load the gene name mapping
    print("\n--- Step 1: load Gene Info ---")
    gene_map = load_gene_info()
    
    # 2. Read the MT (geneM) data
    print("\n--- Step 2: read the MT (geneM) data ---")
    genem_data = read_genem_data()
    
    # 3. Read the IMP data
    print("\n--- Step 3: read the IMP data ---")
    imp_data = read_index_data(IMP_DIR, 'IMP')
    
    # 4. Read the LEF (PDP) data
    print("\n--- Step 4: read the LEF (PDP) data ---")
    lef_data = read_index_data(PDP_DIR, 'LEF')
    
    # 5. Build the table
    print("\n--- Step 5: build the Top10Gene table ---")
    df_result = build_top10gene_table(genem_data, imp_data, lef_data, gene_map)
    
    # 6. Save results
    print("\n--- Step 6: save the results ---")
    df_result.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Table shape: {df_result.shape}")
    print("\nPreview:")
    print(df_result.to_string())
    
    return df_result


if __name__ == '__main__':
    main()
