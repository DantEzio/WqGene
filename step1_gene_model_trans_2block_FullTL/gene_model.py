"""
gene_model.py - two-branch GeneModel definition and data handling
=========================================================
the core module of the project's main model, imported by the training / testing / interpretability scripts.

Main components:
    1) GeneModel            : two-branch network. The WQ branch encodes each pool (INF / ANA+ANO / OX / EFF)
                              encode separately; the gene branch projects n_gene (21 dims) and c_gene (1205 dims)
                              projected into the hidden space and fused with the WQ features (2block: ANA and ANO
                              merged into one block), producing 6 dims (ANA/ANO/OX x DO/AN/NN).
    2) calculate_loss       : multi-output MSE loss (optionally weighted).
    3) WaterQualityDataset  : looks up the monthly gene profile by timestamp and builds window-length samples.
    4) prepare_data         : reads wq_scaled.csv / wq.xlsx and the *_common.xlsx gene tables,
                              after standardization, split chronologically into train / validation / test sets.

Input data:
    ../data_WQ/wq_scaled.csv, ../data_WQ/wq.xlsx
    ../data_Gene/n_gene_dynamic_common.xlsx, ../data_Gene/c_gene_dynamic_common.xlsx

Usage:
    imported as a module by train_gene_model.py / test_gene_model*.py / results4_* and other scripts
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

class GeneModel(nn.Module):
    def __init__(self, hidden_dim=256, dropout_rate=0.3,
                 # Merged gene dimension: ana and ano share the same gene data
                 gene_n_ana_ano_dim=21,    # n_gene ana_ano (21 KEGG IDs)
                 gene_n_ox_dim=21,          # n_gene ox (21 KEGG IDs)
                 gene_c_ana_ano_dim=1205,   # c_gene ana_ano (1205 KEGG IDs)
                 gene_c_ox_dim=1205):       # c_gene ox (1205 KEGG IDs)
        super().__init__()
        # ===== Input dimension definition (ANA+ANO merged) =====
        self.wq_inf_dim = 7  # F-INF, T-INF, pH-INF, COD-INF, C/N-INF, TN-INF, AN-INF
        self.wq_ana_ano_dim = 6  # merged: DO/AN/NN-ANA + DO/AN/NN-ANO = 6
        self.wq_ox_dim = 3   # DO-OX, AN-OX, NN-OX
        self.wq_eff_dim = 3   # COD-EFF, TN-EFF, AN-EFF
        
        # Gene dimension
        self.gene_n_ana_ano_dim = gene_n_ana_ano_dim   # n_gene (ana_ano sheet)
        self.gene_n_ox_dim = gene_n_ox_dim              # n_gene (ox sheet)
        self.gene_c_ana_ano_dim = gene_c_ana_ano_dim   # c_gene (ana_ano sheet)
        self.gene_c_ox_dim = gene_c_ox_dim              # c_gene (ox sheet)
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # ===== Input projection layer =====
        self.inf_proj = nn.Sequential(
            nn.Linear(self.wq_inf_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Merge: ANA+ANO projection layer (6-dim input)
        self.ana_ano_proj = nn.Sequential(
            nn.Linear(self.wq_ana_ano_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.ox_proj = nn.Sequential(
            nn.Linear(self.wq_ox_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.eff_proj = nn.Sequential(
            nn.Linear(self.wq_eff_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ===== n_gene feature projection layer (ANA+ANO merged) =====
        # Original gene_n_proj_ana + gene_n_proj_ano -> merged into one
        self.gene_n_proj_ana_ano = nn.Sequential(
            nn.Linear(self.gene_n_ana_ano_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.wq_ana_ano_dim),  # 6-dim output
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # The n_gene projection for OX stays unchanged
        self.gene_n_proj_ox = nn.Sequential(
            nn.Linear(self.gene_n_ox_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.wq_ox_dim),  # 3-dim output
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ===== c_gene feature projection layer (ANA+ANO merged) =====
        c_gene_hidden = hidden_dim * 2  # 512
        
        # Original gene_c_proj_ana + gene_c_proj_ano -> merged into one
        self.gene_c_proj_ana_ano = nn.Sequential(
            nn.Linear(self.gene_c_ana_ano_dim, c_gene_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(c_gene_hidden, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.wq_ana_ano_dim),  # 6-dim output
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # The c_gene projection for OX stays unchanged
        self.gene_c_proj_ox = nn.Sequential(
            nn.Linear(self.gene_c_ox_dim, c_gene_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(c_gene_hidden, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.wq_ox_dim),  # 3-dim output
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ===== Gene fusion layer (ANA+ANO merged) =====
        # Original fusion_ana + fusion_ano -> merged into one: (6+6)->6
        self.gene_fusion_ana_ano = nn.Linear(self.wq_ana_ano_dim * 2, self.wq_ana_ano_dim)
        # OX fusion unchanged: (3+3)->3
        self.gene_fusion_ox = nn.Linear(self.wq_ox_dim * 2, self.wq_ox_dim)
        
        # ===== LSTM framework =====
        # Input dimension adjustment: inf(7) + ana_ano(6) + ox(3) + eff(3) = 19 (unchanged!)
        self.lstm = nn.LSTM(
            input_size=self.wq_inf_dim + self.wq_ana_ano_dim + self.wq_ox_dim + self.wq_eff_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            bidirectional=False,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # LSTM output processing
        self.lstm_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ===== Output layer (separate ANA and ANO heads) =====
        # Split back into output_ana(3) + output_ano(3) to match the original model format
        self.output_ana = nn.Linear(self.hidden_dim, 3)     # [DO-ANA, AN-ANA, NN-ANA]
        self.output_ano = nn.Linear(self.hidden_dim, 3)     # [DO-ANO, AN-ANO, NN-ANO]
        self.output_ox = nn.Linear(self.hidden_dim, 3)      # unchanged
        self.output_eff = nn.Linear(self.hidden_dim, 3)     # unchanged
    
    def forward(self, wq_inf, wq_ana_ano, wq_ox, wq_eff,
                gene_n_ana_ano, gene_n_ox,
                gene_c_ana_ano, gene_c_ox):
        """
        Parameter changes:
        - Removed: wq_ana, wq_ano, gene_n_ana, gene_n_ano, gene_c_ana, gene_c_ano
        - Added: wq_ana_ano, gene_n_ana_ano, gene_c_ana_ano
        """
        # Input shape: (batch, timestep, dim)
        batch_size, timestep, _ = wq_inf.shape
        
        # Input projection (the same projection is applied at every time step)
        inf_proj = wq_inf  # (batch, timestep, wq_inf_dim)
        ana_ano_proj = wq_ana_ano  # (batch, timestep, wq_ana_ano_dim=6) 
        ox_proj = wq_ox    # (batch, timestep, wq_ox_dim)
        eff_proj = wq_eff  # (batch, timestep, wq_eff_dim)
        
        # ---- n_gene feature projection (merged) ----
        gene_n_ana_ano_proj = self.gene_n_proj_ana_ano(gene_n_ana_ano)  # (batch, 6) 
        gene_n_ox_proj = self.gene_n_proj_ox(gene_n_ox)               # (batch, 3)
        
        # ---- c_gene feature projection (merged) ----
        gene_c_ana_ano_proj = self.gene_c_proj_ana_ano(gene_c_ana_ano)  # (batch, 6) 
        gene_c_ox_proj = self.gene_c_proj_ox(gene_c_ox)                # (batch, 3)
        
        # ---- concatenate n_gene and c_gene, then fuse (merged) ----
        # concat n(6) + c(6) = 12, then project to 6
        gene_ana_ano_fused = self.gene_fusion_ana_ano(
            torch.cat([gene_n_ana_ano_proj, gene_c_ana_ano_proj], dim=-1))  # (batch, 6) 
        
        # OX fusion: concat n(3) + c(3) = 6, then project to 3
        gene_ox_fused = self.gene_fusion_ox(
            torch.cat([gene_n_ox_proj, gene_c_ox_proj], dim=-1))  # (batch, 3)
        
        # Expand the gene features along the time dimension
        gene_ana_ano_fused = gene_ana_ano_fused.unsqueeze(1).repeat(1, timestep, 1)  # (batch, timestep, 6) 
        gene_ox_fused = gene_ox_fused.unsqueeze(1).repeat(1, timestep, 1)             # (batch, timestep, 3)
        
        # Fuse by summing the components
        fusion_ana_ano = ana_ano_proj + gene_ana_ano_fused  # (batch, timestep, 6) 
        fusion_ox = ox_proj + gene_ox_fused                  # (batch, timestep, 3)
        
        # Concatenate into the big input tensor: (batch, timestep, 7+6+3+3=19)
        concat_features = torch.cat([
            inf_proj,           # wq_inf_dim=7
            fusion_ana_ano,     # wq_ana_ano_dim=6 
            fusion_ox,          # wq_ox_dim=3
            eff_proj            # wq_eff_dim=3
        ], dim=-1)  # total = 19
        
        # Feed into the LSTM
        lstm_out, _ = self.lstm(concat_features)  # (batch, timestep, hidden_dim)
        
        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]
        
        # Output the predictions (separate output heads)
        output_ana = self.output_ana(last_output)            # (batch, 3) [DO-ANA, AN-ANA, NN-ANA]
        output_ano = self.output_ano(last_output)            # (batch, 3) [DO-ANO, AN-ANO, NN-ANO]
        output_ox = self.output_ox(last_output)              # (batch, 3)
        output_eff = self.output_eff(last_output)            # (batch, 3)
        
        # Take the fused vector of the last time step
        fusion_ana_ano_last = fusion_ana_ano[:, -1, :]       # (batch, 6)
        fusion_ox_last = fusion_ox[:, -1, :]                  # (batch, 3)
        
        return {
            "ANA": output_ana,        # split out as a separate output
            "ANO": output_ano,        # split out as a separate output
            "OX": output_ox,
            "EFF": output_eff,
            "fusion_ana_ano": fusion_ana_ano_last,
            "fusion_ox": fusion_ox_last
        }


def calculate_loss(outputs, labels):
    """Loss function (separate ANA and ANO outputs)"""
    mse_loss = nn.MSELoss()
    loss_ana = mse_loss(outputs["ANA"], labels["ANA"])
    loss_ano = mse_loss(outputs["ANO"], labels["ANO"])
    loss_ox = mse_loss(outputs["OX"], labels["OX"])
    loss_eff = mse_loss(outputs["EFF"], labels["EFF"])
    total_loss = loss_ana + loss_ano + loss_ox + loss_eff
    return {
        "total": total_loss,
        "ANA": loss_ana,
        "ANO": loss_ano,
        "OX": loss_ox,
        "EFF": loss_eff
    }


class WaterQualityDataset(torch.utils.data.Dataset):
    def __init__(self, wq_data, gene_dynamic, month_col_map, gene_dynamic_ox, month_col_map_ox,
                 gene_c_dynamic, month_col_map_c, gene_c_dynamic_ox, month_col_map_c_ox,
                 labels_dict, window_size=40):
        self.wq_data = wq_data
        self.gene_dynamic = gene_dynamic          # DataFrame: index=KEGG, columns=months (n_gene ana_ano sheet)
        self.month_col_map = month_col_map        # dict: {original column name: month number}
        self.gene_dynamic_ox = gene_dynamic_ox    # DataFrame: index=KEGG, columns=months (n_gene ox sheet)
        self.month_col_map_ox = month_col_map_ox  # dict: {original column name: month number}
        self.gene_c_dynamic = gene_c_dynamic          # DataFrame: index=KEGG, columns=months (c_gene ana_ano sheet)
        self.month_col_map_c = month_col_map_c        # dict: {original column name: month number}
        self.gene_c_dynamic_ox = gene_c_dynamic_ox    # DataFrame: index=KEGG, columns=months (c_gene ox sheet)
        self.month_col_map_c_ox = month_col_map_c_ox  # dict: {original column name: month number}
        self.labels_dict = labels_dict
        self.window_size = window_size
        
        # Input variable definition (ANA+ANO merged)
        self.inf_vars = ['F-INF_MA_smooth', 'T-INF_MA_smooth', 'pH-INF_MA_smooth', 
                        'COD-INF_MA_smooth', 'C/N-INF_MA_smooth', 'TN-INF_MA_smooth', 'AN-INF_MA_smooth']
        # Merge: ANA(3) + ANO(3) = 6 variables
        self.ana_ano_vars = [
            'DO-ANA_MA_smooth', 'AN-ANA_MA_smooth', 'NN-ANA_MA_smooth',  # the 3 ANA ones
            'DO-ANO_MA_smooth', 'AN-ANO_MA_smooth', 'NN-ANO_MA_smooth'   # the 3 ANO ones
        ]
        self.ox_vars = ['DO-OX_MA_smooth', 'AN-OX_MA_smooth', 'NN-OX_MA_smooth']
        self.eff_vars = ['COD-EFF_MA_smooth', 'TN-EFF_MA_smooth', 'AN-EFF_MA_smooth']
        
        # All KEGG IDs
        self.gene_all_kegg_ids = list(gene_dynamic.index)
        self.gene_ox_kegg_ids = list(gene_dynamic_ox.index)
        self.gene_c_all_kegg_ids = list(gene_c_dynamic.index)
        self.gene_c_ox_kegg_ids = list(gene_c_dynamic_ox.index)
        
        # Pre-compute the month mapping
        self.month_to_col = {m: c for c, m in month_col_map.items()}
        self.month_to_col_ox = {m: c for c, m in month_col_map_ox.items()}
        self.month_to_col_c = {m: c for c, m in month_col_map_c.items()}
        self.month_to_col_c_ox = {m: c for c, m in month_col_map_c_ox.items()}
    
    def __len__(self):
        return len(self.wq_data) - self.window_size
    
    def _get_gene_for_time(self, timestamp):
        """
        fetch the gene data of the matching month from the timestamp (merged version)
        
        Returns:
            gene_n_ana_ano: (21,) tensor - n_gene ana_ano (the same data used for ana+ano)
            gene_n_ox:  (21,) tensor - n_gene ox
            gene_c_ana_ano: (1205,) tensor - c_gene ana_ano (the same data used for ana+ano)
            gene_c_ox:  (1205,) tensor - c_gene ox
        """
        month = timestamp.month
        
        # ---- n_gene ana_ano expression data (from the ana_ano sheet) ----
        available_months = sorted(self.month_to_col.keys())
        if month in self.month_to_col:
            target_month = month
        else:
            candidates = [m for m in available_months if m <= month]
            target_month = candidates[-1] if candidates else available_months[0]
        
        col_name = self.month_to_col[target_month]
        
        # n_gene ana_ano uses the same data for all KEGG ids (NaN filled with 0)
        gene_n_vals = self.gene_dynamic[col_name].loc[self.gene_all_kegg_ids].fillna(0.0).values.astype(np.float32)
        gene_n_ana_ano = torch.tensor(gene_n_vals, dtype=torch.float32)  # (21,) 
        
        # ---- n_gene ox expression data (from the ox sheet) ----
        available_months_ox = sorted(self.month_to_col_ox.keys())
        if month in self.month_to_col_ox:
            target_month_ox = month
        else:
            candidates_ox = [m for m in available_months_ox if m <= month]
            target_month_ox = candidates_ox[-1] if candidates_ox else available_months_ox[0]
        
        col_name_ox = self.month_to_col_ox[target_month_ox]
        
        gene_n_ox_vals = self.gene_dynamic_ox[col_name_ox].loc[self.gene_ox_kegg_ids].fillna(0.0).values.astype(np.float32)
        gene_n_ox = torch.tensor(gene_n_ox_vals, dtype=torch.float32)  # (21,)
        
        # ---- c_gene ana_ano expression data (from the c_gene ana_ano sheet) ----
        available_months_c = sorted(self.month_to_col_c.keys())
        if month in self.month_to_col_c:
            target_month_c = month
        else:
            candidates_c = [m for m in available_months_c if m <= month]
            target_month_c = candidates_c[-1] if candidates_c else available_months_c[0]
        
        col_name_c = self.month_to_col_c[target_month_c]
        
        gene_c_ana_ano_vals = self.gene_c_dynamic[col_name_c].loc[self.gene_c_all_kegg_ids].fillna(0.0).values.astype(np.float32)
        gene_c_ana_ano = torch.tensor(gene_c_ana_ano_vals, dtype=torch.float32)  # (1205,) 
        
        # ---- c_gene ox expression data (from the c_gene ox sheet) ----
        available_months_c_ox = sorted(self.month_to_col_c_ox.keys())
        if month in self.month_to_col_c_ox:
            target_month_c_ox = month
        else:
            candidates_c_ox = [m for m in available_months_c_ox if m <= month]
            target_month_c_ox = candidates_c_ox[-1] if candidates_c_ox else available_months_c_ox[0]
        
        col_name_c_ox = self.month_to_col_c_ox[target_month_c_ox]
        
        gene_c_ox_vals = self.gene_c_dynamic_ox[col_name_c_ox].loc[self.gene_c_ox_kegg_ids].fillna(0.0).values.astype(np.float32)
        gene_c_ox = torch.tensor(gene_c_ox_vals, dtype=torch.float32)  # (1205,)
        
        return gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox
    
    def __getitem__(self, idx):
        # Take the first 40 time steps
        window_end = idx + self.window_size
        
        # Fetch the INF data (keeping the time dimension)
        inf_data = self.wq_data[self.inf_vars].iloc[idx:window_end]
        inf_feat = torch.tensor(inf_data.values, dtype=torch.float32)  # (window_size, 7)
        
        # Fetch the merged ANA+ANO data
        ana_ano_data = self.wq_data[self.ana_ano_vars].iloc[idx:window_end]
        ana_ano_feat = torch.tensor(ana_ano_data.values, dtype=torch.float32)  # (window_size, 6) 
        
        # Fetch the OX data
        ox_data = self.wq_data[self.ox_vars].iloc[idx:window_end]
        ox_feat = torch.tensor(ox_data.values, dtype=torch.float32)  # (window_size, 3)
        
        # Fetch the EFF data
        eff_data = self.wq_data[self.eff_vars].iloc[idx:window_end]
        eff_feat = torch.tensor(eff_data.values, dtype=torch.float32)  # (window_size, 3)
        
        # Fetch the gene features of the timestamp matching window_end (returns 4 tensors)
        gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox = self._get_gene_for_time(
            self.wq_data['time'].iloc[window_end])
        
        # Get the labels of the next time step (separate ANA and ANO)
        labels = {
            "ANA": torch.tensor(self.labels_dict["ANA"][window_end], dtype=torch.float32),      # (3,)
            "ANO": torch.tensor(self.labels_dict["ANO"][window_end], dtype=torch.float32),       # (3,)
            "OX": torch.tensor(self.labels_dict["OX"][window_end], dtype=torch.float32),
            "EFF": torch.tensor(self.labels_dict["EFF"][window_end], dtype=torch.float32)
        }
        
        # Return value changed: 11 elements (was 13)
        return inf_feat, ana_ano_feat, ox_feat, eff_feat, \
               gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, labels


def prepare_data():
    """Prepare data (adapted to the merged architecture)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the absolute path of the data file
    wq_path = os.path.join(current_dir, '../data_WQ/wq_scaled.csv')
    wq_raw_path = os.path.join(current_dir, '../data_WQ/wq.xlsx')
    gene_n_path = os.path.join(current_dir, '../data_Gene/n_gene_dynamic_common.xlsx')
    gene_c_path = os.path.join(current_dir, '../data_Gene/c_gene_dynamic_common.xlsx')
    
    print(f"Loading data from:")
    print(f"  - Water quality data: {wq_path}")
    print(f"  - Raw WQ data (for timestamps): {wq_raw_path}")
    print(f"  - n_gene data: {gene_n_path}")
    print(f"  - c_gene data: {gene_c_path}")
    
    try:
        # Read the water quality data
        wq_data = pd.read_csv(wq_path)

        # Read the raw wq data to get the time column
        wq_raw = pd.read_excel(wq_raw_path)
        
        # Keep the row count consistent: follow wq_data and truncate wq_raw to the same length
        if len(wq_raw) != len(wq_data):
            print(f"  Warning: wq.xlsx ({len(wq_raw)} rows) and wq_scaled.csv ({len(wq_data)} rows) have different lengths!")
            print(f"  Aligning to wq_scaled.csv length ({len(wq_data)} rows)...")
            wq_raw = wq_raw.iloc[:len(wq_data)].reset_index(drop=True)
        
        wq_data['time'] = wq_raw['time']
        
        # ---- Read n_gene data ----
        # ana_ano sheet (21 rows x 6 cols: 5 months + KEGG description)
        gene_dynamic = pd.read_excel(gene_n_path, sheet_name='ana_ano')
        gene_dynamic['KEGG'] = gene_dynamic.iloc[:, -1].str.extract(r'(K\d+)')[0]
        gene_dynamic = gene_dynamic.drop(columns=gene_dynamic.columns[-2])
        gene_dynamic = gene_dynamic.set_index('KEGG')
        
        # ox sheet (21 rows x 6 cols: 5 months + KEGG description)
        gene_dynamic_ox = pd.read_excel(gene_n_path, sheet_name='ox')
        gene_dynamic_ox['KEGG'] = gene_dynamic_ox.iloc[:, -1].str.extract(r'(K\d+)')[0]
        gene_dynamic_ox = gene_dynamic_ox.drop(columns=gene_dynamic_ox.columns[-2])
        gene_dynamic_ox = gene_dynamic_ox.set_index('KEGG')
        
        # ---- Read c_gene data ----
        # c_gene ana_ano sheet (1205 rows x 6 cols)
        gene_c_dynamic = pd.read_excel(gene_c_path, sheet_name='ana_ano')
        gene_c_dynamic['KEGG'] = gene_c_dynamic.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
        gene_c_dynamic = gene_c_dynamic.drop(columns=gene_c_dynamic.columns[-2])
        gene_c_dynamic = gene_c_dynamic.dropna(subset=['KEGG'])
        gene_c_dynamic = gene_c_dynamic.set_index('KEGG')
        
        # c_gene ox sheet (1205 rows x 6 cols)
        gene_c_dynamic_ox = pd.read_excel(gene_c_path, sheet_name='ox')
        gene_c_dynamic_ox['KEGG'] = gene_c_dynamic_ox.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
        gene_c_dynamic_ox = gene_c_dynamic_ox.drop(columns=gene_c_dynamic_ox.columns[-2])
        gene_c_dynamic_ox = gene_c_dynamic_ox.dropna(subset=['KEGG'])
        gene_c_dynamic_ox = gene_c_dynamic_ox.set_index('KEGG')
        
        def _build_month_col_map(df):
            """Build the month mapping from the gene DataFrame column names"""
            month_abbr_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            month_col_map = {}
            data_cols = df.columns[:5]
            for col in data_cols:
                col_lower = str(col).lower()
                matched = False
                for abbr, month_num in month_abbr_map.items():
                    if abbr in col_lower:
                        month_col_map[col] = month_num
                        matched = True
                        break
                if not matched:
                    print(f"  Warning: could not determine month for column '{col}'")
            return month_col_map
        
        month_col_map = _build_month_col_map(gene_dynamic)
        month_col_map_ox = _build_month_col_map(gene_dynamic_ox)
        month_col_map_c = _build_month_col_map(gene_c_dynamic)
        month_col_map_c_ox = _build_month_col_map(gene_c_dynamic_ox)
        
        print(f"n_gene (ana_ano) months: {month_col_map}, KEGG count: {len(gene_dynamic)}")
        print(f"n_gene (ox) months: {month_col_map_ox}, KEGG count: {len(gene_dynamic_ox)}")
        print(f"c_gene (ana_ano) months: {month_col_map_c}, KEGG count: {len(gene_c_dynamic)}")
        print(f"c_gene (ox) months: {month_col_map_c_ox}, KEGG count: {len(gene_c_dynamic_ox)}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None, None
    
    print(f"Total samples: {len(wq_data)}")
    print(f"Time range: {wq_data['time'].min()} -> {wq_data['time'].max()}")
    
    # Prepare the label data (separate ANA and ANO, 3 dims each)
    labels_dict = {
        "ANA": wq_data[['DO-ANA_MA_smooth', 'AN-ANA_MA_smooth', 'NN-ANA_MA_smooth']].values,  # (N, 3) 
        "ANO": wq_data[['DO-ANO_MA_smooth', 'AN-ANO_MA_smooth', 'NN-ANO_MA_smooth']].values,   # (N, 3) 
        "OX": wq_data[['DO-OX_MA_smooth', 'AN-OX_MA_smooth', 'NN-OX_MA_smooth']].values,
        "EFF": wq_data[['COD-EFF_MA_smooth', 'TN-EFF_MA_smooth', 'AN-EFF_MA_smooth']].values
    }
    
    return (wq_data, gene_dynamic, month_col_map, gene_dynamic_ox, month_col_map_ox,
            gene_c_dynamic, month_col_map_c, gene_c_dynamic_ox, month_col_map_c_ox, labels_dict)
