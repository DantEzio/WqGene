"""
LSTM_model.py - LSTM baseline model definition and data handling
=========================================================
use it as the GeneModel reference baseline: WQ series only, without any gene information.

Main components:
    1) LSTMModel            : multi-output LSTM taking a window x feature WQ sequence as input,
                              produces 6 dims (ANA/ANO/OX x DO/AN/NN).
    2) calculate_loss       : multi-output MSE loss.
    3) WaterQualityDataset  : builds WQ sliding-window samples (window steps of history -> next-step label).
    4) prepare_data         : reads and standardizes the WQ data and splits it chronologically into train/val/test.

Input data:
    ../data_WQ/wq_scaled.csv, ../data_WQ/wq.xlsx

Usage:
    imported as a module by train_LSTM_model.py / test_LSTM_model*.py
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

class LSTMModel(nn.Module):
    def __init__(self, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        # Input dimension definition
        self.wq_inf_dim = 7  # F-INF, T-INF, pH-INF, COD-INF, C/N-INF, TN-INF, AN-INF
        self.wq_ana_dim = 3  # DO-ANA, AN-ANA, NN-ANA
        self.wq_ano_dim = 3  # DO-ANO, AN-ANO, NN-ANO
        self.wq_ox_dim = 3   # DO-OX, AN-OX, NN-OX
        self.wq_eff_dim = 3   # COD-EFF, TN-EFF, AN-EFF
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # LSTM framework
        self.lstm = nn.LSTM(
            input_size=self.wq_inf_dim + self.wq_ana_dim + self.wq_ano_dim + self.wq_ox_dim + self.wq_eff_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,  # Aligned with gene model
            bidirectional=False,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Output layer
        self.output_ana = nn.Linear(self.hidden_dim, 3)  # DO-ANA, AN-ANA, NN-ANA
        self.output_ano = nn.Linear(self.hidden_dim, 3)  # DO-ANO, AN-ANO, NN-ANO
        self.output_ox = nn.Linear(self.hidden_dim, 3)   # DO-OX, AN-OX, NN-OX
        self.output_eff = nn.Linear(self.hidden_dim, 3)  # COD-EFF, TN-EFF, AN-EFF
    
    def forward(self, wq_inf, wq_ana, wq_ano, wq_ox, wq_eff):
        # Input shape: (batch, timestep, dim)
        batch_size, timestep, _ = wq_inf.shape
        
        # Component assignment (consistent with the gene model forward pass)
        inf_proj = wq_inf  # (batch, timestep, wq_inf_dim)
        ana_proj = wq_ana  # (batch, timestep, wq_ana_dim)
        ano_proj = wq_ano  # (batch, timestep, wq_ano_dim)
        ox_proj = wq_ox    # (batch, timestep, wq_ox_dim)
        eff_proj = wq_eff  # (batch, timestep, wq_eff_dim)
        
        # Concatenate into the big input tensor (consistent with the gene model forward pass)
        concat_features = torch.cat([
            inf_proj,   # wq_inf_dim
            ana_proj,   # wq_ana_dim
            ano_proj,   # wq_ano_dim
            ox_proj,    # wq_ox_dim
            eff_proj    # wq_eff_dim
        ], dim=-1)  # (batch, timestep, wq_inf_dim + wq_ana_dim + wq_ano_dim + wq_ox_dim + wq_eff_dim)
        
        # Feed into the LSTM
        lstm_out, _ = self.lstm(concat_features)  # (batch, timestep, hidden_dim)
        
        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Output the predictions
        output_ana = self.output_ana(last_output)
        output_ano = self.output_ano(last_output)
        output_ox = self.output_ox(last_output)
        output_eff = self.output_eff(last_output)
        
        # Take the component vector of the last time step (matches the gene model fusion output)
        fusion_inf_ana_last = ana_proj[:, -1, :]  # (batch, wq_ana_dim)
        fusion_ana_ano_last = ano_proj[:, -1, :]  # (batch, wq_ano_dim)
        fusion_ano_ox_last = ox_proj[:, -1, :]    # (batch, wq_ox_dim)
        
        return {
            "ANA": output_ana,
            "ANO": output_ano,
            "OX": output_ox,
            "EFF": output_eff,
            "fusion_inf_ana": fusion_inf_ana_last,
            "fusion_ana_ano": fusion_ana_ano_last,
            "fusion_ano_ox": fusion_ano_ox_last
        }

def calculate_loss(outputs, labels):
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
    def __init__(self, wq_data, labels_dict, window_size=40):
        self.wq_data = wq_data
        self.labels_dict = labels_dict
        self.window_size = window_size
        
        # Input variable definition
        self.inf_vars = ['F-INF_MA_smooth', 'T-INF_MA_smooth', 'pH-INF_MA_smooth', 
                        'COD-INF_MA_smooth', 'C/N-INF_MA_smooth', 'TN-INF_MA_smooth', 'AN-INF_MA_smooth']
        self.ana_vars = ['DO-ANA_MA_smooth', 'AN-ANA_MA_smooth', 'NN-ANA_MA_smooth']
        self.ano_vars = ['DO-ANO_MA_smooth', 'AN-ANO_MA_smooth', 'NN-ANO_MA_smooth']
        self.ox_vars = ['DO-OX_MA_smooth', 'AN-OX_MA_smooth', 'NN-OX_MA_smooth']
        self.eff_vars = ['COD-EFF_MA_smooth', 'TN-EFF_MA_smooth', 'AN-EFF_MA_smooth']
    
    def __len__(self):
        return len(self.wq_data) - self.window_size
    
    def __getitem__(self, idx):
        # Take the first 40 time steps
        window_end = idx + self.window_size
        
        # Fetch the INF data (keeping the time dimension)
        inf_data = self.wq_data[self.inf_vars].iloc[idx:window_end]
        inf_feat = torch.tensor(inf_data.values, dtype=torch.float32)  # (window_size, 7)
        
        # Fetch the ANA data
        ana_data = self.wq_data[self.ana_vars].iloc[idx:window_end]
        ana_feat = torch.tensor(ana_data.values, dtype=torch.float32)  # (window_size, 3)
        
        # Fetch the ANO data
        ano_data = self.wq_data[self.ano_vars].iloc[idx:window_end]
        ano_feat = torch.tensor(ano_data.values, dtype=torch.float32)  # (window_size, 3)
        
        # Fetch the OX data
        ox_data = self.wq_data[self.ox_vars].iloc[idx:window_end]
        ox_feat = torch.tensor(ox_data.values, dtype=torch.float32)  # (window_size, 3)
        
        # Fetch the EFF data
        eff_data = self.wq_data[self.eff_vars].iloc[idx:window_end]
        eff_feat = torch.tensor(eff_data.values, dtype=torch.float32)  # (window_size, 3)
        
        # Get the labels of the next time step
        labels = {
            "ANA": torch.tensor(self.labels_dict["ANA"][window_end], dtype=torch.float32),
            "ANO": torch.tensor(self.labels_dict["ANO"][window_end], dtype=torch.float32),
            "OX": torch.tensor(self.labels_dict["OX"][window_end], dtype=torch.float32),
            "EFF": torch.tensor(self.labels_dict["EFF"][window_end], dtype=torch.float32)
        }
        
        return inf_feat, ana_feat, ano_feat, ox_feat, eff_feat, labels

def prepare_data():
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the absolute path of the data file
    wq_path = os.path.join(current_dir, '../data_WQ/wq_scaled.csv')
    
    print(f"Loading data from:")
    print(f"  - Water quality data: {wq_path}")
    
    try:
        # Read the water quality data
        wq_data = pd.read_csv(wq_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    print(f"Total samples: {len(wq_data)}")
    
    # Prepare the label data
    labels_dict = {
        "ANA": wq_data[['DO-ANA_MA_smooth', 'AN-ANA_MA_smooth', 'NN-ANA_MA_smooth']].values,
        "ANO": wq_data[['DO-ANO_MA_smooth', 'AN-ANO_MA_smooth', 'NN-ANO_MA_smooth']].values,
        "OX": wq_data[['DO-OX_MA_smooth', 'AN-OX_MA_smooth', 'NN-OX_MA_smooth']].values,
        "EFF": wq_data[['COD-EFF_MA_smooth', 'TN-EFF_MA_smooth', 'AN-EFF_MA_smooth']].values
    }
    
    return wq_data, labels_dict