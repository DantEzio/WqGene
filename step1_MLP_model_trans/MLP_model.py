import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

class MLPModel(nn.Module):
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
        
        # Input projection layer (keeps the time dimension)
        self.inf_proj = nn.Sequential(
            nn.Linear(self.wq_inf_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.ana_proj = nn.Sequential(
            nn.Linear(self.wq_ana_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.ano_proj = nn.Sequential(
            nn.Linear(self.wq_ano_dim, self.hidden_dim),
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
        
        # MLP framework (replaces LSTM)
        # Compute the input dimension: timestep dimension * projection dimension
        self.timestep_dim = 40  # window size
        self.concat_dim = self.wq_inf_dim + self.wq_ana_dim + self.wq_ano_dim + self.wq_ox_dim + self.wq_eff_dim
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.concat_dim * self.timestep_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.output_ana = nn.Linear(self.hidden_dim, 3)  # DO-ANA, AN-ANA, NN-ANA
        self.output_ano = nn.Linear(self.hidden_dim, 3)  # DO-ANO, AN-ANO, NN-ANO
        self.output_ox = nn.Linear(self.hidden_dim, 3)   # DO-OX, AN-OX, NN-OX
        self.output_eff = nn.Linear(self.hidden_dim, 3)  # COD-EFF, TN-EFF, AN-EFF
    
    def forward(self, wq_inf, wq_ana, wq_ano, wq_ox, wq_eff):
        # Input shape: (batch, timestep, dim)
        batch_size, timestep, _ = wq_inf.shape
        
        # Input projection (the same projection is applied at every time step)
        inf_proj = wq_inf  # (batch, timestep, wq_inf_dim)
        ana_proj = wq_ana  # (batch, timestep, wq_ana_dim)
        ano_proj = wq_ano  # (batch, timestep, wq_ano_dim)
        ox_proj = wq_ox    # (batch, timestep, wq_ox_dim)
        eff_proj = wq_eff  # (batch, timestep, wq_eff_dim)
        
        # Concatenate into the big input tensor
        concat_features = torch.cat([
            inf_proj,             # wq_inf_dim
            ana_proj,             # wq_ana_dim
            ano_proj,             # wq_ano_dim
            ox_proj,              # wq_ox_dim
            eff_proj              # wq_eff_dim
        ], dim=-1)  # (batch, timestep, concat_dim)
        
        # Flatten the time dimension
        flattened = concat_features.reshape(batch_size, -1)  # (batch, timestep * concat_dim)
        
        # Feed into the MLP
        mlp_out = self.mlp(flattened)  # (batch, hidden_dim)
        
        # Output the predictions
        output_ana = self.output_ana(mlp_out)
        output_ano = self.output_ano(mlp_out)
        output_ox = self.output_ox(mlp_out)
        output_eff = self.output_eff(mlp_out)
        
        return {
            "ANA": output_ana,
            "ANO": output_ano,
            "OX": output_ox,
            "EFF": output_eff
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