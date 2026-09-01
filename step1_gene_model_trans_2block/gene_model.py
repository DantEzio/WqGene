import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

class GeneModel(nn.Module):
    def __init__(self, hidden_dim=256, dropout_rate=0.3,
                 # 合并后的基因维度: ana+ano共用同一套基因数据
                 gene_n_ana_ano_dim=21,    # n_gene ana_ano (21 KEGG IDs)
                 gene_n_ox_dim=21,          # n_gene ox (21 KEGG IDs)
                 gene_c_ana_ano_dim=1205,   # c_gene ana_ano (1205 KEGG IDs)
                 gene_c_ox_dim=1205):       # c_gene ox (1205 KEGG IDs)
        super().__init__()
        # ===== 输入维度定义 (合并ANA+ANO) =====
        self.wq_inf_dim = 7  # F-INF, T-INF, pH-INF, COD-INF, C/N-INF, TN-INF, AN-INF
        self.wq_ana_ano_dim = 6  # ★ 合并: DO/AN/NN-ANA + DO/AN/NN-ANO = 6
        self.wq_ox_dim = 3   # DO-OX, AN-OX, NN-OX
        self.wq_eff_dim = 3   # COD-EFF, TN-EFF, AN-EFF
        
        # 基因维度
        self.gene_n_ana_ano_dim = gene_n_ana_ano_dim   # n_gene (ana_ano sheet)
        self.gene_n_ox_dim = gene_n_ox_dim              # n_gene (ox sheet)
        self.gene_c_ana_ano_dim = gene_c_ana_ano_dim   # c_gene (ana_ano sheet)
        self.gene_c_ox_dim = gene_c_ox_dim              # c_gene (ox sheet)
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # ===== 输入投影层 =====
        self.inf_proj = nn.Sequential(
            nn.Linear(self.wq_inf_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ★ 合并: ANA+ANO 投影层 (输入6维)
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
        
        # ===== n_gene 特征投影层 (合并ANA+ANO) =====
        # ★ 原来的 gene_n_proj_ana + gene_n_proj_ano → 合并为一个
        self.gene_n_proj_ana_ano = nn.Sequential(
            nn.Linear(self.gene_n_ana_ano_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.wq_ana_ano_dim),  # 输出6维
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # OX的n_gene投影保持不变
        self.gene_n_proj_ox = nn.Sequential(
            nn.Linear(self.gene_n_ox_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.wq_ox_dim),  # 输出3维
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ===== c_gene 特征投影层 (合并ANA+ANO) =====
        c_gene_hidden = hidden_dim * 2  # 512
        
        # ★ 原来的 gene_c_proj_ana + gene_c_proj_ano → 合并为一个
        self.gene_c_proj_ana_ano = nn.Sequential(
            nn.Linear(self.gene_c_ana_ano_dim, c_gene_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(c_gene_hidden, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.wq_ana_ano_dim),  # 输出6维
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # OX的c_gene投影保持不变
        self.gene_c_proj_ox = nn.Sequential(
            nn.Linear(self.gene_c_ox_dim, c_gene_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(c_gene_hidden, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.wq_ox_dim),  # 输出3维
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ===== 基因融合层 (合并ANA+ANO) =====
        # ★ 原 fusion_ana + fusion_ano → 合并为一个: (6+6)→6
        self.gene_fusion_ana_ano = nn.Linear(self.wq_ana_ano_dim * 2, self.wq_ana_ano_dim)
        # OX融合保持不变: (3+3)→3
        self.gene_fusion_ox = nn.Linear(self.wq_ox_dim * 2, self.wq_ox_dim)
        
        # ===== LSTM framework =====
        # ★ 输入维度调整: inf(7) + ana_ano(6) + ox(3) + eff(3) = 19 (不变!)
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
        
        # ===== 输出层 (独立ANA和ANO输出头) =====
        # ★ 拆分回 output_ana(3) + output_ano(3)，与原模型格式一致
        self.output_ana = nn.Linear(self.hidden_dim, 3)     # [DO-ANA, AN-ANA, NN-ANA]
        self.output_ano = nn.Linear(self.hidden_dim, 3)     # [DO-ANO, AN-ANO, NN-ANO]
        self.output_ox = nn.Linear(self.hidden_dim, 3)      # 不变
        self.output_eff = nn.Linear(self.hidden_dim, 3)     # 不变
    
    def forward(self, wq_inf, wq_ana_ano, wq_ox, wq_eff,
                gene_n_ana_ano, gene_n_ox,
                gene_c_ana_ano, gene_c_ox):
        """
        参数变化:
        - 删除: wq_ana, wq_ano, gene_n_ana, gene_n_ano, gene_c_ana, gene_c_ano
        - 新增: wq_ana_ano, gene_n_ana_ano, gene_c_ana_ano
        """
        # 输入形状: (batch, timestep, dim)
        batch_size, timestep, _ = wq_inf.shape
        
        # 输入投影（对每个时间步应用相同的投影）
        inf_proj = wq_inf  # (batch, timestep, wq_inf_dim)
        ana_ano_proj = wq_ana_ano  # (batch, timestep, wq_ana_ano_dim=6) ★
        ox_proj = wq_ox    # (batch, timestep, wq_ox_dim)
        eff_proj = wq_eff  # (batch, timestep, wq_eff_dim)
        
        # ---- n_gene 特征投影 (合并) ----
        gene_n_ana_ano_proj = self.gene_n_proj_ana_ano(gene_n_ana_ano)  # (batch, 6) ★
        gene_n_ox_proj = self.gene_n_proj_ox(gene_n_ox)               # (batch, 3)
        
        # ---- c_gene 特征投影 (合并) ----
        gene_c_ana_ano_proj = self.gene_c_proj_ana_ano(gene_c_ana_ano)  # (batch, 6) ★
        gene_c_ox_proj = self.gene_c_proj_ox(gene_c_ox)                # (batch, 3)
        
        # ---- n_gene 和 c_gene 拼接后融合 (合并) ----
        # concat n(6) + c(6) = 12, then project to 6
        gene_ana_ano_fused = self.gene_fusion_ana_ano(
            torch.cat([gene_n_ana_ano_proj, gene_c_ana_ano_proj], dim=-1))  # (batch, 6) ★
        
        # OX融合: concat n(3) + c(3) = 6, then project to 3
        gene_ox_fused = self.gene_fusion_ox(
            torch.cat([gene_n_ox_proj, gene_c_ox_proj], dim=-1))  # (batch, 3)
        
        # 扩展基因特征到时间维度
        gene_ana_ano_fused = gene_ana_ano_fused.unsqueeze(1).repeat(1, timestep, 1)  # (batch, timestep, 6) ★
        gene_ox_fused = gene_ox_fused.unsqueeze(1).repeat(1, timestep, 1)             # (batch, timestep, 3)
        
        # 分量相加融合
        fusion_ana_ano = ana_ano_proj + gene_ana_ano_fused  # (batch, timestep, 6) ★
        fusion_ox = ox_proj + gene_ox_fused                  # (batch, timestep, 3)
        
        # 拼接成大变量: (batch, timestep, 7+6+3+3=19)
        concat_features = torch.cat([
            inf_proj,           # wq_inf_dim=7
            fusion_ana_ano,     # wq_ana_ano_dim=6 ★
            fusion_ox,          # wq_ox_dim=3
            eff_proj            # wq_eff_dim=3
        ], dim=-1)  # total = 19
        
        # 送入LSTM
        lstm_out, _ = self.lstm(concat_features)  # (batch, timestep, hidden_dim)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 输出预测 (独立输出头)
        output_ana = self.output_ana(last_output)            # (batch, 3) [DO-ANA, AN-ANA, NN-ANA]
        output_ano = self.output_ano(last_output)            # (batch, 3) [DO-ANO, AN-ANO, NN-ANO]
        output_ox = self.output_ox(last_output)              # (batch, 3)
        output_eff = self.output_eff(last_output)            # (batch, 3)
        
        # 取最后一个时间步的融合向量
        fusion_ana_ano_last = fusion_ana_ano[:, -1, :]       # (batch, 6)
        fusion_ox_last = fusion_ox[:, -1, :]                  # (batch, 3)
        
        return {
            "ANA": output_ana,        # ★ 拆分为独立输出
            "ANO": output_ano,        # ★ 拆分为独立输出
            "OX": output_ox,
            "EFF": output_eff,
            "fusion_ana_ano": fusion_ana_ano_last,
            "fusion_ox": fusion_ox_last
        }


def calculate_loss(outputs, labels):
    """Loss函数 (独立ANA和ANO输出)"""
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
        self.gene_dynamic = gene_dynamic          # DataFrame: index=KEGG, columns=月份 (n_gene ana_ano sheet)
        self.month_col_map = month_col_map        # dict: {原始列名: 月份数字}
        self.gene_dynamic_ox = gene_dynamic_ox    # DataFrame: index=KEGG, columns=月份 (n_gene ox sheet)
        self.month_col_map_ox = month_col_map_ox  # dict: {原始列名: 月份数字}
        self.gene_c_dynamic = gene_c_dynamic          # DataFrame: index=KEGG, columns=月份 (c_gene ana_ano sheet)
        self.month_col_map_c = month_col_map_c        # dict: {原始列名: 月份数字}
        self.gene_c_dynamic_ox = gene_c_dynamic_ox    # DataFrame: index=KEGG, columns=月份 (c_gene ox sheet)
        self.month_col_map_c_ox = month_col_map_c_ox  # dict: {原始列名: 月份数字}
        self.labels_dict = labels_dict
        self.window_size = window_size
        
        # ★ 输入变量定义 (合并ANA+ANO)
        self.inf_vars = ['F-INF_MA_smooth', 'T-INF_MA_smooth', 'pH-INF_MA_smooth', 
                        'COD-INF_MA_smooth', 'C/N-INF_MA_smooth', 'TN-INF_MA_smooth', 'AN-INF_MA_smooth']
        # ★ 合并: ANA(3) + ANO(3) = 6个变量
        self.ana_ano_vars = [
            'DO-ANA_MA_smooth', 'AN-ANA_MA_smooth', 'NN-ANA_MA_smooth',  # ANA的3个
            'DO-ANO_MA_smooth', 'AN-ANO_MA_smooth', 'NN-ANO_MA_smooth'   # ANO的3个
        ]
        self.ox_vars = ['DO-OX_MA_smooth', 'AN-OX_MA_smooth', 'NN-OX_MA_smooth']
        self.eff_vars = ['COD-EFF_MA_smooth', 'TN-EFF_MA_smooth', 'AN-EFF_MA_smooth']
        
        # 所有KEGG ID
        self.gene_all_kegg_ids = list(gene_dynamic.index)
        self.gene_ox_kegg_ids = list(gene_dynamic_ox.index)
        self.gene_c_all_kegg_ids = list(gene_c_dynamic.index)
        self.gene_c_ox_kegg_ids = list(gene_c_dynamic_ox.index)
        
        # 预计算月份映射
        self.month_to_col = {m: c for c, m in month_col_map.items()}
        self.month_to_col_ox = {m: c for c, m in month_col_map_ox.items()}
        self.month_to_col_c = {m: c for c, m in month_col_map_c.items()}
        self.month_to_col_c_ox = {m: c for c, m in month_col_map_c_ox.items()}
    
    def __len__(self):
        return len(self.wq_data) - self.window_size
    
    def _get_gene_for_time(self, timestamp):
        """
        根据时间戳获取对应月份的基因数据 (合并后版本)
        
        Returns:
            gene_n_ana_ano: (21,) tensor - n_gene ana_ano (同一套用于ana+ano)
            gene_n_ox:  (21,) tensor - n_gene ox
            gene_c_ana_ano: (1205,) tensor - c_gene ana_ano (同一套用于ana+ano)
            gene_c_ox:  (1205,) tensor - c_gene ox
        """
        month = timestamp.month
        
        # ---- n_gene ana_ano 基因数据 (来自 ana_ano sheet) ----
        available_months = sorted(self.month_to_col.keys())
        if month in self.month_to_col:
            target_month = month
        else:
            candidates = [m for m in available_months if m <= month]
            target_month = candidates[-1] if candidates else available_months[0]
        
        col_name = self.month_to_col[target_month]
        
        # n_gene ana_ano 使用全部KEGG的同一组数据 (NaN填充为0)
        gene_n_vals = self.gene_dynamic[col_name].loc[self.gene_all_kegg_ids].fillna(0.0).values.astype(np.float32)
        gene_n_ana_ano = torch.tensor(gene_n_vals, dtype=torch.float32)  # (21,) ★
        
        # ---- n_gene ox 基因数据 (来自 ox sheet) ----
        available_months_ox = sorted(self.month_to_col_ox.keys())
        if month in self.month_to_col_ox:
            target_month_ox = month
        else:
            candidates_ox = [m for m in available_months_ox if m <= month]
            target_month_ox = candidates_ox[-1] if candidates_ox else available_months_ox[0]
        
        col_name_ox = self.month_to_col_ox[target_month_ox]
        
        gene_n_ox_vals = self.gene_dynamic_ox[col_name_ox].loc[self.gene_ox_kegg_ids].fillna(0.0).values.astype(np.float32)
        gene_n_ox = torch.tensor(gene_n_ox_vals, dtype=torch.float32)  # (21,)
        
        # ---- c_gene ana_ano 基因数据 (来自 c_gene ana_ano sheet) ----
        available_months_c = sorted(self.month_to_col_c.keys())
        if month in self.month_to_col_c:
            target_month_c = month
        else:
            candidates_c = [m for m in available_months_c if m <= month]
            target_month_c = candidates_c[-1] if candidates_c else available_months_c[0]
        
        col_name_c = self.month_to_col_c[target_month_c]
        
        gene_c_ana_ano_vals = self.gene_c_dynamic[col_name_c].loc[self.gene_c_all_kegg_ids].fillna(0.0).values.astype(np.float32)
        gene_c_ana_ano = torch.tensor(gene_c_ana_ano_vals, dtype=torch.float32)  # (1205,) ★
        
        # ---- c_gene ox 基因数据 (来自 c_gene ox sheet) ----
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
        # 获取前40个时间步的数据
        window_end = idx + self.window_size
        
        # 获取INF数据（保持时间序列维度）
        inf_data = self.wq_data[self.inf_vars].iloc[idx:window_end]
        inf_feat = torch.tensor(inf_data.values, dtype=torch.float32)  # (window_size, 7)
        
        # ★ 获取合并后的 ANA+ANO 数据
        ana_ano_data = self.wq_data[self.ana_ano_vars].iloc[idx:window_end]
        ana_ano_feat = torch.tensor(ana_ano_data.values, dtype=torch.float32)  # (window_size, 6) ★
        
        # 获取OX数据
        ox_data = self.wq_data[self.ox_vars].iloc[idx:window_end]
        ox_feat = torch.tensor(ox_data.values, dtype=torch.float32)  # (window_size, 3)
        
        # 获取EFF数据
        eff_data = self.wq_data[self.eff_vars].iloc[idx:window_end]
        eff_feat = torch.tensor(eff_data.values, dtype=torch.float32)  # (window_size, 3)
        
        # 根据 window_end 对应的时间戳获取基因特征 (★ 返回4个tensor)
        gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox = self._get_gene_for_time(
            self.wq_data['time'].iloc[window_end])
        
        # 获取下一时刻的标签 (★ 独立ANA和ANO)
        labels = {
            "ANA": torch.tensor(self.labels_dict["ANA"][window_end], dtype=torch.float32),      # ★ (3,)
            "ANO": torch.tensor(self.labels_dict["ANO"][window_end], dtype=torch.float32),       # ★ (3,)
            "OX": torch.tensor(self.labels_dict["OX"][window_end], dtype=torch.float32),
            "EFF": torch.tensor(self.labels_dict["EFF"][window_end], dtype=torch.float32)
        }
        
        # ★ 返回值变化: 11个元素 (原13个)
        return inf_feat, ana_ano_feat, ox_feat, eff_feat, \
               gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, labels


def prepare_data():
    """准备数据 (适配合并后的架构)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建数据文件的绝对路径
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
        # 读取水质数据
        wq_data = pd.read_csv(wq_path)

        # 读取原始 wq 数据获取时间列
        wq_raw = pd.read_excel(wq_raw_path)
        
        # ★ 确保行数一致：以wq_data为准，截取wq_raw对应行数
        if len(wq_raw) != len(wq_data):
            print(f"  Warning: wq.xlsx ({len(wq_raw)} rows) and wq_scaled.csv ({len(wq_data)} rows) have different lengths!")
            print(f"  Aligning to wq_scaled.csv length ({len(wq_data)} rows)...")
            wq_raw = wq_raw.iloc[:len(wq_data)].reset_index(drop=True)
        
        wq_data['time'] = wq_raw['time']
        
        # ---- 读取 n_gene 数据 ----
        # ana_ano sheet (21行 x 6列: 5个月份 + KEGG描述)
        gene_dynamic = pd.read_excel(gene_n_path, sheet_name='ana_ano')
        gene_dynamic['KEGG'] = gene_dynamic.iloc[:, -1].str.extract(r'(K\d+)')[0]
        gene_dynamic = gene_dynamic.drop(columns=gene_dynamic.columns[-2])
        gene_dynamic = gene_dynamic.set_index('KEGG')
        
        # ox sheet (21行 x 6列: 5个月份 + KEGG描述)
        gene_dynamic_ox = pd.read_excel(gene_n_path, sheet_name='ox')
        gene_dynamic_ox['KEGG'] = gene_dynamic_ox.iloc[:, -1].str.extract(r'(K\d+)')[0]
        gene_dynamic_ox = gene_dynamic_ox.drop(columns=gene_dynamic_ox.columns[-2])
        gene_dynamic_ox = gene_dynamic_ox.set_index('KEGG')
        
        # ---- 读取 c_gene 数据 ----
        # c_gene ana_ano sheet (1205行 x 6列)
        gene_c_dynamic = pd.read_excel(gene_c_path, sheet_name='ana_ano')
        gene_c_dynamic['KEGG'] = gene_c_dynamic.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
        gene_c_dynamic = gene_c_dynamic.drop(columns=gene_c_dynamic.columns[-2])
        gene_c_dynamic = gene_c_dynamic.dropna(subset=['KEGG'])
        gene_c_dynamic = gene_c_dynamic.set_index('KEGG')
        
        # c_gene ox sheet (1205行 x 6列)
        gene_c_dynamic_ox = pd.read_excel(gene_c_path, sheet_name='ox')
        gene_c_dynamic_ox['KEGG'] = gene_c_dynamic_ox.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
        gene_c_dynamic_ox = gene_c_dynamic_ox.drop(columns=gene_c_dynamic_ox.columns[-2])
        gene_c_dynamic_ox = gene_c_dynamic_ox.dropna(subset=['KEGG'])
        gene_c_dynamic_ox = gene_c_dynamic_ox.set_index('KEGG')
        
        def _build_month_col_map(df):
            """从gene DataFrame列名构建月份映射"""
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
    
    # ★ 准备标签数据 (独立ANA和ANO，各3维)
    labels_dict = {
        "ANA": wq_data[['DO-ANA_MA_smooth', 'AN-ANA_MA_smooth', 'NN-ANA_MA_smooth']].values,  # (N, 3) ★
        "ANO": wq_data[['DO-ANO_MA_smooth', 'AN-ANO_MA_smooth', 'NN-ANO_MA_smooth']].values,   # (N, 3) ★
        "OX": wq_data[['DO-OX_MA_smooth', 'AN-OX_MA_smooth', 'NN-OX_MA_smooth']].values,
        "EFF": wq_data[['COD-EFF_MA_smooth', 'TN-EFF_MA_smooth', 'AN-EFF_MA_smooth']].values
    }
    
    return (wq_data, gene_dynamic, month_col_map, gene_dynamic_ox, month_col_map_ox,
            gene_c_dynamic, month_col_map_c, gene_c_dynamic_ox, month_col_map_c_ox, labels_dict)
