"""
全模型Sobol敏感性分析（仅n_gene版本）
=============================================================
对完整GeneModel进行Sobol敏感性分析（而非仅gene embedding layer），
将n_gene(21维×3池)=63维基因特征作为Sobol输入，
c_gene固定为训练集均值背景，水质(WQ)特征固定为训练集均值背景。
输出按ana/ano/ox分别绘图，保存到results文件夹。
与 results3_Sharply.py (全模型SHAP) 对应，使用Sobol方法替代SHAP。
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import os
import sys
import gc
from collections import Counter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze import sobol

sys.path.append(os.path.join(os.path.dirname(__file__), 'step1_gene_model_trans'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'step1_gene_model_trans'))

from gene_model import GeneModel


# ============================================================
# 配置
# ============================================================
WINDOW_SIZE = 40          # 与模型一致的时间窗口
SOBOL_N = 1024            # Sobol基础采样数 (63变量, 每次运行总采样=1024*(2*63+2)=129024)
NUM_SOBOL_RUNS = 10       # 重复次数取平均
SOBOL_BATCH_SIZE = 512    # 模型前向传播mini-batch大小，防止GPU OOM
CALC_SECOND_ORDER = True  # 计算二阶交互效应
N_GENE_DIM = 21           # n_gene维度
C_GENE_DIM = 1205         # c_gene维度

# 基因输入总维度: 仅 n_gene(21)*3池 = 63
TOTAL_GENE_FEATURES = N_GENE_DIM * 3   # = 63

# 输出目标配置
TARGET_CONFIG = {
    'ana': {'label': 'ANA', 'output_key': 'ANA', 'title': 'Anaerobic Tank'},
    'ano': {'label': 'ANO', 'output_key': 'ANO', 'title': 'Anoxic Tank'},
    'ox':  {'label': 'OX',  'output_key': 'OX',  'title': 'Aerobic Tank'},
}

# 池子颜色配置 (按深浅蓝区分)
POOL_COLORS = {
    'ana': '#08306B',   # Anaerobic: 深蓝
    'ano': '#2171B5',   # Anoxic: 中蓝
    'ox':  '#6BAED6',   # Aerobic: 浅蓝
}


# ============================================================
# 模型加载 (复用 results3_Sharply.py)
# ============================================================
def load_trained_model():
    """加载训练好的GeneModel"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GeneModel(hidden_dim=64, dropout_rate=0.3).to(device)

    model_dir = os.path.join('step1_gene_model_trans', 'model', 'gene_model')
    if not os.path.exists(model_dir):
        model_dir = os.path.join('..', 'step1_gene_model_trans', 'model', 'gene_model')

    model_path = os.path.join(model_dir, 'best_gene_model.pth')
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, device
    else:
        print(f"Model not found: {model_path}")
        return None, device

    model.eval()
    return model, device


def load_wq_background_stats():
    """
    从训练集计算WQ特征的统计量，用于构建固定背景。
    返回各WQ分量的均值字典，供Sobol分析时固定使用。
    """
    wq_scaled_path = os.path.join('data_WQ', 'wq_scaled.csv')
    if not os.path.exists(wq_scaled_path):
        print(f"WQ data not found: {wq_scaled_path}, using zero background")
        return None

    df = pd.read_csv(wq_scaled_path)
    train_end = int(0.3 * len(df))
    train_df = df.iloc[:train_end]

    stats = {}
    inf_vars = ['F-INF_MA_smooth', 'T-INF_MA_smooth', 'pH-INF_MA_smooth',
                'COD-INF_MA_smooth', 'C/N-INF_MA_smooth', 'TN-INF_MA_smooth', 'AN-INF_MA_smooth']
    stats['inf'] = train_df[inf_vars].mean().values

    ana_vars = ['DO-ANA_MA_smooth', 'AN-ANA_MA_smooth', 'NN-ANA_MA_smooth']
    stats['ana'] = train_df[ana_vars].mean().values

    ano_vars = ['DO-ANO_MA_smooth', 'AN-ANO_MA_smooth', 'NN-ANO_MA_smooth']
    stats['ano'] = train_df[ano_vars].mean().values

    ox_vars = ['DO-OX_MA_smooth', 'AN-OX_MA_smooth', 'NN-OX_MA_smooth']
    stats['ox'] = train_df[ox_vars].mean().values

    eff_vars = ['COD-EFF_MA_smooth', 'TN-EFF_MA_smooth', 'AN-EFF_MA_smooth']
    stats['eff'] = train_df[eff_vars].mean().values

    print("WQ background statistics computed from training set")
    return stats


def load_cgene_background_stats():
    """
    从训练集计算c_gene特征的均值，用于构建固定背景。
    返回各c_gene分量的均值字典。
    """
    wq_scaled_path = os.path.join('data_WQ', 'wq_scaled.csv')
    wq_raw_path = os.path.join('data_WQ', 'wq.xlsx')
    if not os.path.exists(wq_scaled_path) or not os.path.exists(wq_raw_path):
        print("WQ data not found for c_gene background, using zero background")
        return None

    wq_data = pd.read_csv(wq_scaled_path)
    wq_raw = pd.read_excel(wq_raw_path)
    wq_data['time'] = wq_raw['time']

    n_samples = len(wq_data) - WINDOW_SIZE
    train_size = int(0.3 * n_samples)

    c_gene_path = os.path.join('data_Gene', 'c_gene_dynamic_common.xlsx')
    if not os.path.exists(c_gene_path):
        print(f"c_gene data not found: {c_gene_path}, using zero background")
        return None

    gene_c_dynamic = pd.read_excel(c_gene_path, sheet_name='ana_ano')
    gene_c_dynamic['KEGG'] = gene_c_dynamic.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
    gene_c_dynamic = gene_c_dynamic.drop(columns=gene_c_dynamic.columns[-2])
    gene_c_dynamic = gene_c_dynamic.dropna(subset=['KEGG'])
    gene_c_dynamic = gene_c_dynamic.set_index('KEGG')

    gene_c_dynamic_ox = pd.read_excel(c_gene_path, sheet_name='ox')
    gene_c_dynamic_ox['KEGG'] = gene_c_dynamic_ox.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
    gene_c_dynamic_ox = gene_c_dynamic_ox.drop(columns=gene_c_dynamic_ox.columns[-2])
    gene_c_dynamic_ox = gene_c_dynamic_ox.dropna(subset=['KEGG'])
    gene_c_dynamic_ox = gene_c_dynamic_ox.set_index('KEGG')

    month_abbr_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    def _build_month_col_map(df):
        mcm = {}
        for col in df.columns[:5]:
            col_lower = str(col).lower()
            for abbr, month_num in month_abbr_map.items():
                if abbr in col_lower:
                    mcm[col] = month_num
                    break
        return mcm

    month_col_map_c = _build_month_col_map(gene_c_dynamic)
    month_col_map_c_ox = _build_month_col_map(gene_c_dynamic_ox)
    month_to_col_c = {m: c for c, m in month_col_map_c.items()}
    month_to_col_c_ox = {m: c for c, m in month_col_map_c_ox.items()}

    c_kegg_ids = list(gene_c_dynamic.index)[:C_GENE_DIM]
    c_ox_kegg_ids = list(gene_c_dynamic_ox.index)[:C_GENE_DIM]
    available_months_c = sorted(month_to_col_c.keys())
    available_months_c_ox = sorted(month_to_col_c_ox.keys())

    month_counter_ana = Counter()
    month_counter_ox = Counter()

    for idx in range(train_size):
        timestamp = wq_data['time'].iloc[idx + WINDOW_SIZE]
        month = timestamp.month

        if month in month_to_col_c:
            target_month = month
        else:
            candidates = [m for m in available_months_c if m <= month]
            target_month = candidates[-1] if candidates else available_months_c[0]
        month_counter_ana[target_month] += 1

        if month in month_to_col_c_ox:
            target_month_ox = month
        else:
            candidates_ox = [m for m in available_months_c_ox if m <= month]
            target_month_ox = candidates_ox[-1] if candidates_ox else available_months_c_ox[0]
        month_counter_ox[target_month_ox] += 1

    c_ana_mean = np.zeros(C_GENE_DIM, dtype=np.float64)
    for m, count in month_counter_ana.items():
        col = month_to_col_c[m]
        vals = gene_c_dynamic[col].loc[c_kegg_ids].fillna(0.0).values.astype(np.float64)
        c_ana_mean += vals * count
    c_ana_mean /= train_size

    c_ox_mean = np.zeros(C_GENE_DIM, dtype=np.float64)
    for m, count in month_counter_ox.items():
        col = month_to_col_c_ox[m]
        vals = gene_c_dynamic_ox[col].loc[c_ox_kegg_ids].fillna(0.0).values.astype(np.float64)
        c_ox_mean += vals * count
    c_ox_mean /= train_size

    stats = {
        'ana': c_ana_mean.astype(np.float32),
        'ano': c_ana_mean.astype(np.float32),
        'ox':  c_ox_mean.astype(np.float32),
    }

    print(f"c_gene background statistics computed from {train_size} training samples")
    print(f"  ana/ano months: {dict(month_counter_ana)}")
    print(f"  ox months: {dict(month_counter_ox)}")
    return stats


# ============================================================
# 基因名称获取 (复用 results3_Sharply.py)
# ============================================================
def get_gene_names():
    """获取n_gene的KEGG ID列表，返回63维特征名"""
    n_gene_names = []

    n_gene_file = os.path.join('data_Gene', 'n_gene_dynamic_common.xlsx')
    if os.path.exists(n_gene_file):
        try:
            df = pd.read_excel(n_gene_file, sheet_name='ana_ano')
            kegg_ids = df.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
            kegg_ids = kegg_ids.dropna().tolist()
            n_gene_names = kegg_ids[:N_GENE_DIM]
        except Exception as e:
            print(f"Error reading n_gene names: {e}")
            n_gene_names = [f'n_gene_{i}' for i in range(N_GENE_DIM)]
    else:
        n_gene_names = [f'n_gene_{i}' for i in range(N_GENE_DIM)]

    print(f"Loaded {len(n_gene_names)} n_gene names")

    all_feature_names = []
    prefixes_ng = ['n_gene_ana_', 'n_gene_ano_', 'n_gene_ox_']
    for prefix in prefixes_ng:
        for name in n_gene_names:
            all_feature_names.append(prefix + name)

    return n_gene_names, all_feature_names


def load_ngene_data_ranges():
    """
    从n_gene数据文件加载实际数据范围，用于Sobol采样边界。
    n_gene数据未归一化，直接使用原始Excel值域作为Sobol边界。
    返回63个特征的(min, max)列表，排列: [ana(21), ano(21), ox(21)]。
    """
    n_gene_file = os.path.join('data_Gene', 'n_gene_dynamic_common.xlsx')
    if not os.path.exists(n_gene_file):
        print(f"n_gene data not found: {n_gene_file}, using default bounds (-1, 1)")
        return [(-1, 1) for _ in range(TOTAL_GENE_FEATURES)]

    # 加载 ana_ano sheet
    df_ana_ano = pd.read_excel(n_gene_file, sheet_name='ana_ano')
    df_ana_ano['KEGG'] = df_ana_ano.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
    df_ana_ano = df_ana_ano.drop(columns=df_ana_ano.columns[-2])
    df_ana_ano = df_ana_ano.dropna(subset=['KEGG'])
    df_ana_ano = df_ana_ano.set_index('KEGG')

    # 加载 ox sheet
    df_ox = pd.read_excel(n_gene_file, sheet_name='ox')
    df_ox['KEGG'] = df_ox.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
    df_ox = df_ox.drop(columns=df_ox.columns[-2])
    df_ox = df_ox.dropna(subset=['KEGG'])
    df_ox = df_ox.set_index('KEGG')

    # 取前21个KEGG
    kegg_ids_ana_ano = list(df_ana_ano.index)[:N_GENE_DIM]
    kegg_ids_ox = list(df_ox.index)[:N_GENE_DIM]

    # 获取数据列 (前5列是月份)
    data_cols_ana_ano = df_ana_ano.columns[:5]
    data_cols_ox = df_ox.columns[:5]

    # 收集所有月份的数据值 (21 KEGG x 5 months)
    vals_ana_ano = df_ana_ano[data_cols_ana_ano].loc[kegg_ids_ana_ano].fillna(0.0).values.astype(np.float64)
    vals_ox = df_ox[data_cols_ox].loc[kegg_ids_ox].fillna(0.0).values.astype(np.float64)

    # 为每个特征计算边界 (min, max)，稍微扩展以覆盖采样范围
    bounds = []

    # n_gene_ana (21个)
    for i in range(N_GENE_DIM):
        row_vals = vals_ana_ano[i, :]
        lo, hi = float(row_vals.min()), float(row_vals.max())
        if lo == hi:
            lo, hi = lo - 0.1, hi + 0.1
        margin = (hi - lo) * 0.1 if hi > lo else 0.1
        bounds.append((lo - margin, hi + margin))

    # n_gene_ano (21个, 与ana同源)
    for i in range(N_GENE_DIM):
        row_vals = vals_ana_ano[i, :]
        lo, hi = float(row_vals.min()), float(row_vals.max())
        if lo == hi:
            lo, hi = lo - 0.1, hi + 0.1
        margin = (hi - lo) * 0.1 if hi > lo else 0.1
        bounds.append((lo - margin, hi + margin))

    # n_gene_ox (21个)
    for i in range(N_GENE_DIM):
        row_vals = vals_ox[i, :]
        lo, hi = float(row_vals.min()), float(row_vals.max())
        if lo == hi:
            lo, hi = lo - 0.1, hi + 0.1
        margin = (hi - lo) * 0.1 if hi > lo else 0.1
        bounds.append((lo - margin, hi + margin))

    # 打印摘要
    all_lo = [b[0] for b in bounds]
    all_hi = [b[1] for b in bounds]
    print(f"n_gene data ranges computed from Excel (with 10% margin):")
    print(f"  Global min={min(all_lo):.4f}, Global max={max(all_hi):.4f}")
    print(f"  ana/ano example: [{bounds[0][0]:.4f}, {bounds[0][1]:.4f}]")
    print(f"  ox example:      [{bounds[42][0]:.4f}, {bounds[42][1]:.4f}]")

    return bounds


# ============================================================
# 全模型Sobol包装器 (参考 FullModelSHAPWrapper)
# ============================================================
class FullModelSobolWrapper:
    """
    将完整GeneModel包装为Sobol可调用的函数（仅n_gene版本）。
    c_gene固定为训练集均值背景，WQ固定为训练集均值背景。
    输入: (n_samples, 63) 仅含 n_gene
    输出: 指定target的预测值 (DO, 即第一个输出通道)
    """

    def __init__(self, model, wq_bg, cgene_bg, target_key, device):
        self.model = model
        self.wq_bg = wq_bg
        self.cgene_bg = cgene_bg
        self.target_key = target_key
        self.device = device

        self._build_fixed_wq_tensors()
        self._build_fixed_c_gene_tensors()

    def _build_fixed_wq_tensors(self):
        """用背景统计值构建固定WQ输入张量"""
        bg = self.wq_bg
        if bg is None:
            zeros_inf = np.zeros((1, WINDOW_SIZE, 7))
            zeros_3 = np.zeros((1, WINDOW_SIZE, 3))
            self.fixed_inf = torch.tensor(zeros_inf, dtype=torch.float32).to(self.device)
            self.fixed_ana = torch.tensor(zeros_3, dtype=torch.float32).to(self.device)
            self.fixed_ano = torch.tensor(zeros_3, dtype=torch.float32).to(self.device)
            self.fixed_ox = torch.tensor(zeros_3, dtype=torch.float32).to(self.device)
            self.fixed_eff = torch.tensor(zeros_3, dtype=torch.float32).to(self.device)
        else:
            inf_val = bg['inf'].reshape(1, 1, -1)
            ana_val = bg['ana'].reshape(1, 1, -1)
            ano_val = bg['ano'].reshape(1, 1, -1)
            ox_val = bg['ox'].reshape(1, 1, -1)
            eff_val = bg['eff'].reshape(1, 1, -1)

            self.fixed_inf = torch.tensor(
                np.repeat(inf_val, WINDOW_SIZE, axis=1), dtype=torch.float32
            ).to(self.device)
            self.fixed_ana = torch.tensor(
                np.repeat(ana_val, WINDOW_SIZE, axis=1), dtype=torch.float32
            ).to(self.device)
            self.fixed_ano = torch.tensor(
                np.repeat(ano_val, WINDOW_SIZE, axis=1), dtype=torch.float32
            ).to(self.device)
            self.fixed_ox = torch.tensor(
                np.repeat(ox_val, WINDOW_SIZE, axis=1), dtype=torch.float32
            ).to(self.device)
            self.fixed_eff = torch.tensor(
                np.repeat(eff_val, WINDOW_SIZE, axis=1), dtype=torch.float32
            ).to(self.device)

    def _build_fixed_c_gene_tensors(self):
        """用背景统计值构建固定c_gene输入张量"""
        bg = self.cgene_bg
        if bg is None:
            self.fixed_c_ana = torch.zeros(1, C_GENE_DIM, dtype=torch.float32).to(self.device)
            self.fixed_c_ano = torch.zeros(1, C_GENE_DIM, dtype=torch.float32).to(self.device)
            self.fixed_c_ox  = torch.zeros(1, C_GENE_DIM, dtype=torch.float32).to(self.device)
        else:
            self.fixed_c_ana = torch.tensor(bg['ana'].reshape(1, -1), dtype=torch.float32).to(self.device)
            self.fixed_c_ano = torch.tensor(bg['ano'].reshape(1, -1), dtype=torch.float32).to(self.device)
            self.fixed_c_ox  = torch.tensor(bg['ox'].reshape(1, -1),  dtype=torch.float32).to(self.device)

    def __call__(self, x):
        """
        Sobol调用接口。

        Args:
            x: (n_samples, 63) numpy array, 仅含 n_gene
               排列: [n_gene_ana(21), n_gene_ano(21), n_gene_ox(21)]
        Returns:
            (n_samples,) 对应target DO输出的numpy数组
        """
        n_samples = x.shape[0]
        self.model.eval()

        results = []
        for start in range(0, n_samples, SOBOL_BATCH_SIZE):
            end = min(start + SOBOL_BATCH_SIZE, n_samples)
            x_batch = x[start:end]
            batch_n = end - start

            # 解析基因输入 (仅 n_gene, 共63维)
            gene_n_ana = torch.tensor(x_batch[:, :N_GENE_DIM], dtype=torch.float32).to(self.device)
            gene_n_ano = torch.tensor(x_batch[:, N_GENE_DIM:N_GENE_DIM*2], dtype=torch.float32).to(self.device)
            gene_n_ox  = torch.tensor(x_batch[:, N_GENE_DIM*2:N_GENE_DIM*3], dtype=torch.float32).to(self.device)

            # 固定 c_gene (训练集均值背景)
            c_ana = self.fixed_c_ana.expand(batch_n, -1)
            c_ano = self.fixed_c_ano.expand(batch_n, -1)
            c_ox  = self.fixed_c_ox.expand(batch_n, -1)

            # 扩展WQ到batch维度
            inf_b = self.fixed_inf.expand(batch_n, -1, -1)
            ana_b = self.fixed_ana.expand(batch_n, -1, -1)
            ano_b = self.fixed_ano.expand(batch_n, -1, -1)
            oxb   = self.fixed_ox.expand(batch_n, -1, -1)
            eff_b = self.fixed_eff.expand(batch_n, -1, -1)

            with torch.no_grad():
                outputs = self.model(inf_b, ana_b, ano_b, oxb, eff_b,
                                     gene_n_ana, gene_n_ano, gene_n_ox,
                                     c_ana, c_ano, c_ox)

            # 提取目标输出 (DO = 第一个输出通道)
            target_out = outputs[self.target_key]  # (batch_n, 3)
            results.append(target_out[:, 0].cpu().numpy())

            del outputs, target_out, gene_n_ana, gene_n_ano, gene_n_ox
            del c_ana, c_ano, c_ox, inf_b, ana_b, ano_b, oxb, eff_b

        return np.concatenate(results)


# ============================================================
# Sobol分析执行（单目标，多次运行取平均）
# ============================================================
def run_sobol_for_target(model, wq_bg, cgene_bg, target_key, target_label,
                         device, all_feature_names, ngene_bounds, n_runs=NUM_SOBOL_RUNS):
    """
    对单个目标(ana/ano/ox)执行全模型Sobol敏感性分析。
    多次运行取平均以稳定估计。

    Returns:
        dict含 mean_ST, mean_S1, all_ST, all_S1, feature_names
    """
    print(f"\n{'='*60}")
    print(f"Full-model Sobol analysis for {target_key} ({target_label})")
    print(f"{'='*60}")

    # 定义Sobol问题: 63个变量 (n_gene 21 × 3池)，使用实际数据范围作为边界
    problem = {
        'num_vars': TOTAL_GENE_FEATURES,
        'names': all_feature_names,
        'bounds': ngene_bounds,
    }

    wrapper = FullModelSobolWrapper(model, wq_bg, cgene_bg, target_key, device)

    all_ST = []   # Total-order indices across runs
    all_S1 = []   # First-order indices across runs

    valid_runs = 0
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...")
        np.random.seed(42 + run)

        try:
            # 生成Sobol采样
            param_values = sobol_sample(problem, SOBOL_N, calc_second_order=CALC_SECOND_ORDER)
            print(f"    Generated {param_values.shape[0]} samples "
                  f"(N={SOBOL_N}, D={TOTAL_GENE_FEATURES})")

            # 执行模型前向传播
            Y = wrapper(param_values)

            # 检查异常值
            if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
                print(f"    Warning: NaN/Inf detected, skipping run {run+1}")
                continue

            # 诊断: 打印Y的统计量，检测方差是否过小
            y_var = np.var(Y)
            print(f"    Y stats: min={Y.min():.6f}, max={Y.max():.6f}, "
                  f"std={Y.std():.6f}, var={y_var:.6e}")
            if y_var < 1e-15:
                print(f"    WARNING: Y variance near-zero! Sobol indices will be NaN.")
                print(f"    Skipping this run. Check input bounds.")
                continue

            # Sobol分析 (对DO输出做敏感性分析)
            Si = sobol.analyze(problem, Y, calc_second_order=CALC_SECOND_ORDER)

            all_ST.append(Si['ST'])
            all_S1.append(Si['S1'])
            valid_runs += 1
            print(f"    ST sum: {np.sum(Si['ST']):.4f}, S1 sum: {np.sum(Si['S1']):.4f}")

        except Exception as e:
            print(f"    Error in run {run+1}: {e}, skipping...")
            continue

        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()

    if valid_runs == 0:
        print(f"  ERROR: All runs failed for {target_key}!")
        return None

    # 多次运行取平均
    all_ST = np.array(all_ST)
    all_S1 = np.array(all_S1)
    mean_ST = np.mean(all_ST, axis=0)
    mean_S1 = np.mean(all_S1, axis=0)
    std_ST = np.std(all_ST, axis=0)
    std_S1 = np.std(all_S1, axis=0)

    print(f"  Completed {valid_runs}/{n_runs} valid runs for {target_key}")
    print(f"  Mean ST total: {np.sum(mean_ST):.4f}, Mean S1 total: {np.sum(mean_S1):.4f}")

    return {
        'mean_ST': mean_ST,
        'mean_S1': mean_S1,
        'std_ST': std_ST,
        'std_S1': std_S1,
        'all_ST': all_ST,
        'all_S1': all_S1,
        'feature_names': all_feature_names,
        'target_key': target_key,
        'target_label': target_label,
        'valid_runs': valid_runs,
    }


# ============================================================
# 绘图 (水平条形图，63特征按池着色)
# ============================================================
def plot_full_model_sobol(results_dict, output_dir):
    """
    绘制每个目标的n_gene Sobol Total-order敏感性条形图。
    63个特征按ST值降序排列，按池子着色（深蓝Anaerobic / 中蓝Anoxic / 浅蓝Aerobic）。
    """
    from matplotlib import rcParams

    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['Arial'],
        "axes.unicode_minus": False
    }
    rcParams.update(config)

    label_size = 16
    tick_label_size = 9

    for target_key, result in results_dict.items():
        if result is None:
            continue

        mean_ST = result['mean_ST']
        names = result['feature_names']
        title = TARGET_CONFIG[target_key]['title']

        fig, ax = plt.subplots(figsize=(14, 12))

        # 按Total-order降序排列
        sorted_idx = np.argsort(mean_ST)[::-1]
        sorted_ST = mean_ST[sorted_idx]
        sorted_names = [names[i] for i in sorted_idx]

        y_pos = np.arange(len(sorted_names))

        # 为每个条形分配对应池子的颜色
        bar_colors = []
        pool_labels_list = []
        for name in sorted_names:
            if name.startswith('n_gene_ana_'):
                bar_colors.append(POOL_COLORS['ana'])
                pool_labels_list.append('Anaerobic')
            elif name.startswith('n_gene_ano_'):
                bar_colors.append(POOL_COLORS['ano'])
                pool_labels_list.append('Anoxic')
            else:
                bar_colors.append(POOL_COLORS['ox'])
                pool_labels_list.append('Aerobic')

        ax.barh(y_pos, sorted_ST, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=tick_label_size)
        ax.set_xlabel('Total-Order Sensitivity Index (ST)', fontsize=label_size)
        ax.set_ylabel('Gene Feature', fontsize=label_size)
        ax.set_title(f'{title} - Full Model Sobol Sensitivity\n(N-Gene Only, {result["valid_runs"]} runs averaged)',
                     fontsize=label_size + 1, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.15, axis='x')

        ax.tick_params(axis='y', labelsize=tick_label_size)
        ax.tick_params(axis='x', labelsize=tick_label_size)

        # 图例: 用不同深浅蓝色区分3个池子
        legend_elements = [
            Patch(facecolor=POOL_COLORS['ana'], label='Anaerobic Pool'),
            Patch(facecolor=POOL_COLORS['ano'], label='Anoxic Pool'),
            Patch(facecolor=POOL_COLORS['ox'],  label='Aerobic Pool'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

        plt.tight_layout()
        path = os.path.join(output_dir, f'Sobol_FullModel_{target_key}_ngene_only.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")


# ============================================================
# 保存CSV（含完整数据便于后续重画图）
# ============================================================
def save_sobol_csv(results_dict, output_dir):
    """
    保存Sobol结果为CSV文件（含原始数据便于后续重画图）。
    输出:
      - Sobol_FullModel_{target}_ngene_only.csv (汇总+每次运行数据)
    """
    for target_key, result in results_dict.items():
        if result is None:
            continue

        mean_ST = result['mean_ST']
        mean_S1 = result['mean_S1']
        std_ST = result['std_ST']
        std_S1 = result['std_S1']
        all_ST = result['all_ST']  # (n_runs, n_features)
        all_S1 = result['all_S1']
        names = result['feature_names']

        df_data = {
            'feature': names,
            'pool': [],
            'gene_name': [],
            'mean_ST': mean_ST,
            'std_ST': std_ST,
            'mean_S1': mean_S1,
            'std_S1': std_S1,
        }

        # 添加每次运行的ST/S1数据
        for run_idx in range(all_ST.shape[0]):
            df_data[f'run_{run_idx+1}_ST'] = all_ST[run_idx]
            df_data[f'run_{run_idx+1}_S1'] = all_S1[run_idx]

        # 填充池子和基因名信息
        for i, fname in enumerate(names):
            if fname.startswith('n_gene_ana_'):
                df_data['pool'].append('ana')
                df_data['gene_name'].append(fname.replace('n_gene_ana_', ''))
            elif fname.startswith('n_gene_ano_'):
                df_data['pool'].append('ano')
                df_data['gene_name'].append(fname.replace('n_gene_ano_', ''))
            else:
                df_data['pool'].append('ox')
                df_data['gene_name'].append(fname.replace('n_gene_ox_', ''))

        df = pd.DataFrame(df_data)
        df_sorted = df.sort_values(by='mean_ST', ascending=False)

        csv_path = os.path.join(output_dir, f'Sobol_FullModel_{target_key}_ngene_only.csv')
        df_sorted.to_csv(csv_path, index=False)
        print(f"  CSV saved: {csv_path} ({len(names)} features x {all_ST.shape[0]+7} cols)")


# ============================================================
# 打印摘要
# ============================================================
def print_summary(results_dict, all_feature_names):
    """打印每个目标的Top 10敏感基因"""
    print("\n" + "=" * 60)
    print("Sobol Sensitivity Analysis Summary (Full Model, n-gene only)")
    print("=" * 60)

    for target_key, cfg in TARGET_CONFIG.items():
        result = results_dict.get(target_key)
        if result is None:
            print(f"\n{cfg['title']}: NO VALID RESULTS")
            continue

        mean_ST = result['mean_ST']
        sorted_idx = np.argsort(mean_ST)[::-1]

        print(f"\n--- {cfg['title']} ({target_key}) ---")
        print(f"  Valid runs: {result['valid_runs']}, Top 10 genes by Total-Order (ST):")
        for rank, idx in enumerate(sorted_idx[:10]):
            fname = all_feature_names[idx]
            print(f"    {rank+1:2d}. {fname:30s}  ST={mean_ST[idx]:.6f}  S1={result['mean_S1'][idx]:.6f}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("Full-Model Sobol Sensitivity Analysis (n-gene only)")
    print("=" * 60)
    print(f"Total gene features: {TOTAL_GENE_FEATURES} (n_gene only, {N_GENE_DIM}x3 pools)")
    print(f"Sobol config: N={SOBOL_N}, runs={NUM_SOBOL_RUNS}, batch={SOBOL_BATCH_SIZE}")
    print(f"c_gene ({C_GENE_DIM} dim) fixed to training set mean background")
    print(f"WQ fixed to training set mean background")
    print(f"Second-order calculation: {CALC_SECOND_ORDER}")

    # 1. 加载模型
    model, device = load_trained_model()
    if model is None:
        print("ERROR: Failed to load model!")
        return

    # 2. 加载WQ背景统计
    print("\nLoading WQ background statistics...")
    wq_bg = load_wq_background_stats()

    # 3. 加载c_gene背景统计
    print("\nLoading c_gene background statistics...")
    cgene_bg = load_cgene_background_stats()

    # 4. 获取基因名称 (仅n_gene, 共63维)
    n_gene_names, all_feature_names = get_gene_names()

    # 4b. 加载n_gene实际数据范围 (用于Sobol采样边界)
    print("\nLoading n_gene data ranges for Sobol bounds...")
    ngene_bounds = load_ngene_data_ranges()

    # 5. 创建输出目录
    output_dir = os.path.join('results')
    os.makedirs(output_dir, exist_ok=True)

    # 6. 对每个目标执行Sobol分析
    results_dict = {}

    for target_key, cfg in TARGET_CONFIG.items():
        result = run_sobol_for_target(
            model, wq_bg, cgene_bg, cfg['output_key'], cfg['label'],
            device, all_feature_names, ngene_bounds, n_runs=NUM_SOBOL_RUNS
        )
        results_dict[target_key] = result

        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()

    # 7. 绘图
    print("\nGenerating plots...")
    plot_full_model_sobol(results_dict, output_dir)

    # 8. 保存CSV
    print("\nSaving CSV files...")
    save_sobol_csv(results_dict, output_dir)

    # 9. 打印摘要
    print_summary(results_dict, all_feature_names)

    print("\n" + "=" * 60)
    print("Full-Model Sobol Analysis Complete! (n-gene only)")
    print("=" * 60)


if __name__ == "__main__":
    main()
