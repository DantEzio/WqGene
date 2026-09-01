"""
全模型SHAP基因重要性分析（仅n_gene版本）
=============================================================
对完整GeneModel进行SHAP分析（而非仅gene embedding layer），
将n_gene(21维×3池)=63维基因特征作为SHAP输入，
c_gene固定为训练集均值背景，水质(WQ)特征固定为训练集均值背景。
输出按ana/ano/ox分别绘图，保存到results文件夹。
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import gc

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import shap

sys.path.append(os.path.join(os.path.dirname(__file__), 'step1_gene_model_trans'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'step1_gene_model_trans'))

from gene_model import GeneModel


# ============================================================
# 配置
# ============================================================
WINDOW_SIZE = 40          # 与模型一致的时间窗口
NUM_SHAP_RUNS = 50        # SHAP采样运行次数（多次平均）
BG_SAMPLES = 200           # 背景数据样本数 (降低以减少总评估次数)
TEST_SAMPLES = 200         # 测试数据样本数 (降低以减少总评估次数)
NSAMPLES = 500            # 每个测试样本的SHAP coalition数 (替代'auto'=2174)
SHAP_BATCH_SIZE = 256     # 模型前向传播mini-batch大小，防止GPU OOM
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


# ============================================================
# 模型加载
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
    返回各WQ分量的均值字典，供SHAP分析时固定使用。
    """
    wq_scaled_path = os.path.join('data_WQ', 'wq_scaled.csv')
    if not os.path.exists(wq_scaled_path):
        print(f"WQ data not found: {wq_scaled_path}, using zero background")
        return None

    df = pd.read_csv(wq_scaled_path)
    # 取前30%作为训练集范围（与train_gene_model.py一致）
    train_end = int(0.3 * len(df))
    train_df = df.iloc[:train_end]

    stats = {}
    # INF变量 (7维)
    inf_vars = ['F-INF_MA_smooth', 'T-INF_MA_smooth', 'pH-INF_MA_smooth',
                'COD-INF_MA_smooth', 'C/N-INF_MA_smooth', 'TN-INF_MA_smooth', 'AN-INF_MA_smooth']
    stats['inf'] = train_df[inf_vars].mean().values  # (7,)

    # ANA变量 (3维)
    ana_vars = ['DO-ANA_MA_smooth', 'AN-ANA_MA_smooth', 'NN-ANA_MA_smooth']
    stats['ana'] = train_df[ana_vars].mean().values  # (3,)

    # ANO变量 (3维)
    ano_vars = ['DO-ANO_MA_smooth', 'AN-ANO_MA_smooth', 'NN-ANO_MA_smooth']
    stats['ano'] = train_df[ano_vars].mean().values  # (3,)

    # OX变量 (3维)
    ox_vars = ['DO-OX_MA_smooth', 'AN-OX_MA_smooth', 'NN-OX_MA_smooth']
    stats['ox'] = train_df[ox_vars].mean().values     # (3,)

    # EFF变量 (3维)
    eff_vars = ['COD-EFF_MA_smooth', 'TN-EFF_MA_smooth', 'AN-EFF_MA_smooth']
    stats['eff'] = train_df[eff_vars].mean().values    # (3,)

    print("WQ background statistics computed from training set")
    return stats


def load_cgene_background_stats():
    """
    从训练集计算c_gene特征的均值，用于构建固定背景。
    与WQ背景统计类似，取前30%训练样本对应的c_gene均值。
    返回各c_gene分量的均值字典。
    """
    # 1. 加载WQ数据获取时间戳
    wq_scaled_path = os.path.join('data_WQ', 'wq_scaled.csv')
    wq_raw_path = os.path.join('data_WQ', 'wq.xlsx')
    if not os.path.exists(wq_scaled_path) or not os.path.exists(wq_raw_path):
        print("WQ data not found for c_gene background, using zero background")
        return None

    wq_data = pd.read_csv(wq_scaled_path)
    wq_raw = pd.read_excel(wq_raw_path)
    wq_data['time'] = wq_raw['time']

    # 训练集范围 (与train_gene_model.py一致)
    n_samples = len(wq_data) - WINDOW_SIZE
    train_size = int(0.3 * n_samples)

    # 2. 加载c_gene数据 (与 prepare_data() 一致)
    c_gene_path = os.path.join('data_Gene', 'c_gene_dynamic_common.xlsx')
    if not os.path.exists(c_gene_path):
        print(f"c_gene data not found: {c_gene_path}, using zero background")
        return None

    # c_gene ana_ano sheet
    gene_c_dynamic = pd.read_excel(c_gene_path, sheet_name='ana_ano')
    gene_c_dynamic['KEGG'] = gene_c_dynamic.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
    gene_c_dynamic = gene_c_dynamic.drop(columns=gene_c_dynamic.columns[-2])
    gene_c_dynamic = gene_c_dynamic.dropna(subset=['KEGG'])
    gene_c_dynamic = gene_c_dynamic.set_index('KEGG')

    # c_gene ox sheet
    gene_c_dynamic_ox = pd.read_excel(c_gene_path, sheet_name='ox')
    gene_c_dynamic_ox['KEGG'] = gene_c_dynamic_ox.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
    gene_c_dynamic_ox = gene_c_dynamic_ox.drop(columns=gene_c_dynamic_ox.columns[-2])
    gene_c_dynamic_ox = gene_c_dynamic_ox.dropna(subset=['KEGG'])
    gene_c_dynamic_ox = gene_c_dynamic_ox.set_index('KEGG')

    # 构建月份映射 (与 prepare_data() 一致)
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

    # 3. 统计训练集中每个月的样本数
    from collections import Counter
    month_counter_ana = Counter()
    month_counter_ox = Counter()

    for idx in range(train_size):
        timestamp = wq_data['time'].iloc[idx + WINDOW_SIZE]
        month = timestamp.month

        # c_gene ana/ano 月份匹配 (与 _get_gene_for_time 一致)
        if month in month_to_col_c:
            target_month = month
        else:
            candidates = [m for m in available_months_c if m <= month]
            target_month = candidates[-1] if candidates else available_months_c[0]
        month_counter_ana[target_month] += 1

        # c_gene ox 月份匹配
        if month in month_to_col_c_ox:
            target_month_ox = month
        else:
            candidates_ox = [m for m in available_months_c_ox if m <= month]
            target_month_ox = candidates_ox[-1] if candidates_ox else available_months_c_ox[0]
        month_counter_ox[target_month_ox] += 1

    # 4. 按月加权平均
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
        'ana': c_ana_mean.astype(np.float32),  # c_gene ana mean (ano共用)
        'ano': c_ana_mean.astype(np.float32),  # 与ana同一组数据
        'ox':  c_ox_mean.astype(np.float32),   # c_gene ox mean
    }

    print(f"c_gene background statistics computed from {train_size} training samples")
    print(f"  ana/ano months: {dict(month_counter_ana)}")
    print(f"  ox months: {dict(month_counter_ox)}")
    return stats


# ============================================================
# 基因名称获取
# ============================================================
def get_gene_names():
    """获取n_gene的KEGG ID列表"""
    n_gene_names = []

    # n_gene (21个)
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

    # 构建仅 n_gene 的特征名列表: [n_gene_ana(21), n_gene_ano(21), n_gene_ox(21)]
    all_feature_names = []
    prefixes_ng = ['n_gene_ana_', 'n_gene_ano_', 'n_gene_ox_']
    for prefix in prefixes_ng:
        for name in n_gene_names:
            all_feature_names.append(prefix + name)

    return n_gene_names, all_feature_names


# ============================================================
# 全模型SHAP包装器
# ============================================================
class FullModelSHAPWrapper:
    """
    将完整GeneModel包装为SHAP可调用的函数（仅n_gene版本）。
    c_gene固定为训练集均值背景，WQ固定为训练集均值背景。
    输入: (n_samples, 63) 仅含 n_gene
    输出: 指定target的预测值
    """

    def __init__(self, model, wq_bg, cgene_bg, target_key, device):
        self.model = model
        self.wq_bg = wq_bg
        self.cgene_bg = cgene_bg
        self.target_key = target_key
        self.device = device

        # 预构造固定的WQ张量: (1, WINDOW_SIZE, dim)
        self._build_fixed_wq_tensors()
        # 预构造固定的c_gene张量（训练集均值背景）
        self._build_fixed_c_gene_tensors()

    def _build_fixed_wq_tensors(self):
        """用背景统计值构建固定WQ输入张量"""
        bg = self.wq_bg
        if bg is None:
            # 无统计数据时用零填充
            zeros_inf = np.zeros((1, WINDOW_SIZE, 7))
            zeros_3 = np.zeros((1, WINDOW_SIZE, 3))
            self.fixed_inf = torch.tensor(zeros_inf, dtype=torch.float32).to(self.device)
            self.fixed_ana = torch.tensor(zeros_3, dtype=torch.float32).to(self.device)
            self.fixed_ano = torch.tensor(zeros_3, dtype=torch.float32).to(self.device)
            self.fixed_ox = torch.tensor(zeros_3, dtype=torch.float32).to(self.device)
            self.fixed_eff = torch.tensor(zeros_3, dtype=torch.float32).to(self.device)
        else:
            # 用均值扩展到window_size时间步
            inf_val = bg['inf'].reshape(1, 1, -1)      # (1,1,7)
            ana_val = bg['ana'].reshape(1, 1, -1)       # (1,1,3)
            ano_val = bg['ano'].reshape(1, 1, -1)       # (1,1,3)
            ox_val = bg['ox'].reshape(1, 1, -1)         # (1,1,3)
            eff_val = bg['eff'].reshape(1, 1, -1)       # (1,1,3)

            self.fixed_inf = torch.tensor(
                np.repeat(inf_val, WINDOW_SIZE, axis=1), dtype=torch.float32
            ).to(self.device)                            # (1,40,7)
            self.fixed_ana = torch.tensor(
                np.repeat(ana_val, WINDOW_SIZE, axis=1), dtype=torch.float32
            ).to(self.device)                            # (1,40,3)
            self.fixed_ano = torch.tensor(
                np.repeat(ano_val, WINDOW_SIZE, axis=1), dtype=torch.float32
            ).to(self.device)                            # (1,40,3)
            self.fixed_ox = torch.tensor(
                np.repeat(ox_val, WINDOW_SIZE, axis=1), dtype=torch.float32
            ).to(self.device)                            # (1,40,3)
            self.fixed_eff = torch.tensor(
                np.repeat(eff_val, WINDOW_SIZE, axis=1), dtype=torch.float32
            ).to(self.device)                            # (1,40,3)

    def _build_fixed_c_gene_tensors(self):
        """用背景统计值构建固定c_gene输入张量（训练集均值）"""
        bg = self.cgene_bg
        if bg is None:
            # 无统计数据时用零填充
            self.fixed_c_ana = torch.zeros(1, C_GENE_DIM, dtype=torch.float32).to(self.device)
            self.fixed_c_ano = torch.zeros(1, C_GENE_DIM, dtype=torch.float32).to(self.device)
            self.fixed_c_ox  = torch.zeros(1, C_GENE_DIM, dtype=torch.float32).to(self.device)
        else:
            # 用训练集均值
            self.fixed_c_ana = torch.tensor(bg['ana'].reshape(1, -1), dtype=torch.float32).to(self.device)
            self.fixed_c_ano = torch.tensor(bg['ano'].reshape(1, -1), dtype=torch.float32).to(self.device)
            self.fixed_c_ox  = torch.tensor(bg['ox'].reshape(1, -1),  dtype=torch.float32).to(self.device)

    def __call__(self, x):
        """
        SHAP调用接口。

        Args:
            x: (n_samples, 63) numpy array, 仅含 n_gene
               排列: [n_gene_ana(21), n_gene_ano(21), n_gene_ox(21)]
        Returns:
            (n_samples,) 对应target输出的numpy数组
        """
        n_samples = x.shape[0]
        self.model.eval()

        results = []
        for start in range(0, n_samples, SHAP_BATCH_SIZE):
            end = min(start + SHAP_BATCH_SIZE, n_samples)
            x_batch = x[start:end]
            batch_n = end - start

            # 解析基因输入 (仅 n_gene)
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

            # 提取目标输出 (取第一个输出通道，如DO)
            target_out = outputs[self.target_key]  # (batch_n, 3)
            results.append(target_out[:, 0].cpu().numpy())

            # 显式释放中间张量
            del outputs, target_out, gene_n_ana, gene_n_ano, gene_n_ox
            del c_ana, c_ano, c_ox, inf_b, ana_b, ano_b, oxb, eff_b

        return np.concatenate(results)


# ============================================================
# SHAP分析执行
# ============================================================
def run_shap_for_target(model, wq_bg, cgene_bg, target_key, target_label,
                        device, all_feature_names, n_runs=NUM_SHAP_RUNS):
    """
    对单个目标(ana/ano/ox)执行全模型SHAP分析。

    Returns:
        dict含 mean_feature_importance, all_importance, feature_names
    """
    print(f"\n{'='*60}")
    print(f"Full-model SHAP analysis for {target_key} ({target_label})")
    print(f"{'='*60}")

    wrapper = FullModelSHAPWrapper(model, wq_bg, cgene_bg, target_key, device)

    all_feature_importance = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...")
        np.random.seed(42 + run)

        # 背景数据: 在合理范围内随机生成基因表达值
        bg_data = np.random.uniform(-0.5, 0.5, size=(BG_SAMPLES, TOTAL_GENE_FEATURES))

        # 测试数据
        test_data = np.random.uniform(-0.5, 0.5, size=(TEST_SAMPLES, TOTAL_GENE_FEATURES))

        explainer = shap.KernelExplainer(wrapper, bg_data)
        shap_values = explainer.shap_values(test_data, nsamples=NSAMPLES)

        # 处理SHAP结果
        if isinstance(shap_values, list):
            shap_arr = np.array(shap_values)  # 可能是list of arrays
            if shap_arr.ndim == 3:
                fi = np.mean(np.abs(shap_arr), axis=(0, 1))  # (n_features,)
            elif shap_arr.ndim == 2:
                fi = np.mean(np.abs(shap_arr), axis=0)
            else:
                fi = np.abs(np.array(shap_values)).flatten()
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 2:
                fi = np.mean(np.abs(shap_values), axis=0)
            elif shap_values.ndim == 3:
                fi = np.mean(np.abs(shap_values), axis=(0, 1))
            else:
                fi = np.abs(shap_values)
        else:
            fi = np.zeros(TOTAL_GENE_FEATURES)

        all_feature_importance.append(fi)

        # 清理本轮显存
        del explainer, shap_values
        gc.collect()
        torch.cuda.empty_cache()

    # 多次运行取平均
    all_feature_importance = np.array(all_feature_importance)
    mean_fi = np.mean(all_feature_importance, axis=0)

    print(f"  Completed {n_runs} runs for {target_key}")

    return {
        'mean_feature_importance': mean_fi,
        'all_feature_importance': all_feature_importance,
        'feature_names': all_feature_names,
        'target_key': target_key,
        'target_label': target_label,
    }


# ============================================================
# 绘图
# ============================================================
def plot_full_model_shap(results_dict, n_gene_names, output_dir):
    """
    绘制每个目标的n_gene重要性条形图（统一蓝色，仅n_gene版本）。
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

    TOP_N = 21       # 显示Top 21个n_gene
    BAR_COLOR = '#2171B5'  # 统一蓝色

    for target_key, result in results_dict.items():
        mean_fi = result['mean_feature_importance']
        names = result['feature_names']
        title = TARGET_CONFIG[target_key]['title']

        # ---- n_gene重要性条形图 (合并3个池子) ----
        fig, ax = plt.subplots(figsize=(12, 8))

        sorted_idx = np.argsort(mean_fi)[::-1][:TOP_N]
        top_vals = mean_fi[sorted_idx]
        top_names = [names[i] for i in sorted_idx]

        y_pos = np.arange(len(top_names))

        ax.barh(y_pos, top_vals, color=BAR_COLOR,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=10)
        ax.set_xlabel('Mean |SHAP Value| (Full Model)', fontsize=13)
        ax.set_ylabel('Gene Feature', fontsize=13)
        ax.set_title(f'{title} - Top {TOP_N} N-Gene Importance\n(Full Model SHAP, n-gene only)',
                     fontsize=14, fontweight='bold')

        # 图例: 用不同深浅蓝色区分3个池子
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#08306B', label='Anaerobic Pool'),
            Patch(facecolor='#2171B5', label='Anoxic Pool'),
            Patch(facecolor='#6BAED6', label='Aerobic Pool'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        ax.invert_yaxis()

        plt.tight_layout()
        path_n = os.path.join(output_dir, f'SHAP_FullModel_{target_key}_ngene_only.png')
        fig.savefig(path_n, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path_n}")


# ============================================================
# 保存CSV（含完整数据便于后续重画图）
# ============================================================
def save_shap_csv(results_dict, output_dir):
    """
    保存SHAP结果为CSV文件（含原始数据便于后续重画图）。
    输出:
      - SHAP_FullModel_importance_{target}_ngene_only.csv (汇总+每次运行数据)
    """
    for target_key, result in results_dict.items():
        mean_fi = result['mean_feature_importance']
        all_fi = result['all_feature_importance']  # (n_runs, n_features)
        names = result['feature_names']

        # 构建完整DataFrame
        df_data = {
            'feature': names,
            'pool': [],          # 所属池子 (ana/ano/ox)
            'gene_name': [],     # 原始基因名
            'mean_abs_SHAP': mean_fi,
            'std_abs_SHAP': np.std(all_fi, axis=0),
        }

        # 添加每次运行的数据
        for run_idx in range(all_fi.shape[0]):
            df_data[f'run_{run_idx+1}_abs_SHAP'] = all_fi[run_idx]

        # 填充池子和基因名信息
        for i, name in enumerate(names):
            if name.startswith('n_gene_ana_'):
                df_data['pool'].append('ana')
                df_data['gene_name'].append(name.replace('n_gene_ana_', ''))
            elif name.startswith('n_gene_ano_'):
                df_data['pool'].append('ano')
                df_data['gene_name'].append(name.replace('n_gene_ano_', ''))
            else:
                df_data['pool'].append('ox')
                df_data['gene_name'].append(name.replace('n_gene_ox_', ''))

        df = pd.DataFrame(df_data)
        df_sorted = df.sort_values(by='mean_abs_SHAP', ascending=False)

        csv_path = os.path.join(output_dir, f'SHAP_FullModel_importance_{target_key}_ngene_only.csv')
        df_sorted.to_csv(csv_path, index=False)
        print(f"  CSV saved: {csv_path} ({len(names)} features x {all_fi.shape[0]+3} cols)")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("Full-Model SHAP Gene Importance Analysis (n-gene only)")
    print("=" * 60)
    print(f"Total gene features: {TOTAL_GENE_FEATURES} (n_gene only, {N_GENE_DIM}x3)")
    print(f"c_gene ({C_GENE_DIM} dim) fixed to training set mean background")

    # 1. 加载模型
    model, device = load_trained_model()
    if model is None:
        print("ERROR: Failed to load model!")
        return

    # 2. 加载WQ背景统计
    print("\nLoading WQ background statistics...")
    wq_bg = load_wq_background_stats()

    # 2b. 加载c_gene背景统计
    print("\nLoading c_gene background statistics...")
    cgene_bg = load_cgene_background_stats()

    # 3. 获取基因名称 (仅n_gene)
    n_gene_names, all_feature_names = get_gene_names()

    # 4. 创建输出目录
    output_dir = os.path.join('results')
    os.makedirs(output_dir, exist_ok=True)

    # 5. 对每个目标执行SHAP分析
    results_dict = {}

    for target_key, cfg in TARGET_CONFIG.items():
        result = run_shap_for_target(
            model, wq_bg, cgene_bg, cfg['output_key'], cfg['label'],
            device, all_feature_names, n_runs=NUM_SHAP_RUNS
        )
        results_dict[target_key] = result

        # 打印Top 10基因
        mean_fi = result['mean_feature_importance']
        sorted_idx = np.argsort(mean_fi)[::-1]
        print(f"\n  Top 10 most important features for {cfg['title']}:")
        for rank, idx in enumerate(sorted_idx[:10]):
            print(f"    {rank+1}. {all_feature_names[idx]}: {mean_fi[idx]:.6f}")

        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()

    # 6. 绘图
    print("\nGenerating plots...")
    plot_full_model_shap(results_dict, n_gene_names, output_dir)

    # 7. 保存CSV（含完整数据便于重画图）
    print("\nSaving CSV files...")
    save_shap_csv(results_dict, output_dir)

    print("\n" + "=" * 60)
    print("Full-Model SHAP Analysis Complete! (n-gene only)")
    print("=" * 60)


if __name__ == "__main__":
    main()
