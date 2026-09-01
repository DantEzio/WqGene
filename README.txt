# AI-Bio 项目说明文档

## 项目简介
AI-Bio 是一个结合基因数据和水质数据的深度学习预测系统，用于预测污水处理过程中的水质参数。项目实现了多种模型架构（GeneModel、LSTM、MLP、WqLSTM），并提供了完整的训练/测试/迁移学习流程以及 SHAP 基因重要性分析。

## 项目结构

```
AI-Bio/
├── data_Gene/                    # 基因数据（c_gene / n_gene 动态表达谱）
│   ├── c_gene_dynamic_common.xlsx # c基因动态数据 (ana_ano/ox sheet)
│   └── n_gene_dynamic_common.xlsx # n基因动态数据 (ana_ano/ox sheet)
├── data_GeneM/                   # 迁移学习用基因数据
├── data_WQ/                      # 水质数据
│   ├── wq.xlsx                   # 原始水质数据（含时间戳）
│   ├── wq_scaled.csv             # 标准化后的水质数据
│   └── wq_scaleddict.npy         # 标准化参数字典
├── results/                      # 分析结果输出目录
│   ├── *.png                     # 图表（SHAP条形图、热力图等）
│   └── *.csv                     # 数据结果（SHAP重要性分数等）
├── step0_getGene/                # 基因数据处理
├── step1_gene_model_trans/       # GeneModel 模型（主模型）
│   ├── gene_model.py             # 模型定义 (GeneModel + Dataset)
│   ├── train_gene_model.py      # 模型训练
│   ├── test_gene_model.py       # 模型测试
│   ├── test_gene_model_transfer_learning.py  # 迁移学习测试
│   └── model/gene_model/        # 保存的模型权重 (.pth)
├── step1_LSTM_model_trans/       # LSTM 基准模型
├── step1_MLP_model_trans/        # MLP 基准模型
├── step1_WqLSTM_model_trans/     # WqLSTM 基准模型
├── step2_geneM_model_trans/      # GeneModel 迁移学习阶段
├── extract_common_c_gene.py      # 提取公共c基因脚本
├── extract_common_n_gene.py      # 提取公共n基因脚本
│
├── results1_training_test_Bollinger.py   # 训练/测试 Bollinger带分析
├── results2_trans_test_Bolling.py        # 迁移测试 Bollinger带分析
├── results3_geneEmbedding.py             # 基因嵌入可视化
├── results3_geneSharply.py              # 基因层SHAP分析
├── results3_geneSobo.py                 # 基因层Sobol敏感性分析
├── results3_Sharply.py                  # 全模型SHAP分析（c_gene+WQ固定背景）
├── results4_trans_shaply.py            # 迁移学习SHAP分析
├── results5_trans_shaply_heatmap.py    # SHAP热力图绘制
├── results6_shap_vs_wq_step1.py         # SHAP vs WQ相关性 Step1
└── results6_shap_vs_wq_step2.py         # SHAP vs WQ相关性 Step2
```

## 核心功能

### 1. 数据处理
- 基因数据处理：从 Excel 动态基因表达数据中按月份匹配提取特征（KEGG ID）
- 水质数据处理：多池（INF/ANA/ANO/OX/EFF）时间序列标准化预处理
- 公共基因提取：跨样本筛选高一致性 c_gene / n_gene

### 2. 模型架构

| 模型 | 输入特征 | 目录 | 说明 |
|------|----------|------|------|
| **GeneModel** | WQ(19维×40步) + n_gene(21维×3池) + c_gene(1205维×3池) | step1_gene_model_trans | 主模型，含基因嵌入层 |
| **LSTMModel** | WQ 时间序列 | step1_LSTM_model_trans | LSTM 基准 |
| **MLPModel** | WQ 时间序列展平 | step1_MLP_model_trans | MLP 基准 |
| **WqLSTMModel** | WQ 时间序列 | step1_WqLSTM_model_trans | 纯WQ-LSTM |

所有模型均为多输出（ANA/ANO/OX 各 3 维 = DO, AN, NN）。

### 3. 训练与测试流程
- **step1**: 在原始数据集上训练各模型
- **step2**: 使用 data_GeneM 数据进行迁移学习微调
- 支持早停机制、学习率自动调整

### 4. 可解释性分析

#### SHAP 分析
- **results3_Sharply.py**: 全模型 KernelSHAP，n_gene 作为可变输入，
  c_gene 与 WQ 固定为训练集均值背景，分别对 ANA/ANO/OX 输出计算 SHAP 值
- **results3_geneSharply.py**: 仅对基因嵌入层的 SHAP 分析
- **results4_trans_shaply.py**: 迁移学习模型的 SHAP 分析
- **results5_trans_shaply_heatmap.py**: SHAP 结果热力图可视化

#### Sobol 敏感性分析
- **results3_geneSobo.py**: 基于 SALib 的 Sobol 指数计算

#### 相关性分析
- **results6_shap_vs_wq_*.py**: SHAP 重要性 vs WQ 参数的统计相关性

## 使用方法

### 1. 环境配置
```bash
pip install -r requirements.txt
```

### 2. 数据准备
确保以下文件存在：
- `data_Gene/c_gene_dynamic_common.xlsx` — c基因动态表达谱
- `data_Gene/n_gene_dynamic_common.xlsx` — n基因动态表达谱
- `data_WQ/wq.xlsx` — 原始水质数据（含 time 列）
- `data_WQ/wq_scaled.csv` — 标准化后水质数据

### 3. 模型训练
```bash
# 训练主模型 (GeneModel)
cd step1_gene_model_trans
python train_gene_model.py

# 训练基准模型 (可选)
cd ../step1_LSTM_model_trans && python train_LSTM_model.py
cd ../step1_MLP_model_trans && python train_MLP_model.py
```

### 4. 模型测试
```bash
cd step1_gene_model_trans
python test_gene_model.py
python test_gene_model_transfer_learning.py   # 迁移学习测试
```

### 5. SHAP / 敏感性分析
```bash
# 全模型SHAP分析（c_gene+WQ固定为训练集均值）
python results3_Sharply.py

# 基因层SHAP分析
python results3_geneSharply.py

# Sobol敏感性分析
python results3_geneSobo.py

# 迁移学习SHAP分析
python results4_trans_shaply.py

# SHAP热力图
python results5_trans_shaply_heatmap.py

# SHAP vs WQ相关性分析
python results6_shap_vs_wq_step1.py
python results6_shap_vs_wq_step2.py
```

### 6. 其他分析
```bash
# 训练/测试Bollinger带
python results1_training_test_Bollinger.py
python results2_trans_test_Bolling.py

# 基因嵌入可视化
python results3_geneEmbedding.py
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3.12+ |
| 深度学习框架 | PyTorch >= 2.0 |
| 数据处理 | pandas, numpy, openpyxl |
| 数据可视化 | matplotlib |
| 统计分析 | scipy, scikit-learn |
| 模型解释 | SHAP >= 0.42 |
| 敏感性分析 | SALib >= 1.4 |

## 关键配置参数 (results3_Sharply.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| WINDOW_SIZE | 40 | 时间窗口长度 |
| N_GENE_DIM | 21 | n_gene 维度 |
| C_GENE_DIM | 1205 | c_gene 维度 |
| NUM_SHAP_RUNS | 10 | SHAP 采样运行次数 |
| BG_SAMPLES | 20 | 背景数据样本数 |
| c_gene 背景 | 训练集均值（按月加权） | 非零值 |
| WQ 背景 | 训练集均值 | 非零值 |

## 注意事项

1. 项目使用相对路径，请从项目根目录运行脚本
2. SHAP 分析（尤其是全模型）计算量较大，建议在 GPU 环境下运行
3. c_gene 背景值通过读取训练集前 30% 样本对应的月份基因数据加权平均得到
4. 模型权重文件较大（每个模型约数百 MB），确保有足够磁盘空间

---

*版本：2.0.0*
*最后更新：2026-07-10*
