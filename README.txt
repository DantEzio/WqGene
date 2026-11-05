# AI-Bio Project Documentation
================================================================

## 1. Overview

AI-Bio is a deep-learning prediction system that combines **water quality time series (WQ)**
with **functional gene expression profiles (n_gene / c_gene)** to predict the key water
quality parameters of each reactor tank in a wastewater treatment process.

Core ideas:
- A two-branch network encodes the WQ sequence and the gene expression profile at the same
  time, demonstrating the gain that gene information brings to water quality prediction.
- **Per-step transfer learning (EWC incremental fine-tuning)** simulates an online operation
  scenario.
- Two sensitivity analyses, **SHAP / PDP**, track how gene importance evolves over time and
  are cross-checked against the gene dynamic data (MT) to select genes that stay stable
  across indicators.

The model always produces 6 outputs: **DO (dissolved oxygen), AN (ammonia nitrogen) and
NN (nitrate nitrogen) in the ANA / ANO / OX zones**.

---

## 2. Directory layout

```
AI-Bio/
├── data_WQ/                              # water quality data
│   ├── wq.xlsx                           # raw WQ data (time column, 15-min steps)
│   ├── wq_scaled.csv                     # standardized WQ data (model input)
│   └── wq_scaleddict.npy                 # standardization dict (for de-normalization)
├── data_Gene/                            # source-domain gene data
│   ├── n_gene_dynamic.xlsx               # raw n_gene expression profiles (ana_ano / ox)
│   ├── c_gene_dynamic.xlsx               # raw c_gene expression profiles
│   ├── n_gene_dynamic_common.xlsx        # genes shared with data_GeneM
│   ├── c_gene_dynamic_common.xlsx
│   └── remove_zero_rows.py               # small gene-table cleaning utility
├── gene_ref/
│   └── Gene_info.xlsx                    # KEGG KO -> gene name mapping (read by results5~9)
├── data_GeneM/                           # target-domain gene data (for transfer learning)
│   ├── n_gene_dynamic.xlsx / c_gene_dynamic.xlsx
│   └── n_gene_dynamic_common.xlsx / c_gene_dynamic_common.xlsx
├── step0_dataprocess/                    # data preprocessing notebook
│   └── data_process.ipynb
├── step1_gene_model_trans_2block_FullTL/ # main model: two-branch GeneModel (FullTL)
│   ├── gene_model.py                     # model definition + Dataset + prepare_data
│   ├── train_gene_model.py               # source-domain training
│   ├── test_gene_model.py                # source-domain testing
│   ├── test_gene_model_transfer_learning.py  # per-step transfer learning (EWC)
│   └── model/, results/                  # weights (.pth) and prediction results
├── step1_gene_model_trans_2block_FullTL - AEGeneRemoveWithTime/   # ablation: no gene input
├── step1_gene_model_trans_2block_FullTL - AEWqRemoveWithTime/     # ablation: no WQ input
├── step1_LSTM_model_trans/               # baseline: WQ-only LSTM
├── step1_MLP_model_trans/                # baseline: WQ-only MLP
├── results/                              # all figures and intermediate results
│   ├── IMP/                              # per-window SHAP importance summaries
│   ├── PDP/                              # per-window PDP sensitivity summaries
│   ├── Shapely for trans/step_*/         # per-step SHAP results
│   ├── PDP for trans/step_*/             # per-step PDP results
│   └── preVStrue/                        # real vs predicted comparison figures
├── *.py                                  # root-level analysis scripts (see section 4)
├── requirements.txt
└── README.txt
```

> Note: every root-level script uses the **project root** as the working directory; internal
> paths are resolved relative to the script location.

---

## 3. Environment setup

```bash
pip install -r requirements.txt
```

- Python >= 3.9 (3.10+ recommended)
- With an NVIDIA GPU, install the CUDA-enabled `torch` build following the official command
- Before the first Excel-related run, make sure both `openpyxl` (.xlsx) and `xlrd` (.xls) are installed

---

## 4. Root-level scripts (in execution order)

| Step | Script | Purpose | Main output |
|------|--------|---------|-------------|
| 0 | `extract_common_c_gene.py` / `extract_common_n_gene.py` | Keep only the KEGG rows shared by data_Gene and data_GeneM so both domains have identical feature dimensions | `*_dynamic_common.xlsx` |
| 1 | `step1_*/train_*.py` | Train the main model and the baselines (source domain) | `best_*.pth`, `training_history_with_lr.npy` |
| 2 | `results1_training_test_Bollinger.py` | Three-model source-domain comparison (loss curves + error Bollinger bands + MSE distribution) | `results/model_performance_comparison.png` |
| 3 | `plot_all_features_comparison.py` | One 4x3 all-feature "real vs predicted" figure per model | `results/preVStrue/*.png` |
| 4 | `find_max_error_timepoint.py` | Locate each model's maximum-error time point and convert it to a date (console only) | - |
| 5 | `step1_*/test_*_transfer_learning.py` | Per-step transfer learning (EWC), saving the weights and predictions of every step | `model_after_transfer_step_*.pth`, `prediction.csv` |
| 6 | `results2_trans_test_Bollinger.py` | Three-model comparison in the transfer stage | `results/model_transfer_performance_comparison.png` |
| 7 | `results3_geneEmbedding.py` | Per-dimension time series of the gene fusion vectors | `results/{fusion}_time_series.png`, `combined_fusion_vectors.png` |
| 8 | `results4_trans_shaply.py` | Per-step gene SHAP computation | `results/Shapely for trans/step_*/` |
| 8 | `results4_trans_PDP.py` | Per-step gene PDP computation (mirrors the SHAP version) | `results/PDP for trans/step_*/` |
| 9 | `results5_SHAP_time_window_stats.py` | Average SHAP over the three time windows -> IMP table | `results/IMP/*.xlsx` |
| 9 | `results5_PDP_time_window_stats.py` | Average PDP over the three time windows -> LEF table | `results/PDP/*.xlsx` |
| 10 | `results5_trans_SHAPPDP_heatmap.py` | Heatmap of SHAP/PDP evolution over transfer steps + cosine similarity | `results/shap_pdp_trans_heatmap_raw.png` |
| 11 | `results6_trans_IMP_PDP_barplot.py` | IMP / PDP / MT horizontal bar comparison (two windows x two zones) | `results/imp_pdp_barplot_comparison.png` |
| 12 | `results7_top10gene.py` | Top10 genes per Period x zone x index, plus the genes shared by all three indices | `results/top10gene.xlsx` |
| 13 | `results8_PDP.py` | PDP composite figure (heatmaps + cosine curve + bar charts) | `results/PDP_analysis_composite.png` |
| 13 | `results9_SHAP.py` | SHAP composite figure (heatmaps + IMP bars + MT bars + Venn diagrams) | `results/SHAP_analysis_composite.png`, `results/top10genev2.xlsx` |
| 14 | `results10_FFT.py` | Frequency-domain error analysis of the ablation models (FFT) + Bollinger bands + EFF ammonia comparison | `results/results10_FFT.png` |
| 14 | `results10_FFT_pred.py` | All 12 output dimensions of the ablation models | `results/results10_FFT_pred.png` |

Every script header documents its **Purpose / Input / Output / Usage**.

---

## 5. Full workflow

### 5.1 Data preparation

Make sure the following files exist:
- `data_WQ/wq.xlsx`, `data_WQ/wq_scaled.csv`, `data_WQ/wq_scaleddict.npy`
- `data_Gene/n_gene_dynamic_common.xlsx`, `data_Gene/c_gene_dynamic_common.xlsx`
- `data_GeneM/n_gene_dynamic.xlsx` (MT dynamic data, used by results6/9)
- `gene_ref/Gene_info.xlsx` (KEGG -> gene name mapping, in the root-level `gene_ref/` folder)

### 5.2 Train the main model

```bash
cd step1_gene_model_trans_2block_FullTL
python train_gene_model.py                     # source-domain training
python test_gene_model.py                      # source-domain test (writes prediction.csv)
python test_gene_model_transfer_learning.py    # per-step transfer learning
```

### 5.3 Train the baselines / ablations (optional, for comparison)

```bash
cd ../step1_LSTM_model_trans
python train_LSTM_model.py
python test_LSTM_model.py
python test_LSTM_model_transfer_learning.py
```

The ablation models must first generate `prediction.csv` by running
`test_gene_model_transfer_learning.py` in their own folders before
`results10_FFT*.py` can read them.

### 5.4 Sensitivity analysis (SHAP / PDP)

```bash
cd ..                                  # back to the project root
python results4_trans_shaply.py        # per-step SHAP (heavy, a GPU is recommended)
python results4_trans_PDP.py           # per-step PDP
python results5_SHAP_time_window_stats.py   # -> results/IMP/
python results5_PDP_time_window_stats.py    # -> results/PDP/
```

### 5.5 Visualization

```bash
python results5_trans_SHAPPDP_heatmap.py
python results6_trans_IMP_PDP_barplot.py
python results7_top10gene.py
python results8_PDP.py
python results9_SHAP.py
```

### 5.6 Error analysis

```bash
python results1_training_test_Bollinger.py
python results2_trans_test_Bollinger.py
python results10_FFT.py
python results10_FFT_pred.py
```

---

## 6. Key parameters

| Parameter | Location | Value | Description |
|-----------|----------|-------|-------------|
| `WINDOW_SIZE` | gene_model.py / results scripts | 40 | Time window length (steps, 15 min each, about 10 hours) |
| `N_GENE_DIM` | gene_model.py | 21 | n_gene input dimension |
| `C_GENE_DIM` | gene_model.py | 1205 | c_gene input dimension |
| `hidden_dim` | train/test scripts | 64 | Hidden size (must match the weight file) |
| `batch_size` | train_gene_model.py | 256 | Training batch size |
| `EWC_LAMBDA` | test_gene_model_transfer_learning.py | 1000 | EWC regularization strength; larger means less forgetting |
| `transfer_window` / `transfer_epochs` | same as above | 40 / 30 | Data window and fine-tuning epochs per transfer step |
| `window_size` / `num_std` | results1/2 Bollinger | 40 / 3 | Rolling window length and band width multiplier |
| `VENN_RX/RY/DX` | results9_SHAP.py | 0.42/0.30/0.22 | Ellipse size and spacing of the hand-drawn Venn diagrams |

---

## 7. Notes and FAQ

1. **Root-level scripts must run from the project root** (`cd AI-Bio && python results9_SHAP.py`).
   Scripts inside the step1 folders must be run from their own directory, because they read
   data through `../data_*`.
2. **OpenMP duplicate-library error**: all scripts already set `KMP_DUPLICATE_LIB_OK=TRUE`.
   If the error persists, also set `OMP_NUM_THREADS=1`.
3. **`hidden_dim` must match the checkpoint**: `test_gene_model.py` defaults to
   `hidden_dim=256`, while the saved `best_gene_model.pth` actually uses `hidden_dim=64`.
   If loading fails, follow the dimension stored in the checkpoint (the transfer-learning
   script already uses 64).
4. **SHAP / PDP are expensive**: computing every transfer step (about 100 checkpoints) takes a
   long time. A GPU is recommended, or thin out the steps with the `interval` argument of
   `discover_models`.
5. **Time alignment rule**: `step_N` maps to row `N + WINDOW_SIZE - 1` of `wq.xlsx`. The
   `time_step` values of all three models have been unified to an absolute index so they can be
   compared directly.
6. **Figure output**: all figures are saved at 200-300 dpi under `results/` (and its
   subfolders), with `bbox_inches='tight'` to remove the extra margins.

---

## 8. Tech stack

| Category | Technology |
|----------|------------|
| Language | Python >= 3.9 |
| Deep learning | PyTorch >= 2.0 |
| Data processing | pandas, numpy, openpyxl, xlrd |
| Visualization | matplotlib |
| Numerics / statistics | scipy |
| Model interpretation | SHAP >= 0.42 (KernelSHAP), PDP (custom implementation) |
| Incremental learning | EWC (elastic weight consolidation, custom implementation) |

---

*Version: 3.0.0*
*Last updated: 2026-11-05*
