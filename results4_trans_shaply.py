import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import gc
import re

# Set environment variable to avoid OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Import SHAP library
import shap

# Add step1_gene_model_trans_2block directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'step1_gene_model_trans_2block'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'step1_gene_model_trans_2block'))

from gene_model import GeneModel


# ============================================================
# Core functions (reused from results3_geneSharply.py)
# ============================================================

def extract_gene_projection_models(gene_model, device):
    """Extract gene projection models from trained 2block gene model"""
    class GeneProjModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

        def forward(self, x):
            return self.proj(x)

    # ★ 2block模型: ana_ano合并(输出6维), ox(输出3维)
    gene_n_dim = 21
    hidden_dim = 64
    
    models = {
        'ana_ano': GeneProjModel(gene_n_dim, hidden_dim, 6).to(device),  # ★ 合并ANA+ANO，输出6维
        'ox': GeneProjModel(gene_n_dim, hidden_dim, 3).to(device)
    }

    # Copy weights from trained gene model for ana_ano (★ 2block新层名)
    ana_ano_state_dict = models['ana_ano'].state_dict()
    gene_ana_ano_state_dict = gene_model.gene_n_proj_ana_ano.state_dict()
    for key in ana_ano_state_dict.keys():
        if key in gene_ana_ano_state_dict:
            ana_ano_state_dict[key] = gene_ana_ano_state_dict[key]
    models['ana_ano'].load_state_dict(ana_ano_state_dict)

    # Copy weights for ox (层名不变)
    ox_state_dict = models['ox'].state_dict()
    gene_ox_state_dict = gene_model.gene_n_proj_ox.state_dict()
    for key in ox_state_dict.keys():
        if key in gene_ox_state_dict:
            ox_state_dict[key] = gene_ox_state_dict[key]
    models['ox'].load_state_dict(ox_state_dict)

    return models


def create_model_wrapper(model, device):
    """Create model wrapper for SHAP analysis (★ 2block: 适配不同输出维度)"""
    def model_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(x_tensor)
        # ★ 返回所有输出的均值（适配ana_ano的6维和ox的3维）
        return np.mean(output.cpu().numpy(), axis=1, keepdims=True)  # Shape: (n_samples, 1)
    return model_wrapper


def collect_shap_results(models, device):
    """Collect SHAP analysis results for all models with 50 runs (★ 2block适配)"""
    all_results = {}
    gene_names = get_actual_gene_names()
    num_runs = 50

    for model_name, model in models.items():
        print(f"  Collecting SHAP for {model_name} ({num_runs} runs)...")

        model_wrapper = create_model_wrapper(model, device)
        all_feature_importance = []

        for run in range(num_runs):
            np.random.seed(42 + run)
            background_data = np.random.uniform(-1, 1, size=(200, 21))
            explainer = shap.KernelExplainer(model_wrapper, background_data)
            test_data = np.random.uniform(-1, 1, size=(100, 21))
            shap_values = explainer.shap_values(test_data, nsamples='auto')

            # ★ 2block: 简化处理，统一取绝对值均值
            if isinstance(shap_values, list):
                # 多输出情况
                for sv in shap_values:
                    if isinstance(sv, np.ndarray) and sv.ndim == 2:
                        feature_importance = np.mean(np.abs(sv), axis=0)
                        all_feature_importance.append(feature_importance)
                        break
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    feature_importance = np.mean(np.abs(shap_values[:, :, 0]), axis=0)
                    all_feature_importance.append(feature_importance)
                elif shap_values.ndim == 2:
                    feature_importance = np.mean(np.abs(shap_values), axis=0)
                    all_feature_importance.append(feature_importance)

        if len(all_feature_importance) > 0:
            all_feature_importance = np.array(all_feature_importance)
            mean_feature_importance = np.mean(all_feature_importance, axis=0)
        else:
            mean_feature_importance = np.zeros(21)

        all_results[model_name] = {
            'mean_feature_importance': mean_feature_importance,
            'gene_names': gene_names
        }

    return all_results


def get_actual_gene_names():
    """Get actual gene names from n_gene_dynamic_common.xlsx (21 genes matching model input)"""
    gene_file = os.path.join('data_Gene', 'n_gene_dynamic_common.xlsx')
    if os.path.exists(gene_file):
        try:
            df = pd.read_excel(gene_file, sheet_name='ana_ano')
            # 最后一列是 KEGG 描述，提取 K 开头编号
            kegg_ids = df.iloc[:, -1].astype(str).str.extract(r'(K\d+)')[0]
            kegg_ids = kegg_ids.dropna().tolist()
            gene_names = kegg_ids[:21]
            print(f"Loaded {len(gene_names)} gene names from {gene_file}")
            return gene_names
        except Exception as e:
            print(f"Error reading gene names: {e}")
            return [f'gene_{i}' for i in range(21)]
    else:
        print(f"Gene file not found at: {gene_file}")
        return [f'gene_{i}' for i in range(21)]


def plot_shap_results(all_results, output_dir):
    """Plot SHAP analysis results for all models"""
    from matplotlib import rcParams

    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['Arial'],
        "axes.unicode_minus": False
    }
    rcParams.update(config)
    label_size = 18
    tick_label_size = 12

    model_names = ['ana_ano', 'ox']  # ★ 2block: 合并ANA+ANO
    model_labels = {
        'ana_ano': 'ANA+ANO (Anaerobic + Anoxic Tank)',  # ★ 2block新标签
        'ox': 'OX (Aerobic Tank)'
    }

    plt.figure(figsize=(15, 12))  # ★ 调整高度适应2行

    for i, model_name in enumerate(model_names):
        plt.subplot(2, 1, i + 1)  # ★ 从3行改为2行
        results = all_results[model_name]
        mean_feature_importance = results['mean_feature_importance']
        gene_names = results['gene_names']

        sorted_indices = np.argsort(mean_feature_importance)[::-1]
        sorted_importance = mean_feature_importance[sorted_indices]
        sorted_gene_names = [gene_names[idx] for idx in sorted_indices]

        y_pos = np.arange(len(sorted_gene_names))
        plt.barh(y_pos, sorted_importance, color='#1f77b4', edgecolor='black', linewidth=1)

        ax = plt.gca()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_gene_names, fontsize=tick_label_size)
        ax.set_title(f'{model_labels[model_name]} - DO Feature Importance (SHAP)',
                     fontsize=label_size, fontweight='bold')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=label_size)
        ax.set_ylabel('Gene', fontsize=label_size)
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.tick_params(axis='y', labelsize=tick_label_size)
        plt.tight_layout()

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'combined_do_shap_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")
    plt.close()


def save_shap_csv(all_results, output_dir):
    """Save SHAP feature importance to CSV files"""
    for model_name in ['ana_ano', 'ox']:  # ★ 2block模型名
        results = all_results[model_name]
        mean_feature_importance = results['mean_feature_importance']
        gene_names = results['gene_names']

        shap_data = []
        for idx, gene_name in enumerate(gene_names):
            shap_data.append({
                'gene': gene_name,
                'mean_abs_SHAP': mean_feature_importance[idx]
            })

        df = pd.DataFrame(shap_data)
        df_sorted = df.sort_values(by='mean_abs_SHAP', ascending=False)

        csv_path = os.path.join(output_dir, f'SHAP_feature_importance_{model_name}.csv')
        df_sorted.to_csv(csv_path, index=False)
        print(f"  CSV saved: {csv_path}")


# ============================================================
# New functions for batch transfer learning model processing
# ============================================================

def discover_models(model_dir, first_step=4264, interval=100):
    """
    Scan model directory and filter models by step number.
    
    Args:
        model_dir: Path to gene_model_transfer_learning folder
        first_step: First model step number (default 4264)
        interval: Selection interval (default 100)
    
    Returns:
        List of (step_num, filepath) tuples sorted by step_num
    """
    pattern = re.compile(r'model_after_transfer_step_(\d+)\.pth')
    models = []

    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return models

    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            step_num = int(match.group(1))
            if step_num >= first_step and (step_num - first_step) % interval == 0:
                filepath = os.path.join(model_dir, filename)
                models.append((step_num, filepath))

    models.sort(key=lambda x: x[0])
    return models


def process_single_model(model_path, step_num, device, base_output_dir):
    """
    Load a single transfer learning model, perform SHAP analysis, and save results.
    
    Args:
        model_path: Path to .pth file
        step_num: Model step number
        device: Torch device
        base_output_dir: Base output directory (results/Shapely for trans)
    """
    print(f"\n{'='*60}")
    print(f"Processing model: step_{step_num}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")

    # Create output subdirectory for this model
    model_output_dir = os.path.join(base_output_dir, f'step_{step_num}')
    os.makedirs(model_output_dir, exist_ok=True)

    # Initialize and load model
    model = GeneModel(hidden_dim=64, dropout_rate=0.3).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Model loaded successfully")
    except Exception as e:
        print(f"  Error loading model: {e}")
        del model
        torch.cuda.empty_cache()
        return False

    # Extract projection models
    models = extract_gene_projection_models(model, device)
    print(f"  Projection models extracted")

    # Perform SHAP analysis
    print(f"  Running SHAP analysis...")
    all_results = collect_shap_results(models, device)
    print(f"  SHAP analysis completed")

    # Save results
    save_shap_csv(all_results, model_output_dir)
    plot_shap_results(all_results, model_output_dir)

    # Cleanup
    del model
    del models
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Done: step_{step_num}")
    return True


# ============================================================
# Main function
# ============================================================

def main():
    """Main function for batch SHAP analysis on transfer learning models"""

    # Configuration
    FIRST_STEP = 4314       # First model step number: in round one 4264
    INTERVAL = 100          # Select every Nth model
    
    # Paths
    model_dir = os.path.join(
        'step1_gene_model_trans_2block', 'model', 'gene_model_transfer_learning'  # ★ 2block路径
    )
    base_output_dir = os.path.join('results', 'Shapely for trans')

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Discover models
    print(f"\nScanning model directory: {model_dir}")
    print(f"Selection rule: from step_{FIRST_STEP}, every {INTERVAL} models")
    
    selected_models = discover_models(model_dir, FIRST_STEP, INTERVAL)
    total = len(selected_models)

    if total == 0:
        print("\nNo models found! Please check:")
        print(f"  1. Directory exists: {model_dir}")
        print(f"  2. Files match pattern: model_after_transfer_step_XXXXX.pth")
        print(f"  3. Step numbers >= {FIRST_STEP} and (num-{FIRST_STEP}) % {INTERVAL} == 0")
        return

    print(f"\nFound {total} models to process:")
    for idx, (step_num, _) in enumerate(selected_models):
        print(f"  [{idx + 1}/{total}] step_{step_num}")

    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)

    # Process each model
    success_count = 0
    fail_count = 0

    for idx, (step_num, model_path) in enumerate(selected_models):
        print(f"\n>>> Progress: [{idx + 1}/{total}] step_{step_num}")

        success = process_single_model(
            model_path, step_num, device, base_output_dir
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH SHAP ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Total models:   {total}")
    print(f"  Succeeded:      {success_count}")
    print(f"  Failed:         {fail_count}")
    print(f"  Output dir:     {base_output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
