"""
results4_trans_PDP.py - gene PDP (partial dependence) sensitivity analysis of the transfer-learned models
=========================================================
Purpose:
    compute the SHAP values of the gene input dims for each checkpoint saved during transfer learning
    PDP (Partial Dependence Plot) sensitivity, tracking how gene contributions evolve along the transfer steps:
    1) Extract the gene projection sub-model (GeneProjModel) from the full GeneModel
    2) Perturb each gene dimension over a grid and average the output change -> PDP curve
    3) Each checkpoint writes a PDP figure and a feature importance CSV
    4) Reusable by results5_trans_SHAPPDP_heatmap.py / results6_trans_IMP_PDP_barplot.py
Input:
    .pth checkpoints under step1_gene_model_trans_2block_FullTL/model/...
    gene_ref/Gene_info.xlsx (KEGG ID -> gene name mapping)
Output:
    results/PDP for trans/step_{N}/PDP_*.png
    results/PDP for trans/step_{N}/PDP_feature_importance_{model}.csv
Usage:
    python results4_trans_PDP.py            # run from the project root (a GPU is recommended)
Note:
    results4_trans_shaply.py is the SHAP version of the same workflow; both produce interchangeable outputs
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import sys
import gc
import re
from itertools import product
from tqdm import tqdm

# Set environment variable to avoid OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Add step1_gene_model_trans_2block directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'step1_gene_model_trans_2block_FullTL'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'step1_gene_model_trans_2block_FullTL'))

from gene_model import GeneModel


# ============================================================
# Core functions - PDP Analysis (replacing SHAP)
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

    # 2block model: ana_ano merged (6-dim output), ox (3-dim output)
    gene_n_dim = 21
    hidden_dim = 64
    
    models = {
        'ana_ano': GeneProjModel(gene_n_dim, hidden_dim, 6).to(device),  # ANA+ANO merged, 6-dim output
        'ox': GeneProjModel(gene_n_dim, hidden_dim, 3).to(device)
    }

    # Copy the weights of the trained gene model for ana_ano (2block layer names)
    ana_ano_state_dict = models['ana_ano'].state_dict()
    gene_ana_ano_state_dict = gene_model.gene_n_proj_ana_ano.state_dict()
    for key in ana_ano_state_dict.keys():
        if key in gene_ana_ano_state_dict:
            ana_ano_state_dict[key] = gene_ana_ano_state_dict[key]
    models['ana_ano'].load_state_dict(ana_ano_state_dict)

    # Copy the weights for ox (layer names unchanged)
    ox_state_dict = models['ox'].state_dict()
    gene_ox_state_dict = gene_model.gene_n_proj_ox.state_dict()
    for key in ox_state_dict.keys():
        if key in gene_ox_state_dict:
            ox_state_dict[key] = gene_ox_state_dict[key]
    models['ox'].load_state_dict(ox_state_dict)

    return models


def compute_pdp(model, feature_idx, grid_points=50, n_samples=200, device='cpu', output_dim=6):
    """
    Compute Partial Dependence Plot (PDP) for a single feature.
    
    Args:
        model: PyTorch model
        feature_idx: Index of the feature to analyze
        grid_points: Number of points in the feature grid
        n_samples: Number of background samples for marginalization
        device: Torch device
        output_dim: Output dimension (6 for ana_ano, 3 for ox)
    
    Returns:
        feature_values: Array of feature values
        pdp_values: Array of averaged predictions at each feature value
    """
    model.eval()
    
    # Generate feature value grid (assuming normalized input range [-1, 1])
    feature_values = np.linspace(-1, 1, grid_points)
    
    # Generate background data for the other features (2block: the 21 n_gene features)
    np.random.seed(42)
    background_data = np.random.uniform(-1, 1, size=(n_samples, 21))
    
    pdp_values = []
    
    with torch.no_grad():
        for val in feature_values:
            # Replace feature_idx column with current value
            test_data = background_data.copy()
            test_data[:, feature_idx] = val
            
            # Get model predictions
            x_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
            outputs = model(x_tensor).cpu().numpy()  # Shape: (n_samples, output_dim)
            
            # Average over all samples and all output dimensions
            avg_output = np.mean(outputs)  # mean over all outputs
            pdp_values.append(avg_output)
    
    return feature_values, np.array(pdp_values)


def collect_pdp_results(models, device):
    """
    Collect PDP analysis results for all models.
    
    For each gene (feature), compute PDP and extract importance metric
    as the range of PDP values (max - min), indicating the feature's effect magnitude.
    """
    all_results = {}
    gene_names = get_actual_gene_names()
    num_features = 21
    grid_points = 50
    
    # 2block: output dimensions of the different models
    model_output_dims = {'ana_ano': 6, 'ox': 3}
    
    for model_name, model in models.items():
        output_dim = model_output_dims.get(model_name, 3)
        print(f"  Computing PDP for {model_name} ({num_features} features, output_dim={output_dim})...")
        
        feature_importances = []
        pdp_curves = {}  # Store full PDP curves for plotting
        
        for feat_idx in tqdm(range(num_features), desc=f"  {model_name} features"):
            feat_vals, pdp_vals = compute_pdp(model, feat_idx, grid_points=grid_points, device=device, output_dim=output_dim)
            
            # Importance metric: range of PDP values (effect magnitude)
            importance = np.max(pdp_vals) - np.min(pdp_vals)
            feature_importances.append(importance)
            
            # Store curve data
            pdp_curves[feat_idx] = {
                'feature_values': feat_vals,
                'pdp_values': pdp_vals,
                'importance': importance
            }
        
        all_results[model_name] = {
            'mean_feature_importance': np.array(feature_importances),
            'gene_names': gene_names,
            'pdp_curves': pdp_curves
        }
    
    return all_results


def get_actual_gene_names():
    """Get actual gene names from n_gene_dynamic_common.xlsx (21 genes matching model input)"""
    gene_file = os.path.join('data_Gene', 'n_gene_dynamic_common.xlsx')
    if os.path.exists(gene_file):
        try:
            df = pd.read_excel(gene_file, sheet_name='ana_ano')
            # The last column holds the KEGG description; extract the K-prefixed id
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


def plot_pdp_results(all_results, output_dir):
    """Plot PDP analysis results: feature importance bar charts and PDP curves"""
    
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
    
    model_names = ['ana_ano', 'ox']  # 2block: ANA+ANO merged
    model_labels = {
        'ana_ano': 'ANA+ANO (Anaerobic + Anoxic Tank)',  # 2block new label
        'ox': 'OX (Aerobic Tank)'
    }
    
    # ===== Figure 1: Feature Importance Bar Charts =====
    plt.figure(figsize=(15, 12))  # adjust the height for 2 rows
    
    for i, model_name in enumerate(model_names):
        plt.subplot(2, 1, i + 1)  # 3 rows reduced to 2
        results = all_results[model_name]
        mean_feature_importance = results['mean_feature_importance']
        gene_names = results['gene_names']
        
        sorted_indices = np.argsort(mean_feature_importance)[::-1]
        sorted_importance = mean_feature_importance[sorted_indices]
        sorted_gene_names = [gene_names[idx] for idx in sorted_indices]
        
        y_pos = np.arange(len(sorted_gene_names))
        plt.barh(y_pos, sorted_importance, color='#2ca02c', edgecolor='black', linewidth=1)
        
        ax = plt.gca()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_gene_names, fontsize=tick_label_size)
        ax.set_title(f'{model_labels[model_name]} - DO Feature Importance (PDP)',
                     fontsize=label_size, fontweight='bold')
        ax.set_xlabel('PDP Importance (Range)', fontsize=label_size)
        ax.set_ylabel('Gene', fontsize=label_size)
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.tick_params(axis='y', labelsize=tick_label_size)
        plt.tight_layout()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'combined_do_pdp_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Importance plot saved: {output_path}")
    plt.close()
    
    # ===== Figure 2: PDP Curves for Top Genes =====
    top_n = 7  # Show top 7 genes per model
    
    for model_name in model_names:
        results = all_results[model_name]
        gene_names = results['gene_names']
        pdp_curves = results['pdp_curves']
        
        # Sort by importance and get top N
        importances = [(idx, pdp_curves[idx]['importance']) for idx in range(len(gene_names))]
        importances.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in importances[:top_n]]
        
        fig, axes = plt.subplots(top_n, 1, figsize=(10, 4 * top_n))
        if top_n == 1:
            axes = [axes]
        
        for i, feat_idx in enumerate(top_indices):
            ax = axes[i]
            curve = pdp_curves[feat_idx]
            ax.plot(curve['feature_values'], curve['pdp_values'], 
                   'b-', linewidth=2, marker='o', markersize=3)
            ax.fill_between(curve['feature_values'], curve['pdp_values'], alpha=0.3)
            ax.set_xlabel('Feature Value', fontsize=12)
            ax.set_ylabel('Partial Dependence', fontsize=12)
            ax.set_title(f'{gene_names[feat_idx]} (Importance: {curve["importance"]:.4f})', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_labels[model_name]} - Top {top_n} Genes PDP Curves', 
                    fontsize=label_size, fontweight='bold', y=1.01)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'pdp_curves_{model_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  PDP curves saved: {output_path}")
        plt.close()


def save_pdp_csv(all_results, output_dir):
    """Save PDP feature importance to CSV files"""
    for model_name in ['ana_ano', 'ox']:  # 2block model names
        results = all_results[model_name]
        mean_feature_importance = results['mean_feature_importance']
        gene_names = results['gene_names']
        
        pdp_data = []
        for idx, gene_name in enumerate(gene_names):
            pdp_data.append({
                'gene': gene_name,
                'mean_PDP': mean_feature_importance[idx]
            })
        
        df = pd.DataFrame(pdp_data)
        df_sorted = df.sort_values(by='mean_PDP', ascending=False)
        
        csv_path = os.path.join(output_dir, f'PDP_feature_importance_{model_name}.csv')
        df_sorted.to_csv(csv_path, index=False)
        print(f"  CSV saved: {csv_path}")


# ============================================================
# Functions for batch transfer learning model processing
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
    Load a single transfer learning model, perform PDP analysis, and save results.
    
    Args:
        model_path: Path to .pth file
        step_num: Model step number
        device: Torch device
        base_output_dir: Base output directory (results/PDP for trans)
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

    # Perform PDP analysis
    print(f"  Running PDP analysis...")
    all_results = collect_pdp_results(models, device)
    print(f"  PDP analysis completed")

    # Save results
    save_pdp_csv(all_results, model_output_dir)
    plot_pdp_results(all_results, model_output_dir)

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
    """Main function for batch PDP analysis on transfer learning models"""

    # Configuration
    FIRST_STEP = 4264 #4314       # First model step number: in round one 4264
    INTERVAL = 100          # Select every Nth model
    
    # Paths
    model_dir = os.path.join(
        'step1_gene_model_trans_2block', 'model', 'gene_model_transfer_learning'  # 2block path
    )
    base_output_dir = os.path.join('results', 'PDP for trans')  # Changed from Shapely for trans

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
    print(f"BATCH PDP ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Total models:   {total}")
    print(f"  Succeeded:      {success_count}")
    print(f"  Failed:         {fail_count}")
    print(f"  Output dir:     {base_output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
