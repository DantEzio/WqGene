import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import os

# Set environment variable to avoid OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Import SHAP library
import shap

# Add step1_gene_model_trans directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'step1_gene_model_trans'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'step1_gene_model_trans'))

from gene_model import GeneModel

# Set font for Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Function to load trained gene model
def load_trained_gene_model():
    """Load trained gene model"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = GeneModel(hidden_dim=64, dropout_rate=0.3).to(device)
    
    # Load trained model
    model_dir = os.path.join('step1_gene_model_trans', 'model', 'gene_model')
    if not os.path.exists(model_dir):
        model_dir = os.path.join('..', 'step1_gene_model_trans', 'model', 'gene_model')
    
    model_path = os.path.join(model_dir, 'best_gene_model.pth')
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Trained model loaded successfully from: {model_path}")
            return model, device
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, device
    else:
        print(f"Model not found at: {model_path}")
        return None, device

# Function to extract gene projection models
def extract_gene_projection_models(gene_model, device):
    """Extract gene projection models from trained gene model"""
    # 统一的 Gene 投影模型: Linear(21→256) → ReLU → Drop → Linear(256→3) → ReLU → Drop
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
    
    # Create models (匹配新 gene_model.py 中 gene_n_proj_* 的结构: 21→256→3)
    gene_dim = 21
    hidden_dim = 256   # gene_n_proj 中间隐藏层维度
    output_dim = 3

    models = {
        'ana': GeneProjModel(gene_dim, hidden_dim, output_dim).to(device),
        'ano': GeneProjModel(gene_dim, hidden_dim, output_dim).to(device),
        'ox': GeneProjModel(gene_dim, hidden_dim, output_dim).to(device)
    }
    
    # Copy weights from trained gene model
    # Extract weights for gene_proj_ana
    ana_state_dict = models['ana'].state_dict()
    gene_ana_state_dict = gene_model.gene_n_proj_ana.state_dict()
    
    # Map weights
    for key in ana_state_dict.keys():
        if key in gene_ana_state_dict:
            ana_state_dict[key] = gene_ana_state_dict[key]
    models['ana'].load_state_dict(ana_state_dict)
    
    # Extract weights for gene_proj_ano
    ano_state_dict = models['ano'].state_dict()
    gene_ano_state_dict = gene_model.gene_n_proj_ano.state_dict()
    
    # Map weights
    for key in ano_state_dict.keys():
        if key in gene_ano_state_dict:
            ano_state_dict[key] = gene_ano_state_dict[key]
    models['ano'].load_state_dict(ano_state_dict)
    
    # Extract weights for gene_proj_ox (structure: 21→32→3, matches GeneProjModelOX)
    ox_state_dict = models['ox'].state_dict()
    gene_ox_state_dict = gene_model.gene_n_proj_ox.state_dict()
    
    # Map weights
    for key in ox_state_dict.keys():
        if key in gene_ox_state_dict:
            ox_state_dict[key] = gene_ox_state_dict[key]
    models['ox'].load_state_dict(ox_state_dict)
    
    print("Gene projection models extracted successfully")
    return models

# Function to create model wrapper for SHAP analysis
def create_model_wrapper(model, device):
    """Create model wrapper for SHAP analysis"""
    def model_wrapper(x):
        # x is (n_samples, n_features)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(x_tensor)
        return output.cpu().numpy()
    return model_wrapper

# Function to collect all SHAP analysis results with multiple runs
def collect_shap_results(models, device):
    """Collect SHAP analysis results for all models with 50 runs"""
    all_results = {}
    
    # Get actual gene names
    gene_names = get_actual_gene_names()
    
    # Number of runs for averaging
    num_runs = 50
    
    for model_name, model in models.items():
        print(f"\n=== Collecting SHAP analysis for {model_name} ===")
        print(f"Running {num_runs} iterations...")
        
        # Create model wrapper
        model_wrapper = create_model_wrapper(model, device)
        
        # Initialize storage for feature importance values across runs
        all_feature_importance = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...")
            
            # Generate background data with different random seed for each run
            np.random.seed(42 + run)
            
            # Create SHAP explainer with significantly more background samples
            background_data = np.random.uniform(-1, 1, size=(500, 21))
            explainer = shap.KernelExplainer(model_wrapper, background_data)
            
            # Generate significantly more test samples
            test_data = np.random.uniform(-1, 1, size=(200, 21))
            
            # Test model output for debugging
            test_output = model_wrapper(test_data)
            
            # Compute SHAP values with higher accuracy
            shap_values = explainer.shap_values(test_data, nsamples='auto')
            
            # Calculate feature importance (mean absolute SHAP values) for DO
            if isinstance(shap_values, list):
                # Case 1: List of SHAP values for each sample
                if len(shap_values) > 0:
                    # Get SHAP values for DO (first output)
                    if isinstance(shap_values[0], np.ndarray) and shap_values[0].ndim == 2:
                        # Each element is (n_features, n_outputs)
                        do_shap_values = np.array([s[:, 0] for s in shap_values])
                        # Calculate mean absolute SHAP values across samples
                        feature_importance = np.mean(np.abs(do_shap_values), axis=0)
                        all_feature_importance.append(feature_importance)
            else:
                # Case 2: Array of SHAP values
                if isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        # Shape is (n_samples, n_features, n_outputs)
                        # Get DO feature importance (first output)
                        do_shap_values = shap_values[:, :, 0]
                        # Calculate mean absolute SHAP values across samples
                        feature_importance = np.mean(np.abs(do_shap_values), axis=0)
                        all_feature_importance.append(feature_importance)
                    elif shap_values.ndim == 2:
                        # Shape is (n_samples, n_features) - single output
                        # Calculate mean absolute SHAP values across samples
                        feature_importance = np.mean(np.abs(shap_values), axis=0)
                        all_feature_importance.append(feature_importance)
        
        # Convert to numpy array
        all_feature_importance = np.array(all_feature_importance)
        
        # Calculate mean feature importance
        mean_feature_importance = np.mean(all_feature_importance, axis=0)
        
        # Store results
        all_results[model_name] = {
            'mean_feature_importance': mean_feature_importance,  # Mean feature importance
            'all_feature_importance': all_feature_importance,    # All feature importance values
            'gene_names': gene_names                           # Gene names
        }
        
        print(f"Completed {num_runs} iterations for {model_name}")
    
    return all_results

# Function to get actual gene names from n_genedata.csv
def get_actual_gene_names():
    """Get actual gene names from n_genedata.csv (21 genes matching model input)"""
    gene_file = os.path.join('data_Gene', 'bc data v2', 'n_genedata.csv')
    if os.path.exists(gene_file):
        try:
            df = pd.read_csv(gene_file)
            gene_names = df['KEGG'].tolist()
            # Model uses only first 21 genes (input_dim=21)
            gene_names = gene_names[:21]
            print(f"Successfully loaded {len(gene_names)} gene names from n_genedata.csv")
            return gene_names
        except Exception as e:
            print(f"Error reading gene names: {e}")
            return [f'gene_{i}' for i in range(21)]
    else:
        print(f"Gene file not found at: {gene_file}")
        return [f'gene_{i}' for i in range(21)]

# Function to plot SHAP analysis results
def plot_shap_results(all_results, output_dir):
    """Plot SHAP analysis results for all models"""
    # Set font and style parameters
    from matplotlib import rcParams
    
    # Font settings
    config = {
                "font.family": 'serif',
                "font.size": 12,
                "mathtext.fontset": 'stix',
                "font.serif": ['Arial'],
                'axes.unicode_minus': False
             }
    rcParams.update(config)
    label_size = 18
    tick_label_size = 12
    
    model_names = ['ana', 'ano', 'ox']
    # Use full names
    model_labels = {
        'ana': 'ANA (Anaerobic Tank)', 
        'ano': 'ANO (Anoxic Tank)', 
        'ox': 'OX (Aerobic Tank)'
    }
    
    # Only create combined plot for DO
    plt.figure(figsize=(15, 18))
    
    for i, model_name in enumerate(model_names):
        plt.subplot(3, 1, i+1)
        
        results = all_results[model_name]
        mean_feature_importance = results['mean_feature_importance']
        gene_names = results['gene_names']
        
        # Sort features by importance (descending order)
        sorted_indices = np.argsort(mean_feature_importance)[::-1]
        sorted_importance = mean_feature_importance[sorted_indices]
        sorted_gene_names = [gene_names[idx] for idx in sorted_indices]
        
        # Create custom bar plot
        y_pos = np.arange(len(sorted_gene_names))
        plt.barh(y_pos, sorted_importance, color='#1f77b4', edgecolor='black', linewidth=1)
        
        # Get the current axis
        ax = plt.gca()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_gene_names, fontsize=tick_label_size)
        ax.set_title(f'{model_labels[model_name]} - DO Feature Importance (SHAP)', 
                    fontsize=label_size, fontweight='bold')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=label_size)
        ax.set_ylabel('Gene', fontsize=label_size)
        
        # Adjust tick label size
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.tick_params(axis='y', labelsize=tick_label_size)
        
        # Adjust layout to prevent label overlap
        plt.tight_layout()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save combined DO plot
    output_path = os.path.join(output_dir, 'combined_do_shap_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined DO SHAP analysis plot saved: {output_path}")
    plt.close()

# Main function
def main():
    """Main function"""
    # Load trained gene model
    gene_model, device = load_trained_gene_model()
    if gene_model is None:
        print("Error: Failed to load gene model")
        return
    
    # Extract gene projection models
    models = extract_gene_projection_models(gene_model, device)
    
    # Debug: Check OX model weights and output
    print("\n=== Debug: Checking OX model ===")
    ox_model = models['ox']
    
    # Check model weights
    print("OX model weights summary:")
    total_weight_sum = 0
    for name, param in ox_model.named_parameters():
        if param.requires_grad:
            weight_sum = param.data.abs().sum().item()
            total_weight_sum += weight_sum
            print(f"  {name}: sum(abs(weights)) = {weight_sum:.6f}")
    print(f"  Total weight sum: {total_weight_sum:.6f}")
    
    # Test OX model output
    test_input = torch.randn(10, 21).to(device)
    with torch.no_grad():
        ox_output = ox_model(test_input)
    print(f"OX model output shape: {ox_output.shape}")
    print(f"OX model DO output range: [{ox_output[:, 0].min().item():.6f}, {ox_output[:, 0].max().item():.6f}]")
    print(f"OX model DO output std: {ox_output[:, 0].std().item():.6f}")
    
    # Create results directory
    results_dir = os.path.join('results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Collect SHAP analysis results for all models
    print("\n=== Collecting SHAP analysis results for all models ===")
    all_results = collect_shap_results(models, device)
    
    # Plot SHAP analysis results
    print("\n=== Plotting SHAP analysis results ===")
    plot_shap_results(all_results, results_dir)
    
    # Print SHAP analysis summary for DO
    print("\n=== DO Feature Importance Summary (SHAP) ===")
    for model_name in ['ana', 'ano', 'ox']:
        results = all_results[model_name]
        mean_feature_importance = results['mean_feature_importance']
        gene_names = results['gene_names']
        
        # Sort by mean feature importance
        sorted_indices = np.argsort(mean_feature_importance)[::-1]  # Descending order
        
        # Print top 5 most important genes
        print(f"\nTop 5 most important genes for {model_name} DO:")
        for i in range(5):
            if i < len(sorted_indices):
                idx = sorted_indices[i]
                print(f"  {i+1}. {gene_names[idx]}: Mean |SHAP|={mean_feature_importance[idx]:.6f}")
    
    # Save SHAP feature importance results to CSV files
    print("\n=== Saving SHAP feature importance results to CSV files ===")
    
    for model_name in ['ana', 'ano', 'ox']:
        results = all_results[model_name]
        mean_feature_importance = results['mean_feature_importance']
        gene_names = results['gene_names']
        
        # Create DataFrame with gene names and their SHAP importance
        shap_data = []
        for idx, gene_name in enumerate(gene_names):
            shap_data.append({
                'gene': gene_name,
                'mean_abs_SHAP': mean_feature_importance[idx]
            })
        
        df = pd.DataFrame(shap_data)
        
        # Sort by mean absolute SHAP value in descending order
        df_sorted = df.sort_values(by='mean_abs_SHAP', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(results_dir, f'SHAP_feature_importance_{model_name}.csv')
        df_sorted.to_csv(csv_path, index=False)
        print(f"{model_name} SHAP feature importance saved to: {csv_path}")
    
    print("\n✅ SHAP analysis completed!")

if __name__ == "__main__":
    main()
