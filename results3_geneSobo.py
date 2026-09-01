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

from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze import sobol

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
    
    # Debug: verify weights were loaded correctly
    for name in ['ana', 'ano', 'ox']:
        total_weight = sum(p.data.abs().sum().item() for p in models[name].parameters())
        print(f"  {name} model total abs(weight) sum: {total_weight:.6f}")
    
    return models

# Function to create model wrapper for sensitivity analysis
def create_model_wrapper(model, device):
    """Create model wrapper for sensitivity analysis"""
    def model_wrapper(x):
        # x is (n_samples, n_features)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        # Set model to evaluation mode to disable dropout
        model.eval()
        with torch.no_grad():
            output = model(x_tensor)
        return output.cpu().numpy()
    return model_wrapper

# Function to perform Sobol sensitivity analysis
def perform_sobol_analysis(model, model_name, device):
    """Perform Sobol sensitivity analysis on a model"""
    # Define problem for Sobol analysis
    problem = {
        'num_vars': 21,  # 21 gene features
        'names': [f'gene_{i}' for i in range(21)],
        'bounds': [(-1, 1) for _ in range(21)]  # Assuming normalized inputs
    }
    
    # Generate significantly more samples for better stability
    print(f"Generating samples for {model_name}...")
    param_values = sobol_sample(problem, 4096, calc_second_order=True)
    print(f"Generated {param_values.shape[0]} samples")
    
    # Create model wrapper
    model_wrapper = create_model_wrapper(model, device)
    
    # Run model
    print(f"Running model for {model_name}...")
    Y = model_wrapper(param_values)
    
    # For each output dimension
    sensitivity_results = {}
    for dim in range(3):
        print(f"Analyzing sensitivity for {model_name} output dimension {dim}...")
        Si = sobol.analyze(problem, Y[:, dim], calc_second_order=True)
        sensitivity_results[dim] = Si
    
    return sensitivity_results

# Function to collect all sensitivity results with multiple runs
def collect_sensitivity_results(models, device):
    """Collect sensitivity results for all models with 50 runs"""
    all_results = {}
    
    # Number of runs for averaging
    num_runs = 50
    
    for model_name, model in models.items():
        print(f"\n=== Collecting sensitivity analysis for {model_name} ===")
        print(f"Running {num_runs} iterations...")
        
        # Define problem for Sobol analysis
        problem = {
            'num_vars': 21,  # 21 gene features
            'names': [f'gene_{i}' for i in range(21)],
            'bounds': [(-1, 1) for _ in range(21)]  # Assuming normalized inputs
        }
        
        # Initialize storage for total sensitivity values across runs
        all_ST = []
        all_S1 = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...")
            
            try:
                # Generate samples with different random seed for each run
                # Use run-specific seed to ensure variety
                np.random.seed(42 + run)
                
                # Generate significantly more samples for better stability
                param_values = sobol_sample(problem, 4096, calc_second_order=True)
                
                # Create model wrapper
                model_wrapper = create_model_wrapper(model, device)
                
                # Run model
                Y = model_wrapper(param_values)
                
                # Check for NaN/Inf in output
                if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
                    print(f"    Warning: NaN/Inf detected in {model_name} output, skipping run {run+1}")
                    continue
                
                # Analyze DO sensitivity (first output dimension)
                Si = sobol.analyze(problem, Y[:, 0], calc_second_order=True)
                
                # Store results
                all_ST.append(Si['ST'])
                all_S1.append(Si['S1'])
            except Exception as e:
                print(f"    Error in {model_name} run {run+1}: {e}, skipping...")
                continue
        
        # Convert to numpy arrays
        all_ST = np.array(all_ST)
        all_S1 = np.array(all_S1)
        
        # Calculate mean sensitivity values
        mean_ST = np.mean(all_ST, axis=0)
        mean_S1 = np.mean(all_S1, axis=0)
        
        # Store mean results
        all_results[model_name] = {
            'ST': mean_ST,  # Mean total-order
            'S1': mean_S1,  # Mean first-order
            'all_ST': all_ST,  # All total-order results for reference
            'all_S1': all_S1   # All first-order results for reference
        }
        
        print(f"Completed {num_runs} iterations for {model_name}")
    
    return all_results

# Function to get actual gene names from n_genedata.csv
def get_actual_gene_names():
    """Get actual gene names from n_genedata.csv"""
    gene_file = os.path.join('data_Gene', 'n_genedata.csv')
    if os.path.exists(gene_file):
        try:
            df = pd.read_csv(gene_file)
            gene_names = df['KEGG'].tolist()
            print(f"Successfully loaded {len(gene_names)} gene names from n_genedata.csv")
            return gene_names
        except Exception as e:
            print(f"Error reading gene names: {e}")
            return [f'gene_{i}' for i in range(21)]
    else:
        print(f"Gene file not found at: {gene_file}")
        return [f'gene_{i}' for i in range(21)]

# Function to plot all sensitivity results in one figure
def plot_combined_sensitivity_results(all_results, output_dir):
    """Plot all sensitivity results in one 3x1 figure with only Total-order"""
    # Set font and style parameters to match results2
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
    label_size = 18  # Match label size from results2
    tick_label_size = 12  # Increased tick label size
    
    # Get actual gene names (use first 21 to match model input dimension)
    gene_names = get_actual_gene_names()[:21]
    
    # Create figure with 3 rows and 1 column (smaller canvas)
    fig = plt.figure(figsize=(12, 15))  # Smaller canvas
    
    # Use GridSpec for better layout control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 1, figure=fig, hspace=0.6)  # Increased spacing
    
    model_names = ['ana', 'ano', 'ox']
    # Use full names
    model_labels = {
        'ana': 'ANA (Anaerobic Tank)', 
        'ano': 'ANO (Anoxic Tank)', 
        'ox': 'OX (Aerobic Tank)'
    }  # Full English labels
    
    # Deep blue color scheme
    deep_blue = '#1f77b4'  # Total-order color
    
    for row, model_name in enumerate(model_names):
        Si = all_results[model_name]
        
        # Get total-order sensitivities
        ST = Si['ST']  # Total-order
        
        # Sort by total-order sensitivity
        sorted_indices = np.argsort(ST)[::-1]  # Descending order
        sorted_genes = [gene_names[i] for i in sorted_indices]
        sorted_ST = ST[sorted_indices]
        
        # Plot Total-order
        ax_total = fig.add_subplot(gs[row, 0])
        bars_total = ax_total.bar(range(21), sorted_ST, color=deep_blue, edgecolor='black', linewidth=1)
        ax_total.set_title(f'{model_labels[model_name]} - Total-order Sensitivity', 
                          fontsize=label_size, fontweight='bold')
        ax_total.set_ylabel('Sensitivity Index', fontsize=label_size)
        ax_total.grid(True, alpha=0.15)  # Match grid alpha from results2
        
        # Set x-ticks for all genes with adjusted positions
        ax_total.set_xticks(range(21))
        ax_total.set_xticklabels(sorted_genes, rotation=45, ha='center', fontsize=tick_label_size)
        
        # Adjust y-tick label size
        ax_total.tick_params(axis='y', labelsize=tick_label_size)
        
        # Adjust x-axis limits to center the bars
        ax_total.set_xlim(-0.5, 20.5)
        
        # Adjust layout to prevent label overlap
        plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'combined_do_sensitivity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined sensitivity plot saved: {output_path}")
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
    print(models.keys())
    # Create results directory
    results_dir = os.path.join('results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Collect sensitivity results for all models
    print("\n=== Collecting sensitivity results for all models ===")
    all_results = collect_sensitivity_results(models, device)
    
    # Plot combined sensitivity results
    print("\n=== Plotting combined sensitivity results ===")
    plot_combined_sensitivity_results(all_results, results_dir)
    
    # Print DO sensitivity analysis for each model
    print("\n=== DO Sensitivity Analysis Summary ===")
    # Get actual gene names (use first 21 to match model input dimension)
    actual_gene_names = get_actual_gene_names()[:21]
    
    for model_name in ['ana', 'ano', 'ox']:
        Si = all_results[model_name]
        S1 = Si['S1']  # First-order
        ST = Si['ST']  # Total-order
        
        # Sort by total-order sensitivity (mean values)
        sorted_indices = np.argsort(ST)[::-1]  # Descending order
        
        # Print top 5 most sensitive genes
        print(f"\nTop 5 most sensitive genes for {model_name} DO:")
        for i in range(5):
            if i < len(sorted_indices):
                idx = sorted_indices[i]
                gene_name = actual_gene_names[idx]
                print(f"  {i+1}. {gene_name}: Total={ST[idx]:.6f}, First={S1[idx]:.6f}")

    # Save sensitivity results to separate CSV files for each model
    print("\n=== Saving sensitivity results to CSV files ===")
    
    # Use only first 21 gene names to match model input dimension
    gene_names_21 = actual_gene_names[:21]
    
    for model_name in ['ana', 'ano', 'ox']:
        Si = all_results[model_name]
        S1 = Si['S1']  # First-order
        ST = Si['ST']  # Total-order
        
        # Get gene names and their sensitivity indices for this model
        model_sensitivity_data = []
        for idx, gene_name in enumerate(gene_names_21):
            model_sensitivity_data.append({
                'gene': gene_name,
                'total_sensitivity': ST[idx],
                'first_sensitivity': S1[idx]
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(model_sensitivity_data)
        
        # Sort by total sensitivity in descending order
        df_sorted = df.sort_values(by='total_sensitivity', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(results_dir, f'Sobol_sensitivity_{model_name}.csv')
        df_sorted.to_csv(csv_path, index=False)
        print(f"{model_name} sensitivity results saved to: {csv_path}")

    print("\n✅ Sobol sensitivity analysis completed!")

if __name__ == "__main__":
    main()
