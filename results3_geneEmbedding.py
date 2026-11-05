"""
results3_geneEmbedding.py - time series visualization of the gene fusion vectors (gene embedding)
=========================================================
Purpose:
    read the fusion-layer gene vectors saved per time step during transfer learning (fusion vector),
    plot the per-dimension curves over time steps to observe drift / periodicity of the gene representation:
    1) Plot the N-dimensional time series of each fusion vector (inf_ana / ana_ano / ano_ox) separately
    2) Plot all fusion vectors together in one comparison figure (one row per vector, one line per dim)
    3) Print the mean, std, min and max statistics of each dimension
Input:
    step1_gene_model_trans_2block_FullTL/results/gene_model_transfer_learning/
        fusion_inf_ana.npy / fusion_ana_ano.npy / fusion_ano_ox.npy
        (shape: [number of time steps, fusion dimension])
    data_WQ/wq_scaled.csv (only used to size the time axis; a warning is printed if missing)
Output:
    results/{fusion_name}_time_series.png   time series of each fusion vector
    results/combined_fusion_vectors.png     combined comparison figure
Usage:
    python results3_geneEmbedding.py         # run from the project root
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Set font for Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Function to get actual dates from wq.xlsx
def get_actual_dates():
    """Read dates from wq.xlsx"""
    try:
        wq_path = os.path.join('data_WQ', 'wq_scaled.csv')
        if not os.path.exists(wq_path):
            # Try relative path
            wq_path = os.path.join('..', 'data_WQ', 'wq_scaled.csv')
        
        if os.path.exists(wq_path):
            df = pd.read_csv(wq_path)
            # Generate date sequence if no date column
            start_date = datetime.strptime('2025-05-30', '%Y-%m-%d')
            dates = [start_date + timedelta(days=i) for i in range(len(df))]
            print(f"Successfully generated {len(dates)} dates")
            return dates
        else:
            print(f"Warning: wq_scaled.csv not found at {wq_path}")
            return None
    except Exception as e:
        print(f"Error reading dates: {e}")
        return None

# Function to analyze time series of gene embedding results
def analyze_gene_embedding_time_series():
    """Analyze time series of gene embedding results"""
    # Get actual dates
    actual_dates = get_actual_dates()
    
    # Define results directory
    results_dir = os.path.join('step1_gene_model_trans_2block_FullTL', 'results', 'gene_model_transfer_learning')
    if not os.path.exists(results_dir):
        results_dir = os.path.join('..', 'step1_gene_model_trans_2block_FullTL', 'results', 'gene_model_transfer_learning')
    
    print(f"Analyzing gene embedding results from: {results_dir}")
    
    # Check if fusion vector files exist
    fusion_files = {
        'fusion_inf_ana': os.path.join(results_dir, 'fusion_inf_ana.npy'),
        'fusion_ana_ano': os.path.join(results_dir, 'fusion_ana_ano.npy'),
        'fusion_ano_ox': os.path.join(results_dir, 'fusion_ano_ox.npy')
    }
    
    # Load fusion vectors
    fusion_vectors = {}
    for name, file_path in fusion_files.items():
        if os.path.exists(file_path):
            try:
                vectors = np.load(file_path)
                fusion_vectors[name] = vectors
                print(f"Loaded {name}: shape={vectors.shape}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"Warning: {name} not found at {file_path}")
    
    # Analyze each fusion vector
    for name, vectors in fusion_vectors.items():
        print(f"\n=== Analysis for {name} ===")
        print(f"Total time steps: {vectors.shape[0]}")
        print(f"Vector dimension: {vectors.shape[1]}")
        
        # Calculate statistics
        mean_values = np.mean(vectors, axis=0)
        std_values = np.std(vectors, axis=0)
        min_values = np.min(vectors, axis=0)
        max_values = np.max(vectors, axis=0)
        
        print(f"Mean values: {mean_values}")
        print(f"Std values: {std_values}")
        print(f"Min values: {min_values}")
        print(f"Max values: {max_values}")
        
        # Plot time series for each dimension
        plt.figure(figsize=(16, 12))
        for dim in range(vectors.shape[1]):
            plt.subplot(vectors.shape[1], 1, dim+1)
            plt.plot(range(vectors.shape[0]), vectors[:, dim])
            plt.title(f'{name} - Dimension {dim+1}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join('results', f'{name}_time_series.png')
        os.makedirs('results', exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved: {plot_path}")
        plt.close()
    
    # Create combined analysis
    if len(fusion_vectors) > 0:
        print("\n=== Combined Analysis ===")
        
        # Plot all fusion vectors together
        plt.figure(figsize=(16, 18))
        
        for idx, (name, vectors) in enumerate(fusion_vectors.items()):
            plt.subplot(len(fusion_vectors), 1, idx+1)
            for dim in range(vectors.shape[1]):
                plt.plot(range(vectors.shape[0]), vectors[:, dim], label=f'Dim {dim+1}')
            plt.title(name)
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        combined_plot_path = os.path.join('results', 'combined_fusion_vectors.png')
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved: {combined_plot_path}")
        plt.close()
    
    print("\nGene embedding time series analysis completed!")

if __name__ == "__main__":
    analyze_gene_embedding_time_series()
