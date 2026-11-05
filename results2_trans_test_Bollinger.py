"""
results2_trans_test_Bollinger.py - multi-model performance comparison in the transfer stage
=========================================================
Purpose:
    the same analysis framework as results1, but applied to the transfer-learning results:
    1) Error statistics of the three models (Gene / LSTM / MLP) on the transfer test set
    2) Adaptive Bollinger bands of the absolute error (rolling window of +/-num_std standard deviations)
    3) MSE distribution of each model (histogram + KDE) and mean/variance comparison
Input:
    prediction_paths:
        step1_gene_model_trans_2block_FullTL/results/
            gene_model_transfer_learning/prediction.csv
        step1_LSTM_model_trans/results/LSTM_model_transfer_learning/prediction.csv
        step1_MLP_model_trans/results/MLP_model_transfer_learning/prediction.csv
Output:
    results/model_transfer_performance_comparison.png
Usage:
    python results2_trans_test_Bollinger.py    # run from the project root
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import os
import pandas as pd
from scipy import stats

# Set environment variable to suppress OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Font setup
config = {
            "font.family": 'serif',
            "font.size": 12,# roughly 12pt body text
            "mathtext.fontset": 'stix',# math font used by matplotlib, close to Times New Roman
            "font.serif": ['Arial'],#['Times New Roman'],#'SimSun'
            'axes.unicode_minus': False # handle the minus sign
         }
rcParams.update(config)
label_size = 18

# Define transfer learning prediction paths
# Gene uses the transfer-learning result of the V2 comparison model (no gene embedding + gene reconstruction head)
prediction_paths = {
    'Gene': 'step1_gene_model_trans_2block_FullTL/results/gene_model_transfer_learning/prediction.csv',
    'LSTM': 'step1_LSTM_model_trans/results/LSTM_model_transfer_learning/prediction.csv',
    'MLP': 'step1_MLP_model_trans/results/MLP_model_transfer_learning/prediction.csv'
}

# Load prediction data and calculate MSE
mse_data = {}
for model_name, path in prediction_paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # Extract real and predicted values
        real_cols = [col for col in df.columns if 'real' in col.lower()]
        pred_cols = [col for col in df.columns if 'pred' in col.lower()]
        
        real_values = df[real_cols].values
        pred_values = df[pred_cols].values
        
        # Calculate MSE for each sample
        sample_mse = np.mean((real_values - pred_values) ** 2, axis=1)
        mse_data[model_name] = sample_mse
        
        print(f"\n{model_name} Model (Transfer Learning) Prediction Statistics:")
        print(f"  Number of samples: {len(sample_mse)}")
        print(f"  Mean MSE: {np.mean(sample_mse):.6f}")
        print(f"  Std MSE: {np.std(sample_mse):.6f}")
        print(f"  Min MSE: {np.min(sample_mse):.6f}")
        print(f"  Max MSE: {np.max(sample_mse):.6f}")
    else:
        print(f"Warning: {model_name} transfer learning prediction file not found at {path}")

# Create results directory
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Colors per model (blues, dark to light: Gene > LSTM > MLP)
model_colors = {
    'Gene': {
        'hist': '#3182bd',     # bar color
        'bb_fill': '#9ecae1',  # Bollinger band fill color
        'bb_line': '#08306b'   # Bollinger band middle-line color
    },
    'LSTM': {
        'hist': '#6baed6',     # bar color
        'bb_fill': '#c6dbef',  # Bollinger band fill color
        'bb_line': '#2171b5'   # Bollinger band middle-line color
    },
    'MLP': {
        'hist': '#c6dbef',     # bar color
        'bb_fill': '#deebf7',  # Bollinger band fill color
        'bb_line': '#6baed6'   # Bollinger band middle-line color
    }
}

# Function to calculate absolute errors from prediction file
def calculate_absolute_errors(prediction_path):
    """Calculate absolute errors from prediction file"""
    if not os.path.exists(prediction_path):
        print(f"Warning: Prediction file not found at {prediction_path}")
        return None
    
    df = pd.read_csv(prediction_path)
    
    # Extract real and predicted values
    real_cols = [col for col in df.columns if 'real' in col.lower()]
    pred_cols = [col for col in df.columns if 'pred' in col.lower()]
    
    real_values = df[real_cols].values
    pred_values = df[pred_cols].values
    
    # Calculate absolute error for each time step
    absolute_errors = np.mean(np.abs(pred_values - real_values), axis=1)
    
    return absolute_errors

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(errors, window_size=40, num_std=3):
    """Calculate Bollinger Bands for error series"""
    if len(errors) < window_size:
        print(f"Warning: Not enough data to calculate Bollinger Bands. Need at least {window_size} data points, got {len(errors)}")
        return None, None, None
    
    upper_band = []
    middle_band = []
    lower_band = []
    
    for i in range(len(errors)):
        if i < window_size - 1:
            # Not enough data for first few points
            upper_band.append(np.nan)
            middle_band.append(np.nan)
            lower_band.append(np.nan)
        else:
            # Use the last window_size errors
            window_errors = errors[i - window_size + 1 : i + 1]
            mean_error = np.mean(window_errors)
            std_error = np.std(window_errors)
            
            upper_band.append(mean_error + num_std * std_error)
            middle_band.append(mean_error)
            lower_band.append(mean_error - num_std * std_error)
    
    return upper_band, middle_band, lower_band

# Read actual dates from wq.xlsx
def get_actual_dates():
    """Read actual dates from wq.xlsx file"""
    try:
        # Read the Excel file
        df = pd.read_excel('data_WQ/wq.xlsx')
        # Extract the time column
        dates = df['time'].tolist()
        print(f"Successfully read {len(dates)} dates from wq.xlsx")
        return dates
    except Exception as e:
        print(f"Error reading wq.xlsx: {e}")
        return None

# Get actual dates
actual_dates = get_actual_dates()

from datetime import datetime, timedelta
def generate_date_sequence_for_model(model_name, path, num_points):
    """Generate date sequence based on actual dates from wq.xlsx"""
    if actual_dates is None:
        # Fallback to default date generation
        start = datetime.strptime('2025-05-30', '%Y-%m-%d')
        return [start + timedelta(days=i) for i in range(num_points)]
    
    try:
        # Read the prediction file to get time_steps
        df_pred = pd.read_csv(path)
        if 'time_step' in df_pred.columns:
            # Get the time_steps from prediction file
            time_steps = df_pred['time_step'].tolist()
            print(f"Found {len(time_steps)} time_steps in {model_name} prediction file")
            
            # Map time_steps to actual dates
            model_dates = []
            for step in time_steps:
                # Adjust time_step to match results1's starting point (4263)
                adjusted_step = step + 4263
                if adjusted_step < len(actual_dates):
                    model_dates.append(actual_dates[adjusted_step])
                else:
                    # If step is out of range, use the last date + delta
                    if model_dates:
                        last_date = model_dates[-1]
                        model_dates.append(last_date + timedelta(minutes=15))
                    else:
                        # Fallback if no dates found
                        start = datetime.strptime('2025-05-30', '%Y-%m-%d')
                        model_dates.append(start + timedelta(days=adjusted_step))
            
            print(f"Generated {len(model_dates)} dates for {model_name} model")
            return model_dates
        else:
            # Fallback if no time_step column
            start = datetime.strptime('2025-05-30', '%Y-%m-%d')
            return [start + timedelta(days=i) for i in range(num_points)]
    except Exception as e:
        print(f"Error generating dates for {model_name}: {e}")
        # Fallback
        start = datetime.strptime('2025-05-30', '%Y-%m-%d')
        return [start + timedelta(days=i) for i in range(num_points)]

# Calculate absolute errors and Bollinger Bands for each model
model_absolute_errors = {}
model_bollinger_bands = {}
model_dates = {}

for model_name, path in prediction_paths.items():
    errors = calculate_absolute_errors(path)
    if errors is not None:
        model_absolute_errors[model_name] = errors
        print(f"Calculated {model_name} model absolute errors: {len(errors)} data points")
        
        # Generate date sequence based on actual dates
        dates = generate_date_sequence_for_model(model_name, path, len(errors))
        model_dates[model_name] = dates
        
        # Calculate Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(errors)
        if upper is not None:
            model_bollinger_bands[model_name] = {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'errors': errors,
                'dates': dates
            }
            print(f"Calculated Bollinger Bands for {model_name} model")
        else:
            print(f"Failed to calculate Bollinger Bands for {model_name} model")
    else:
        print(f"Failed to calculate {model_name} model absolute errors")

# Create figure with GridSpec (3 rows, 2 columns)
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

# Plot each model
model_names = ['Gene', 'LSTM', 'MLP']
titles = ['WqGene-LL', 'LSTM-LL', 'MLP-LL']

for idx, (model_name, title) in enumerate(zip(model_names, titles)):
    # Plot MSE probability density on the left (column 0)
    ax_hist = fig.add_subplot(gs[idx, 0])
    
    # Add label (a) to the first MSE plot
    if idx == 0:
        ax_hist.text(-0.1, 1.1, '(a)', transform=ax_hist.transAxes, 
                     fontsize=label_size, fontweight='bold', 
                     verticalalignment='top', horizontalalignment='left')
    
    if model_name in mse_data:
        mse_values = mse_data[model_name]
        
        # Filter MSE values to 0-0.06 range
        filtered_mse = mse_values[(mse_values >= 0) & (mse_values <= 0.06)]
        
        # Create histogram with probability density
        n, bins, patches = ax_hist.hist(filtered_mse, bins=60, density=True, 
                                       color=model_colors[model_name]['hist'], 
                                       alpha=0.7, edgecolor='black', linewidth=1)
        
        # Calculate and plot kernel density estimate
        kde = stats.gaussian_kde(filtered_mse)
        x = np.linspace(0, 0.06, 100)
        ax_hist.plot(x, kde(x), color=model_colors[model_name]['bb_line'], 
                     linewidth=2)
        
        # Set labels, title and x-axis limit
        ax_hist.set_xlabel('MSE', fontsize=label_size)
        ax_hist.set_ylabel('Probability Density', fontsize=label_size)  # show the ylabel on every row
        ax_hist.set_xlim(0, 0.06)
        ax_hist.set_title(f'{title} MSE Distribution (0-0.06)', fontsize=label_size, fontweight='bold')
        
        # Add MSE statistics as text on the plot
        mean_mse = np.mean(filtered_mse)
        std_mse = np.std(filtered_mse)
        min_mse = np.min(filtered_mse)
        max_mse = np.max(filtered_mse)
        text_str = f'Mean: {mean_mse:.4f}\nStd: {std_mse:.4f}\nMin: {min_mse:.4f}\nMax: {max_mse:.4f}'
        ax_hist.text(0.95, 0.95, text_str, transform=ax_hist.transAxes, 
                     fontsize=label_size, verticalalignment='top', 
                     horizontalalignment='right', bbox=dict(boxstyle='round', 
                     facecolor='white', alpha=0.8))
        ax_hist.grid(True, alpha=0.15)
    else:
        ax_hist.text(0.5, 0.5, f'{title}\nPrediction data not found', 
                     ha='center', va='center', fontsize=12, transform=ax_hist.transAxes)
        ax_hist.set_xlabel('MSE', fontsize=14)
        ax_hist.set_ylabel('Probability Density', fontsize=14)  # show the ylabel on every row
        ax_hist.set_title(f'{title} MSE Distribution', fontsize=13, fontweight='bold')
        ax_hist.grid(True, alpha=0.15)
    
    # Plot Bollinger Bands on the right (column 1)
    ax_bb = fig.add_subplot(gs[idx, 1])
    
    # Add label (b) to the first Bollinger Bands plot
    if idx == 0:
        ax_bb.text(-0.1, 1.1, '(b)', transform=ax_bb.transAxes, 
                   fontsize=label_size, fontweight='bold', 
                   verticalalignment='top', horizontalalignment='left')
    
    if model_name in model_bollinger_bands:
        bb_data = model_bollinger_bands[model_name]
        upper_band = bb_data['upper']
        middle_band = bb_data['middle']
        lower_band = bb_data['lower']
        errors = bb_data['errors']
        dates = bb_data['dates']
        
        # Create time steps
        time_steps = range(len(errors))
        
        # Fill between upper and lower bands
        ax_bb.fill_between(time_steps, lower_band, upper_band, 
                          color=model_colors[model_name]['bb_fill'], 
                          alpha=0.7, label='Bands Range')
        
        # Plot middle band
        ax_bb.plot(time_steps, middle_band, 
                  color=model_colors[model_name]['bb_line'], 
                  linewidth=2, label='Middle Band')
        
        # Mark the highest point on 2025-08-02
        target_date = '2025-08-02'
        target_date_points = []
        
        # Find all points on the target date
        for i, date in enumerate(dates):
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%Y-%m-%d')
            else:
                try:
                    dt = pd.to_datetime(date)
                    date_str = dt.strftime('%Y-%m-%d')
                except:
                    date_str = str(date)
            
            if date_str == target_date and i < len(middle_band):
                # Get the value from middle band
                value = middle_band[i]
                if not np.isnan(value):
                    target_date_points.append((i, value))
        
        # Find the highest point on the target date
        if target_date_points:
            # Sort by value in descending order
            target_date_points.sort(key=lambda x: x[1], reverse=True)
            # Get the highest point
            highest_idx, highest_value = target_date_points[0]
            
            # Plot the marker
            ax_bb.plot(highest_idx, highest_value, 'o', 
                      color='red', markersize=8, alpha=0.8)
            
            # Add text annotation with date
            ax_bb.annotate(f'{target_date}', 
                           xy=(highest_idx, highest_value), 
                           xytext=(5, 5), 
                           textcoords='offset points', 
                           fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.5', 
                                    fc='white', alpha=0.7))
        
        # Set labels and title
        ax_bb.set_xlabel('Date', fontsize=label_size)
        ax_bb.set_ylabel('Absolute Error', fontsize=label_size)
        ax_bb.set_title(f'{title} Bollinger Bands', fontsize=label_size, fontweight='bold')
        ax_bb.legend(loc='upper right', fontsize=label_size)
        ax_bb.grid(True, alpha=0.15)
        
        # Set y-axis limit to avoid negative values
        ax_bb.set_ylim(bottom=0, top=0.25)
        
        # Set x-axis to show only 3 evenly spaced dates
        num_dates = len(dates)
        if num_dates > 0:
            # Calculate positions for 3 evenly spaced dates
            tick_positions = [i for i in range(0, num_dates, num_dates // 2)][:3]
            # Ensure the last date is included
            if tick_positions[-1] != num_dates - 1:
                tick_positions[-1] = num_dates - 1
            
            # Format dates for display
            tick_labels = []
            for i in tick_positions:
                if i < len(dates):
                    date = dates[i]
                    if hasattr(date, 'strftime'):
                        # If it's a datetime object
                        tick_labels.append(date.strftime('%Y-%m-%d'))
                    else:
                        # If it's a string or other type
                        try:
                            # Try to parse it as datetime
                            dt = pd.to_datetime(date)
                            tick_labels.append(dt.strftime('%Y-%m-%d'))
                        except:
                            # Fallback to string representation
                            tick_labels.append(str(date))
                else:
                    tick_labels.append('')
            
            # Set ticks without rotation
            ax_bb.set_xticks(tick_positions)
            ax_bb.set_xticklabels(tick_labels, ha='center')
    else:
        ax_bb.text(0.5, 0.5, f'{title}\nBollinger Bands data not found', 
                   ha='center', va='center', fontsize=12, transform=ax_bb.transAxes)
        ax_bb.set_xlabel('Date', fontsize=14)
        ax_bb.set_ylabel('Absolute Error', fontsize=14)
        ax_bb.set_title(f'{title} Bollinger Bands', fontsize=13, fontweight='bold')
        ax_bb.grid(True, alpha=0.15)

# Save figure
output_path = os.path.join(results_dir, 'model_transfer_performance_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n[SUCC] Model transfer performance comparison saved: {output_path}")

print("\n[SUCC] All models processed successfully!")
