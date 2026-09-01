import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import os
import pandas as pd
from scipy import stats

# Set environment variable to suppress OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 字体设置
config = {
            "font.family": 'serif',
            "font.size": 12,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Arial'],#['Times New Roman'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)
label_size = 18

# Define model paths
model_paths = {
    'Gene': 'step1_gene_model_trans_2block/model/gene_model/training_history_with_lr.npy',
    'LSTM': 'step1_LSTM_model_trans/model/LSTM_model/training_history_with_lr.npy',
    'MLP': 'step1_MLP_model_trans/model/MLP_model/training_history_with_lr.npy'
}

# Define prediction paths
prediction_paths = {
    'Gene': 'step1_gene_model_trans_2block/results/gene_model/prediction.csv',
    'LSTM': 'step1_LSTM_model_trans/results/LSTM_model/prediction.csv',
    'MLP': 'step1_MLP_model_trans/results/MLP_model/prediction.csv'
}

# Load training histories
training_histories = {}
for model_name, path in model_paths.items():
    if os.path.exists(path):
        history = np.load(path, allow_pickle=True).item()
        training_histories[model_name] = history
        print(f"Loaded {model_name} training history")
        print(f"  Keys: {list(history.keys())}")
    else:
        print(f"Warning: {model_name} training history not found at {path}")

# Load prediction data and calculate MSE
mse_data = {}
for model_name, path in prediction_paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # Extract real and predicted values
        # Note: Gene(2block) uses '_true_' suffix, LSTM/MLP use '_real_' suffix
        real_cols = [col for col in df.columns if ('true' in col.lower() or 'real' in col.lower())]
        pred_cols = [col for col in df.columns if 'pred' in col.lower()]        
        real_values = df[real_cols].values
        pred_values = df[pred_cols].values
        
        # Calculate MSE for each sample
        sample_mse = np.mean((real_values - pred_values) ** 2, axis=1)
        mse_data[model_name] = sample_mse
        
        print(f"\n{model_name} Model Prediction Statistics:")
        print(f"  Number of samples: {len(sample_mse)}")
        print(f"  Mean MSE: {np.mean(sample_mse):.6f}")
        print(f"  Std MSE: {np.std(sample_mse):.6f}")
        print(f"  Min MSE: {np.min(sample_mse):.6f}")
        print(f"  Max MSE: {np.max(sample_mse):.6f}")
    else:
        print(f"Warning: {model_name} prediction file not found at {path}")

# Create results directory
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Create figure with GridSpec
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# Define colors for each model
model_colors = {
    'Gene': {
        'train': '#1f77b4',  # 深蓝色
        'val': '#aec7e8',      # 浅蓝色
        'hist': '#6baed6',     # 柱状图颜色
        'bb_fill': '#aec7e8',  # Bollinger Bands填充颜色
        'bb_line': '#1f77b4'   # Bollinger Bands中线颜色
    },
    'LSTM': {
        'train': '#d62728',  # 深粉红色
        'val': '#ff9896',      # 浅粉红色
        'hist': '#fd8d3c',     # 柱状图颜色
        'bb_fill': '#ff9896',  # Bollinger Bands填充颜色
        'bb_line': '#d62728'   # Bollinger Bands中线颜色
    },
    'MLP': {
        'train': '#2ca02c',   # 深绿色
        'val': '#98df8a',      # 浅绿色
        'hist': '#74c476',     # 柱状图颜色
        'bb_fill': '#98df8a',  # Bollinger Bands填充颜色
        'bb_line': '#2ca02c'   # Bollinger Bands中线颜色
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
    # Note: Gene(2block) uses '_true_' suffix, LSTM/MLP use '_real_' suffix
    real_cols = [col for col in df.columns if ('true' in col.lower() or 'real' in col.lower())]
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

# Calculate absolute errors and Bollinger Bands for each model
model_absolute_errors = {}
model_bollinger_bands = {}
model_dates = {}

# Read actual dates from wq.xlsx
import pandas as pd
from datetime import datetime, timedelta

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
                if step < len(actual_dates):
                    model_dates.append(actual_dates[step])
                else:
                    # If step is out of range, use the last date + delta
                    if model_dates:
                        last_date = model_dates[-1]
                        model_dates.append(last_date + timedelta(minutes=15))
                    else:
                        # Fallback if no dates found
                        start = datetime.strptime('2025-05-30', '%Y-%m-%d')
                        model_dates.append(start + timedelta(days=step))
            
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

# First pass: calculate errors and generate dates for all models
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

# ★ Align time ranges across all models to ensure consistent x-axis
print("\n===== Aligning Time Ranges Across Models =====")
if len(model_dates) >= 2:
    # Find the common date range (intersection of all models' date ranges)
    all_start_dates = []
    all_end_dates = []
    
    for model_name, dates in model_dates.items():
        if dates and len(dates) > 0:
            # Convert to datetime for comparison
            try:
                start_date = pd.to_datetime(dates[0])
                end_date = pd.to_datetime(dates[-1])
                all_start_dates.append((model_name, start_date))
                all_end_dates.append((model_name, end_date))
                print(f"  {model_name}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({len(dates)} samples)")
            except Exception as e:
                print(f"  {model_name}: Error parsing dates - {e}")
    
    if all_start_dates and all_end_dates:
        # Use the latest start date and earliest end date (intersection)
        latest_start = max([d[1] for d in all_start_dates])
        earliest_end = min([d[1] for d in all_end_dates])
        
        print(f"\n  Common time range: {latest_start.strftime('%Y-%m-%d')} to {earliest_end.strftime('%Y-%m-%d')}")
        
        # Trim each model's data to the common range
        for model_name in list(model_bollinger_bands.keys()):
            bb_data = model_bollinger_bands[model_name]
            dates = bb_data['dates']
            
            # Find indices within the common range
            valid_indices = []
            for i, date in enumerate(dates):
                try:
                    dt = pd.to_datetime(date)
                    if latest_start <= dt <= earliest_end:
                        valid_indices.append(i)
                except:
                    pass
            
            if valid_indices:
                print(f"  {model_name}: Trimming from {len(dates)} to {len(valid_indices)} samples")
                
                # Trim all arrays to valid_indices
                model_bollinger_bands[model_name] = {
                    'upper': [bb_data['upper'][i] if i < len(bb_data['upper']) else np.nan for i in valid_indices],
                    'middle': [bb_data['middle'][i] if i < len(bb_data['middle']) else np.nan for i in valid_indices],
                    'lower': [bb_data['lower'][i] if i < len(bb_data['lower']) else np.nan for i in valid_indices],
                    'errors': [bb_data['errors'][i] if i < len(bb_data['errors']) else np.nan for i in valid_indices],
                    'dates': [dates[i] for i in valid_indices]
                }

# Plot each model
model_names = ['Gene', 'LSTM', 'MLP']
titles = ['WqGene', 'LSTM', 'MLP']

for idx, (model_name, title) in enumerate(zip(model_names, titles)):
    # Plot training history on the left (column 0)
    ax = fig.add_subplot(gs[idx, 0])
    
    if model_name in training_histories:
        history = training_histories[model_name]
        
        # Get loss data
        train_loss = history.get('train_loss', history.get('train_losses', []))
        val_loss = history.get('val_loss', history.get('val_losses', []))
        
        # Create epoch array
        epochs = range(1, len(train_loss) + 1)
        
        # Plot training loss (darker color)
        ax.plot(epochs, train_loss, color=model_colors[model_name]['train'], 
                label=f'Training Loss: {train_loss[-1]:.3f}', linewidth=1.5)
        
        # Plot validation loss (lighter color)
        ax.plot(epochs, val_loss, color=model_colors[model_name]['val'], 
                label=f'Validation Loss: {val_loss[-1]:.3f}', linewidth=1.5)
        
        # Set labels and title
        if idx == len(model_names) - 1:  # Only show xlabel on the last plot
            ax.set_xlabel('Epoch', fontsize=label_size )
        ax.set_ylabel('Loss', fontsize=label_size )
        ax.set_title(title, fontsize=label_size , fontweight='bold')
        ax.legend(loc='upper right', fontsize=label_size )
        # Add (a) label to top-left corner only for first model
        if idx == 0:
            ax.text(-0.1, 1.1, '(a)', transform=ax.transAxes, 
                    fontsize=label_size, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='left')
        ax.grid(True, alpha=0.15)
        
        # Set y-axis to log scale if losses vary widely
        if len(train_loss) > 0 and len(val_loss) > 0:
            min_loss = min(min(train_loss), min(val_loss))
            max_loss = max(max(train_loss), max(val_loss))
            if max_loss / min_loss > 100:
                ax.set_yscale('log')
        
        # Print statistics
        print(f"\n{title} Statistics:")
        print(f"  Total epochs: {len(train_loss)}")
        print(f"  Final train loss: {train_loss[-1]:.6f}")
        print(f"  Final val loss: {val_loss[-1]:.6f}")
        print(f"  Best val loss: {min(val_loss):.6f}")
        print(f"  Best epoch: {np.argmin(val_loss) + 1}")
    else:
        ax.text(0.5, 0.5, f'{title}\nTraining history not found', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.15)
    
    # Plot MSE probability density on the right (column 1)
    ax_hist = fig.add_subplot(gs[idx, 1])
    
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
        ax_hist.plot(x, kde(x), color=model_colors[model_name]['train'], 
                     linewidth=2)
        
        # Set labels, title and x-axis limit
        ax_hist.set_xlabel('MSE', fontsize=label_size )
        ax_hist.set_xlim(0, 0.06)
        if idx == len(model_names) - 1:  # Only show xlabel on the last plot
            ax_hist.set_ylabel('Probability Density', fontsize=label_size )
        ax_hist.set_title(f'{title} MSE Distribution (0-0.06)', fontsize=label_size , fontweight='bold')
        # Add (b) label to top-left corner only for first model
        if idx == 0:
            ax_hist.text(-0.15, 1.1, '(b)', transform=ax_hist.transAxes, 
                         fontsize=label_size, fontweight='bold', 
                         verticalalignment='top', horizontalalignment='left')
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
        if idx == len(model_names) - 1:  # Only show xlabel on the last plot
            ax_hist.set_ylabel('Probability Density', fontsize=14)
        ax_hist.set_title(f'{title} MSE Distribution', fontsize=13, fontweight='bold')
        ax_hist.grid(True, alpha=0.15)
    
    # Plot Bollinger Bands on the right (column 2)
    ax_bb = fig.add_subplot(gs[idx, 2])
    
    # Add label (c) to the first Bollinger Bands plot
    if idx == 0:
        ax_bb.text(-0.1, 1.0, '(c)', transform=ax_bb.transAxes, 
                  fontsize=label_size, fontweight='bold', ha='right', va='bottom')
    
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
                value = middle_band[i]
                if not np.isnan(value):
                    target_date_points.append((i, value))
        
        # Find the highest point on the target date
        if target_date_points:
            target_date_points.sort(key=lambda x: x[1], reverse=True)
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
        ax_bb.set_ylim(bottom=0, top=0.6)
        
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
output_path = os.path.join(results_dir, 'model_performance_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n[SUCC] Model performance comparison saved: {output_path}")

print("\n[SUCC] All models processed successfully!")
