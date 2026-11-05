"""
find_max_error_timepoint.py - locate the time point with the largest error for each model
=========================================================
Purpose:
    read the prediction.csv of the Gene / LSTM / MLP models and compute the per-time-step
    absolute error, find the time point with the largest error and convert time_step into a real date,
    makes it easy to look back at the operating conditions of that moment (influent shock, loading spike, etc.) and explain the failure.
    report per feature (each indicator of each region) and give the overall worst time point.
Input:
    step1_gene_model_trans_2block_FullTL/results/gene_model/prediction.csv
    step1_LSTM_model_trans/results/LSTM_model/prediction.csv
    step1_MLP_model_trans/results/MLP_model/prediction.csv
    data_WQ/wq.xlsx (time column, used to map time_step -> date)
Output:
    prints only the maximum-error time points and values to the console (no files are written)
Usage:
    python find_max_error_timepoint.py           # run from the project root
"""

import os
import numpy as np
import pandas as pd

# Suppress the OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================
# Configuration: paths of each model's prediction.csv
# ============================================================
PREDICTION_PATHS = {
    'Gene': 'step1_gene_model_trans_2block_FullTL/results/gene_model/prediction.csv',
    'LSTM':  'step1_LSTM_model_trans/results/LSTM_model/prediction.csv',
    'MLP':   'step1_MLP_model_trans/results/MLP_model/prediction.csv',
}

# Raw WQ date file (maps time_step to the actual date)
WQ_DATE_FILE = 'data_WQ/wq.xlsx'


def load_dates(wq_path):
    """Read the list of actual dates from wq.xlsx"""
    try:
        df = pd.read_excel(wq_path)
        dates = df['time'].tolist()
        print(f"Successfully read {len(dates)} date records")
        return dates
    except Exception as e:
        print(f"Warning: cannot read {wq_path}: {e}")
        return None


def load_prediction(path):
    """Load prediction.csv and return a DataFrame"""
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {path}: {len(df)} rows, {len(df.columns)} columns")
    return df


def calculate_errors(df):
    """
    compute the absolute error (MAE) of each time step.
    
    for each region (ANA/ANO/OX/EFF), compute the mean of |pred - real|,
    then average the errors over all regions to get the overall MAE of that time step.
    
    Returns:
        errors: (N,) array of the MAE at each time step
        region_errors: dict of the error of each region
    """
    real_cols = [c for c in df.columns if 'real' in c.lower()]
    pred_cols = [c for c in df.columns if 'pred' in c.lower()]
    
    real_vals = df[real_cols].values   # (N, 12)
    pred_vals = df[pred_cols].values   # (N, 12)
    
    # Overall MAE (averaged over all 12 features)
    total_mae = np.mean(np.abs(pred_vals - real_vals), axis=1)  # (N,)
    
    # Break the error down by region
    regions = ['ANA', 'ANO', 'OX', 'EFF']
    region_errors = {}
    for region in regions:
        r_cols = [c for c in real_cols if region in c]
        p_cols = [c for c in pred_cols if region in c]
        if r_cols and p_cols:
            region_mae = np.mean(np.abs(df[p_cols].values - df[r_cols].values), axis=1)
            region_errors[region] = region_mae
    
    return total_mae, region_errors


def find_max_error_timepoint(errors, dates=None, time_steps=None):
    """
    find the time point with the largest error.
    
    Args:
        errors: (N,) error array
        dates: list of actual dates (optional)
        time_steps: values of the time_step column (optional)
    
    Returns:
        dict: holds the index, date and error value
    """
    max_idx = int(np.argmax(errors))
    max_err = errors[max_idx]
    
    result = {
        'index': max_idx,
        'max_error': float(max_err),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'total_samples': len(errors),
    }
    
    # Try to obtain the dates
    if time_steps is not None and max_idx < len(time_steps):
        ts = time_steps[max_idx]
        result['time_step'] = int(ts)
        
        if dates is not None and ts < len(dates):
            result['date'] = dates[ts]
    
    return result


def main():
    print("=" * 60)
    print("Prediction error analysis of the Gene / LSTM / MLP models - find the maximum-error time points")
    print("=" * 60)
    
    # Load the date mapping
    dates = load_dates(WQ_DATE_FILE)
    print()
    
    # Store the results of each model
    all_results = {}
    
    for model_name, path in PREDICTION_PATHS.items():
        print(f"--- {model_name} Model ---")
        df = load_prediction(path)
        if df is None:
            continue
        
        # Compute the error
        total_mae, region_errors = calculate_errors(df)
        
        # Locate the time point of the maximum error
        time_steps = df['time_step'].values if 'time_step' in df.columns else None
        result = find_max_error_timepoint(total_mae, dates=dates, time_steps=time_steps)
        result['region_errors'] = region_errors
        result['all_errors'] = total_mae
        
        all_results[model_name] = result
        
        # Print a summary
        date_str = result.get('date', 'N/A')
        if hasattr(date_str, 'strftime'):
            date_str = date_str.strftime('%Y-%m-%d %H:%M')
        ts_str = result.get('time_step', 'N/A')
        
        print(f"  Sample count:       {result['total_samples']}")
        print(f"  Mean MAE:     {result['mean_error']:.6f}")
        print(f"  MAE std:   {result['std_error']:.6f}")
        print(f"  ** Max MAE:  {result['max_error']:.6f}**")
        print(f"  ** Time point:  time_step={ts_str}, date={date_str} **")
        
        # Per-region error at this time point
        print(f"  Per-region error at this time point:")
        for region, re_arr in region_errors.items():
            print(f"    {region}: {re_arr[result['index']]:.6f}")
        print()
    
    # ============================================================
    # Cross-model comparison: do the three models peak at the same time point?
    # ============================================================
    print("=" * 60)
    print("Cross-model comparison")
    print("=" * 60)
    
    summary_rows = []
    for model_name, res in all_results.items():
        date_str = res.get('date', 'N/A')
        if hasattr(date_str, 'strftime'):
            date_str = date_str.strftime('%Y-%m-%d')
        ts_str = res.get('time_step', 'N/A')
        summary_rows.append({
            'Model': model_name,
            'Max_MAE': f"{res['max_error']:.6f}",
            'Mean_MAE': f"{res['mean_error']:.6f}",
            'Time_Step': ts_str,
            'Date': str(date_str),
            'Index': res['index'],
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # Find the global maximum error (cross-model comparison)
    global_max_model = max(all_results.keys(), key=lambda m: all_results[m]['max_error'])
    global_max_result = all_results[global_max_model]
    
    print(f"\n>>> Model with the global maximum error: {global_max_model}")
    print(f">>> Maximum MAE value:     {global_max_result['max_error']:.6f}")
    print(f">>> Corresponding time point:   time_step={global_max_result.get('time_step', 'N/A')}")
    
    g_date = global_max_result.get('date', 'N/A')
    if hasattr(g_date, 'strftime'):
        g_date = g_date.strftime('%Y-%m-%d %H:%M')
    print(f">>> Corresponding date:       {g_date}")
    
    # Save the summary results to CSV
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the detailed summary
    detail_rows = []
    for model_name, res in all_results.items():
        row = {
            'Model': model_name,
            'Max_MAE': res['max_error'],
            'Mean_MAE': res['mean_error'],
            'Std_MAE': res['std_error'],
            'Time_Step': res.get('time_step', -1),
            'Sample_Index': res['index'],
        }
        # Add the per-region errors at this point
        for region, re_arr in res['region_errors'].items():
            row[f'{region}_MAE_at_max'] = re_arr[res['index']]
        detail_rows.append(row)
    
    detail_df = pd.DataFrame(detail_rows)
    csv_path = os.path.join(output_dir, 'max_error_timepoint_summary.csv')
    detail_df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")
    
    # Also save the full error series of each time step (for later plotting)
    error_series = {'time_step': []}
    for model_name, res in all_results.items():
        if 'time_step' in locals() or True:
            # Re-read time_step from the original df
            pass
    # Simpler approach: store the complete error series of each model directly
    for model_name, res in all_results.items():
        error_series[f'{model_name}_MAE'] = res['all_errors']
    
    # Get the shared time_step values (keyed on the first model)
    first_model = list(all_results.keys())[0]
    df_first = load_prediction(PREDICTION_PATHS[first_model])
    if df_first is not None and 'time_step' in df_first.columns:
        error_series['time_step'] = df_first['time_step'].values
    
    error_df = pd.DataFrame(error_series)
    error_csv_path = os.path.join(output_dir, 'all_models_error_timeseries.csv')
    error_df.to_csv(error_csv_path, index=False)
    print(f"Error time series saved to: {error_csv_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
