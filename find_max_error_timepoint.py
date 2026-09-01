"""
找出 Gene Model、LSTM、MLP 三种模型预测结果中误差最高的时间节点。
读取各模型的 prediction.csv，计算每个时间步的绝对误差（MAE），
输出误差最大的时间节点及其对应的日期和误差值。

用法: python find_max_error_timepoint.py
"""

import os
import numpy as np
import pandas as pd

# 抑制 OpenMP 警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================
# 配置：各模型 prediction.csv 的路径
# ============================================================
PREDICTION_PATHS = {
    'Gene': 'step1_gene_model_trans/results/gene_model/prediction.csv',
    'LSTM':  'step1_LSTM_model_trans/results/LSTM_model/prediction.csv',
    'MLP':   'step1_MLP_model_trans/results/MLP_model/prediction.csv',
}

# 水质原始日期文件（用于将 time_step 映射为实际日期）
WQ_DATE_FILE = 'data_WQ/wq.xlsx'


def load_dates(wq_path):
    """从 wq.xlsx 读取实际日期列表"""
    try:
        df = pd.read_excel(wq_path)
        dates = df['time'].tolist()
        print(f"成功读取 {len(dates)} 条日期记录")
        return dates
    except Exception as e:
        print(f"警告：无法读取 {wq_path}: {e}")
        return None


def load_prediction(path):
    """加载 prediction.csv，返回 DataFrame"""
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return None
    df = pd.read_csv(path)
    print(f"加载 {path}: {len(df)} 行, {len(df.columns)} 列")
    return df


def calculate_errors(df):
    """
    计算每个时间步的绝对误差（MAE）。
    
    对每个区域（ANA/ANO/OX/EFF），计算 |pred - real| 的均值，
    再对所有区域的误差取平均，得到该时间步的综合 MAE。
    
    Returns:
        errors: (N,) array — 每个时间步的 MAE
        region_errors: dict — 各区域单独的误差
    """
    real_cols = [c for c in df.columns if 'real' in c.lower()]
    pred_cols = [c for c in df.columns if 'pred' in c.lower()]
    
    real_vals = df[real_cols].values   # (N, 12)
    pred_vals = df[pred_cols].values   # (N, 12)
    
    # 总体 MAE（所有 12 个特征取平均）
    total_mae = np.mean(np.abs(pred_vals - real_vals), axis=1)  # (N,)
    
    # 按区域分解误差
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
    找出误差最大的时间节点。
    
    Args:
        errors: (N,) 误差数组
        dates: 实际日期列表（可选）
        time_steps: time_step 列值（可选）
    
    Returns:
        dict: 包含索引、日期、误差值等信息
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
    
    # 尝试获取日期
    if time_steps is not None and max_idx < len(time_steps):
        ts = time_steps[max_idx]
        result['time_step'] = int(ts)
        
        if dates is not None and ts < len(dates):
            result['date'] = dates[ts]
    
    return result


def main():
    print("=" * 60)
    print("Gene / LSTM / MLP 模型预测误差分析 — 找出最高误差时间节点")
    print("=" * 60)
    
    # 加载日期映射
    dates = load_dates(WQ_DATE_FILE)
    print()
    
    # 存储各模型结果
    all_results = {}
    
    for model_name, path in PREDICTION_PATHS.items():
        print(f"--- {model_name} Model ---")
        df = load_prediction(path)
        if df is None:
            continue
        
        # 计算误差
        total_mae, region_errors = calculate_errors(df)
        
        # 找最大误差时间点
        time_steps = df['time_step'].values if 'time_step' in df.columns else None
        result = find_max_error_timepoint(total_mae, dates=dates, time_steps=time_steps)
        result['region_errors'] = region_errors
        result['all_errors'] = total_mae
        
        all_results[model_name] = result
        
        # 打印摘要
        date_str = result.get('date', 'N/A')
        if hasattr(date_str, 'strftime'):
            date_str = date_str.strftime('%Y-%m-%d %H:%M')
        ts_str = result.get('time_step', 'N/A')
        
        print(f"  样本数:       {result['total_samples']}")
        print(f"  平均 MAE:     {result['mean_error']:.6f}")
        print(f"  MAE 标准差:   {result['std_error']:.6f}")
        print(f"  ** 最大 MAE:  {result['max_error']:.6f}**")
        print(f"  ** 时间节点:  time_step={ts_str}, 日期={date_str} **")
        
        # 该时间点各区域误差
        print(f"  该时间点各区域误差:")
        for region, re_arr in region_errors.items():
            print(f"    {region}: {re_arr[result['index']]:.6f}")
        print()
    
    # ============================================================
    # 汇总对比：三个模型的最高误差是否出现在同一时间点？
    # ============================================================
    print("=" * 60)
    print("汇总对比")
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
    
    # 找出全局最高误差（跨模型比较）
    global_max_model = max(all_results.keys(), key=lambda m: all_results[m]['max_error'])
    global_max_result = all_results[global_max_model]
    
    print(f"\n>>> 全局最高误差模型: {global_max_model}")
    print(f">>> 最高 MAE 值:     {global_max_result['max_error']:.6f}")
    print(f">>> 对应时间节点:   time_step={global_max_result.get('time_step', 'N/A')}")
    
    g_date = global_max_result.get('date', 'N/A')
    if hasattr(g_date, 'strftime'):
        g_date = g_date.strftime('%Y-%m-%d %H:%M')
    print(f">>> 对应日期:       {g_date}")
    
    # 保存汇总结果到 CSV
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细汇总
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
        # 加入各区域在该点的误差
        for region, re_arr in res['region_errors'].items():
            row[f'{region}_MAE_at_max'] = re_arr[res['index']]
        detail_rows.append(row)
    
    detail_df = pd.DataFrame(detail_rows)
    csv_path = os.path.join(output_dir, 'max_error_timepoint_summary.csv')
    detail_df.to_csv(csv_path, index=False)
    print(f"\n汇总已保存至: {csv_path}")
    
    # 同时保存每个时间步的完整误差序列（用于后续绘图等）
    error_series = {'time_step': []}
    for model_name, res in all_results.items():
        if 'time_step' in locals() or True:
            # 从原始 df 中重新获取 time_step
            pass
    # 用更简单的方式：直接保存各模型完整误差
    for model_name, res in all_results.items():
        error_series[f'{model_name}_MAE'] = res['all_errors']
    
    # 获取公共的 time_step（以第一个模型为准）
    first_model = list(all_results.keys())[0]
    df_first = load_prediction(PREDICTION_PATHS[first_model])
    if df_first is not None and 'time_step' in df_first.columns:
        error_series['time_step'] = df_first['time_step'].values
    
    error_df = pd.DataFrame(error_series)
    error_csv_path = os.path.join(output_dir, 'all_models_error_timeseries.csv')
    error_df.to_csv(error_csv_path, index=False)
    print(f"误差时序数据已保存至: {error_csv_path}")
    
    print("\n完成！")


if __name__ == '__main__':
    main()
