import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, random_split, Subset
from gene_model import GeneModel, WaterQualityDataset, prepare_data, calculate_loss
import matplotlib.pyplot as plt

def test_gene_model_transfer_learning():
    """测试基因模型迁移学习"""
    
    # ===== 1. 准备数据 =====
    print("=" * 60)
    print("Step 1: Preparing data...")
    
    (wq_data, gene_dynamic, month_col_map, gene_dynamic_ox, month_col_map_ox,
     gene_c_dynamic, month_col_map_c, gene_c_dynamic_ox, month_col_map_c_ox, labels_dict) = prepare_data()
    
    if wq_data is None:
        print("ERROR: Data preparation failed!")
        return None
    
    # 创建完整数据集
    dataset = WaterQualityDataset(
        wq_data=wq_data,
        gene_dynamic=gene_dynamic,
        month_col_map=month_col_map,
        gene_dynamic_ox=gene_dynamic_ox,
        month_col_map_ox=month_col_map_ox,
        gene_c_dynamic=gene_c_dynamic,
        month_col_map_c=month_col_map_c,
        gene_c_dynamic_ox=gene_c_dynamic_ox,
        month_col_map_c_ox=month_col_map_c_ox,
        labels_dict=labels_dict,
        window_size=40
    )
    
    total_size = len(dataset)
    train_size = int(total_size * 0.3)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size
    
    # 使用与训练相同的划分方式
    _, _, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    batch_size = 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Test set size: {len(test_dataset)}")
    
    # ===== 2. 加载预训练模型并应用迁移学习 =====
    print("\n" + "=" * 60)
    print("Step 2: Loading pre-trained model for transfer learning...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GeneModel(hidden_dim=256, dropout_rate=0.3)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, 'model', 'gene_model', 'best_gene_model.pth')
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 冻结基因投影层和融合层（保持预训练的基因知识）
    freeze_layers = [
        'gene_n_proj_ana_ano',  # ★ 冻结合并后的层
        'gene_n_proj_ox',
        'gene_c_proj_ana_ano',  # ★ 冻结合并后的层
        'gene_c_proj_ox',
        'gene_fusion_ana_ano',  # ★ 冻结合并后的层
        'gene_fusion_ox'
    ]
    
    for name, param in model.named_parameters():
        for layer_name in freeze_layers:
            if layer_name in name:
                param.requires_grad = False
                break
    
    # 只解冻输出层进行微调 (★ 独立输出头)
    unfreeze_layers = ['output_ana', 'output_ano', 'output_ox', 'output_eff']  # ★ 拆分为独立输出头
    for name, param in model.named_parameters():
        for layer_name in unfreeze_layers:
            if layer_name in name:
                param.requires_grad = True
                break
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (for fine-tuning): {trainable_params:,}")
    
    model.eval()
    
    # ===== 3. 测试模型 =====
    print("\n" + "=" * 60)
    print("Step 3: Testing transfer learning model...")
    
    all_predictions = {
        "ANA": [], "ANO": [], "OX": [], "EFF": []  # ★
    }
    all_labels = {
        "ANA": [], "ANO": [], "OX": [], "EFF": []  # ★
    }
    all_fusion_features = {
        "fusion_ana_ano": [], "fusion_ox": []  # ★
    }
    test_loss_sum = 0.0
    test_loss_components = {"ANA": 0.0, "ANO": 0.0, "OX": 0.0, "EFF": 0.0}  # ★
    
    with torch.no_grad():
        for batch_idx, (inf_feat, ana_ano_feat, ox_feat, eff_feat,  # ★
                         gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, labels) in enumerate(test_loader):
            
            inf_feat = inf_feat.to(device)
            ana_ano_feat = ana_ano_feat.to(device)  # ★
            ox_feat = ox_feat.to(device)
            eff_feat = eff_feat.to(device)
            gene_n_ana_ano = gene_n_ana_ano.to(device)  # ★
            gene_n_ox = gene_n_ox.to(device)
            gene_c_ana_ano = gene_c_ana_ano.to(device)  # ★
            gene_c_ox = gene_c_ox.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            outputs = model(inf_feat, ana_ano_feat, ox_feat, eff_feat,
                           gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox)
            
            loss_dict = calculate_loss(outputs, labels)
            test_loss_sum += loss_dict['total'].item()
            for key in ["ANA", "ANO", "OX", "EFF"]:  # ★
                test_loss_components[key] += loss_dict[key].item()
            
            # 收集预测结果和标签 (★ 独立收集ANA和ANO)
            for key in ["ANA", "ANO", "OX", "EFF"]:  # ★
                all_predictions[key].append(outputs[key].cpu().numpy())
                all_labels[key].append(labels[key].cpu().numpy())
            
            # 收集融合特征
            all_fusion_features["fusion_ana_ano"].append(outputs["fusion_ana_ano"].cpu().numpy())  # ★
            all_fusion_features["fusion_ox"].append(outputs["fusion_ox"].cpu().numpy())
    
    # 拼接所有批次的结果
    predictions = {key: np.concatenate(vals, axis=0) for key, vals in all_predictions.items()}  # ★
    true_labels = {key: np.concatenate(vals, axis=0) for key, vals in all_labels.items()}  # ★
    fusion_features = {key: np.concatenate(vals, axis=0) for key, vals in all_fusion_features.items()}  # ★
    
    # 计算MSE (★ 独立计算ANA和ANO)
    mse_scores = {}
    for key in ["ANA", "ANO", "OX", "EFF"]:  # ★
        mse_scores[key] = np.mean((predictions[key] - true_labels[key]) ** 2)
    
    avg_test_loss = test_loss_sum / len(test_loader)
    avg_test_components = {k: v / len(test_loader) for k, v in test_loss_components.items()}
    
    print(f"\nTransfer Learning Test Set Results:")
    print(f"  Total MSE Loss: {avg_test_loss:.6f}")
    for key in ["ANA_ANO", "OX", "EFF"]:  # ★
        print(f"  {key} MSE: {mse_scores[key]:.6f}")
    
    # ===== 4. 保存结果 =====
    print("\n" + "=" * 60)
    print("Step 4: Saving results...")
    
    result_dir = os.path.join(current_dir, 'results', 'gene_model_transfer_learning')
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存MSE分数
    mse_df = pd.DataFrame([mse_scores])
    mse_df.index = ['MSE']
    mse_df.to_csv(os.path.join(result_dir, 'mse_scores.csv'))
    print(f"MSE scores saved to: {os.path.join(result_dir, 'mse_scores.csv')}")
    
    # 保存预测结果
    pred_df_list = []
    for i in range(len(predictions["ANA_ANO"])):  # ★
        row = {'sample_index': i}
        for key in ["ANA_ANO", "OX", "EFF"]:  # ★
            row.update({f'{key}_pred_{j}': predictions[key][i][j] for j in range(predictions[key].shape[1])})
            row.update({f'{key}_true_{j}': true_labels[key][i][j] for j in range(true_labels[key].shape[1])})
        pred_df_list.append(row)
    
    pred_df = pd.DataFrame(pred_df_list)
    pred_df.to_csv(os.path.join(result_dir, 'prediction.csv'), index=False)
    print(f"Predictions saved to: {os.path.join(result_dir, 'prediction.csv')}")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ANA_ANO 对比 (★ 合并绘制)
    ax = axes[0]
    ax.scatter(true_labels["ANA_ANO"][:, 0], predictions["ANA_ANO"][:, 0], alpha=0.5, label='DO-ANA')
    ax.scatter(true_labels["ANA_ANO"][:, 3], predictions["ANA_ANO"][:, 3], alpha=0.5, label='DO-ANO')  # ★
    ax.plot([true_labels["ANA_ANO"][:, 0].min(), true_labels["ANA_ANO"][:, 0].max()],
            [true_labels["ANA_ANO"][:, 0].min(), true_labels["ANA_ANO"][:, 0].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Transfer Learning ANA_ANO DO\n(MSE={mse_scores["ANA_ANO"]:.4f})')  # ★
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # OX 对比
    ax = axes[1]
    ax.scatter(true_labels["OX"][:, 0], predictions["OX"][:, 0], alpha=0.5)
    ax.plot([true_labels["OX"][:, 0].min(), true_labels["OX"][:, 0].max()],
            [true_labels["OX"][:, 0].min(), true_labels["OX"][:, 0].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Transfer Learning OX DO\n(MSE={mse_scores["OX"]:.4f})')
    ax.grid(True, alpha=0.3)
    
    # EFF COD 对比
    ax = axes[2]
    ax.scatter(true_labels["EFF"][:, 0], predictions["EFF"][:, 0], alpha=0.5)
    ax.plot([true_labels["EFF"][:, 0].min(), true_labels["EFF"][:, 0].max()],
            [true_labels["EFF"][:, 0].min(), true_labels["EFF"][:, 0].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Transfer Learning EFF COD\n(MSE={mse_scores["EFF"]:.4f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'prediction_vs_real.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to: {os.path.join(result_dir, 'prediction_vs_real.png')}")
    
    # 绘制时间序列预测图
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    time_indices = np.arange(len(predictions["ANA_ANO"]))  # ★
    
    # ANA_ANO DO 时间序列 (★ 合并绘制)
    ax = axes[0]
    ax.plot(time_indices, true_labels["ANA_ANO"][:, 0], 'b-', label='True DO-ANA', alpha=0.7)
    ax.plot(time_indices, predictions["ANA_ANO"][:, 0], 'r--', label='Pred DO-ANA', alpha=0.7)
    ax.plot(time_indices, true_labels["ANA_ANO"][:, 3], 'g-', label='True DO-ANO', alpha=0.7)  # ★
    ax.plot(time_indices, predictions["ANA_ANO"][:, 3], 'm--', label='Pred DO-ANO', alpha=0.7)  # ★
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('DO Value')
    ax.set_title('Transfer Learning: ANA_ANO DO Time Series Prediction')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # OX DO 时间序列
    ax = axes[1]
    ax.plot(time_indices, true_labels["OX"][:, 0], 'b-', label='True DO-OX', alpha=0.7)
    ax.plot(time_indices, predictions["OX"][:, 0], 'r--', label='Pred DO-OX', alpha=0.7)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('DO Value')
    ax.set_title('Transfer Learning: OX DO Time Series Prediction')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # EFF COD 时间序列
    ax = axes[2]
    ax.plot(time_indices, true_labels["EFF"][:, 0], 'b-', label='True COD-EFF', alpha=0.7)
    ax.plot(time_indices, predictions["EFF"][:, 0], 'r--', label='Pred COD-EFF', alpha=0.7)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('COD Value')
    ax.set_title('Transfer Learning: EFF COD Time Series Prediction')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'time_series_prediction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Time series plot saved to: {os.path.join(result_dir, 'time_series_prediction.png')}")
    
    # 计算并保存布林带监控指标
    print("\n" + "=" * 60)
    print("Step 5: Calculating Bollinger Bands monitoring metrics...")
    
    bollinger_results = []
    
    for key in ["ANA_ANO", "OX", "EFF"]:  # ★
        pred = predictions[key][:, 0] if key != "ANA_ANO" else predictions[key][:, 0]  # 取第一个特征
        true_val = true_labels[key][:, 0] if key != "ANA_ANO" else true_labels[key][:, 0]
        
        window = min(20, len(pred))
        rolling_mean = pd.Series(pred).rolling(window=window).mean()
        rolling_std = pd.Series(pred).rolling(window=window).std()
        
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        
        anomalies = []
        for i in range(len(pred)):
            if not np.isnan(upper_band.iloc[i]) and not np.isnan(lower_band.iloc[i]):
                is_anomaly = pred[i] > upper_band.iloc[i] or pred[i] < lower_band.iloc[i]
                error = abs(pred[i] - true_val[i])
                anomalies.append({
                    'zone': key,
                    'sample_index': i,
                    'predicted': pred[i],
                    'true_value': true_val[i],
                    'error': error,
                    'upper_band': upper_band.iloc[i],
                    'lower_band': lower_band.iloc[i],
                    'is_anomaly': is_anomaly
                })
        
        bollinger_results.extend(anomalies)
        
        anomaly_count = sum(1 for a in anomalies if a['is_anomaly'])
        total_valid = len(anomalies)
        anomaly_rate = anomaly_count / total_valid * 100 if total_valid > 0 else 0
        
        mean_error = np.mean([a['error'] for a in anomalies])
        max_error = max(a['error'] for a in anomalies) if anomalies else 0
        
        print(f"\n{key} Bollinger Bands Analysis:")
        print(f"  Total valid samples: {total_valid}")
        print(f"  Anomaly count: {anomaly_count}")
        print(f"  Anomaly rate: {anomaly_rate:.2f}%")
        print(f"  Mean prediction error: {mean_error:.6f}")
        print(f"  Max prediction error: {max_error:.6f}")
    
    bollinger_df = pd.DataFrame(bollinger_results)
    bollinger_df.to_csv(os.path.join(result_dir, 'bollinger_bands.csv'), index=False)
    print(f"\nBollinger Bands results saved to: {os.path.join(result_dir, 'bollinger_bands.csv')}")
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'fusion_features': fusion_features,
        'mse_scores': mse_scores,
        'bollinger_results': bollinger_results
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    results = test_gene_model_transfer_learning()
    
    if results is not None:
        print("\n" + "=" * 60)
        print("Gene Model Transfer Learning Testing Completed Successfully!")
        print("=" * 60)
