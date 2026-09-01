import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from gene_model import GeneModel, WaterQualityDataset, prepare_data, calculate_loss
import matplotlib.pyplot as plt

def test_gene_model():
    """测试基因模型"""
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
    from torch.utils.data import random_split
    _, _, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    batch_size = 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Test set size: {len(test_dataset)}")
    
    # ===== 2. 加载模型 =====
    print("\n" + "=" * 60)
    print("Step 2: Loading model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GeneModel(hidden_dim=256, dropout_rate=0.3)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', 'gene_model', 'best_gene_model.pth')
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Device: {device}")
    
    # ===== 3. 测试模型 =====
    print("\n" + "=" * 60)
    print("Step 3: Testing model...")
    
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
    
    print(f"\nTest Set Results:")
    print(f"  Total MSE Loss: {avg_test_loss:.6f}")
    for key in ["ANA", "ANO", "OX", "EFF"]:  # ★ 独立输出块
        print(f"  {key} MSE: {mse_scores[key]:.6f}")
    
    # ===== 4. 保存结果 =====
    print("\n" + "=" * 60)
    print("Step 4: Saving results...")
    
    result_dir = os.path.join(current_dir, 'results', 'gene_model')
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存MSE分数
    mse_df = pd.DataFrame([mse_scores])
    mse_df.index = ['MSE']
    mse_df.to_csv(os.path.join(result_dir, 'mse_scores.csv'))
    print(f"MSE scores saved to: {os.path.join(result_dir, 'mse_scores.csv')}")
    
    # 保存预测结果 (★ 独立输出块)
    pred_df_list = []
    for i in range(len(predictions["ANA"])):  # ★ 使用ANA的长度
        row = {'sample_index': i}
        for key in ["ANA", "ANO", "OX", "EFF"]:  # ★ 独立输出块
            row.update({f'{key}_pred_{j}': predictions[key][i][j] for j in range(predictions[key].shape[1])})
            row.update({f'{key}_true_{j}': true_labels[key][i][j] for j in range(true_labels[key].shape[1])})
        pred_df_list.append(row)
    
    pred_df = pd.DataFrame(pred_df_list)
    pred_df.to_csv(os.path.join(result_dir, 'prediction.csv'), index=False)
    print(f"Predictions saved to: {os.path.join(result_dir, 'prediction.csv')}")
    
    # 绘制对比图 (★ 独立输出块，改为2行布局)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ANA 对比 (★ 独立子图)
    ax = axes[0, 0]
    ax.scatter(true_labels["ANA"][:, 0], predictions["ANA"][:, 0], alpha=0.5, label='DO-ANA', c='blue')
    ax.scatter(true_labels["ANA"][:, 1], predictions["ANA"][:, 1], alpha=0.5, label='AN-ANA', c='green')
    ax.scatter(true_labels["ANA"][:, 2], predictions["ANA"][:, 2], alpha=0.5, label='NN-ANA', c='orange')
    ax.plot([true_labels["ANA"][:, :3].min(), true_labels["ANA"][:, :3].max()],
            [true_labels["ANA"][:, :3].min(), true_labels["ANA"][:, :3].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'ANA Prediction\n(MSE={mse_scores["ANA"]:.4f})')  # ★
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ANO 对比 (★ 新增独立子图)
    ax = axes[0, 1]
    ax.scatter(true_labels["ANO"][:, 0], predictions["ANO"][:, 0], alpha=0.5, label='DO-ANO', c='blue')
    ax.scatter(true_labels["ANO"][:, 1], predictions["ANO"][:, 1], alpha=0.5, label='AN-ANO', c='green')
    ax.scatter(true_labels["ANO"][:, 2], predictions["ANO"][:, 2], alpha=0.5, label='NN-ANO', c='orange')
    ax.plot([true_labels["ANO"][:, :3].min(), true_labels["ANO"][:, :3].max()],
            [true_labels["ANO"][:, :3].min(), true_labels["ANO"][:, :3].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'ANO Prediction\n(MSE={mse_scores["ANO"]:.4f})')  # ★ 新增
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # OX 对比 (★ 位置调整到 [1,0])
    ax = axes[1, 0]
    ax.scatter(true_labels["OX"][:, 0], predictions["OX"][:, 0], alpha=0.5)
    ax.plot([true_labels["OX"][:, 0].min(), true_labels["OX"][:, 0].max()],
            [true_labels["OX"][:, 0].min(), true_labels["OX"][:, 0].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'OX DO Prediction\n(MSE={mse_scores["OX"]:.4f})')
    ax.grid(True, alpha=0.3)
    
    # EFF COD 对比 (★ 位置调整到 [1,1])
    ax = axes[1, 1]
    ax.scatter(true_labels["EFF"][:, 0], predictions["EFF"][:, 0], alpha=0.5)
    ax.plot([true_labels["EFF"][:, 0].min(), true_labels["EFF"][:, 0].max()],
            [true_labels["EFF"][:, 0].min(), true_labels["EFF"][:, 0].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'EFF COD Prediction\n(MSE={mse_scores["EFF"]:.4f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'prediction_vs_real.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to: {os.path.join(result_dir, 'prediction_vs_real.png')}")
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'fusion_features': fusion_features,
        'mse_scores': mse_scores
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    results = test_gene_model()
    
    if results is not None:
        print("\n" + "=" * 60)
        print("Gene Model Testing Completed Successfully!")
        print("=" * 60)
