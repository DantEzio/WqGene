import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from gene_model import GeneModel, WaterQualityDataset, prepare_data, calculate_loss
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def train_gene_model():
    """训练基因模型"""
    # ===== 1. 准备数据 =====
    print("=" * 60)
    print("Step 1: Preparing data...")
    
    (wq_data, gene_dynamic, month_col_map, gene_dynamic_ox, month_col_map_ox,
     gene_c_dynamic, month_col_map_c, gene_c_dynamic_ox, month_col_map_c_ox, labels_dict) = prepare_data()
    
    if wq_data is None:
        print("ERROR: Data preparation failed!")
        return None
    
    # 创建数据集
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
    
    # 划分数据集: train 30%, val 10%, test 60%
    total_size = len(dataset)
    train_size = int(total_size * 0.3)   # 前30%用于训练
    val_size = int(total_size * 0.1)     # 接下来10%用于验证
    test_size = total_size - train_size - val_size  # 剩余60%用于测试
    
    # ★ 改为与LSTM一致的顺序划分方式（Subset）
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_size))
    
    # 创建DataLoader (★ batch_size改为256与LSTM一致)
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # ===== 2. 初始化模型 =====
    print("\n" + "=" * 60)
    print("Step 2: Initializing model...")
    
    model = GeneModel(hidden_dim=64, dropout_rate=0.3)  # ★ 与LSTM保持一致
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    
    # ===== 3. 定义优化器和损失函数 =====
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )
    
    # ===== 4. 训练循环 =====
    print("\n" + "=" * 60)
    print("Step 3: Training...")
    
    num_epochs = 300
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 150  # ★ 与LSTM保持一致
    
    # 记录训练历史 (★ 独立ANA和ANO)
    history = {
        'epoch': [],
        'train_loss': [], 'val_loss': [],
        'train_loss_ANA': [], 'val_loss_ANA': [],      # ★
        'train_loss_ANO': [], 'val_loss_ANO': [],       # ★
        'train_loss_OX': [], 'val_loss_OX': [],
        'train_loss_EFF': [], 'val_loss_EFF': [],
        'lr_history': []  # 学习率历史
    }
    
    for epoch in range(num_epochs):
        # ---- 训练阶段 ----
        model.train()
        train_loss_sum = 0.0
        train_loss_components = {"ANA": 0.0, "ANO": 0.0, "OX": 0.0, "EFF": 0.0}  # ★
        
        for batch_idx, (inf_feat, ana_ano_feat, ox_feat, eff_feat,  # ★ 变量名更新
                         gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, labels) in enumerate(train_loader):
            
            # 移动到设备
            inf_feat = inf_feat.to(device)
            ana_ano_feat = ana_ano_feat.to(device)  # ★
            ox_feat = ox_feat.to(device)
            eff_feat = eff_feat.to(device)
            gene_n_ana_ano = gene_n_ana_ano.to(device)  # ★
            gene_n_ox = gene_n_ox.to(device)
            gene_c_ana_ano = gene_c_ana_ano.to(device)  # ★
            gene_c_ox = gene_c_ox.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # 前向传播 (★ 参数更新)
            outputs = model(inf_feat, ana_ano_feat, ox_feat, eff_feat,
                           gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox)
            
            # 计算损失
            loss_dict = calculate_loss(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()
            
            train_loss_sum += loss_dict['total'].item()
            for key in ["ANA", "ANO", "OX", "EFF"]:  # ★
                train_loss_components[key] += loss_dict[key].item()
        
        # 平均训练损失
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_components = {k: v / len(train_loader) for k, v in train_loss_components.items()}
        
        # ---- 验证阶段 ----
        model.eval()
        val_loss_sum = 0.0
        val_loss_components = {"ANA": 0.0, "ANO": 0.0, "OX": 0.0, "EFF": 0.0}  # ★
        
        with torch.no_grad():
            for batch_idx, (inf_feat, ana_ano_feat, ox_feat, eff_feat,  # ★
                             gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, labels) in enumerate(val_loader):
                
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
                
                val_loss_sum += loss_dict['total'].item()
                for key in ["ANA", "ANO", "OX", "EFF"]:  # ★
                    val_loss_components[key] += loss_dict[key].item()
        
        # 平均验证损失
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_components = {k: v / len(val_loader) for k, v in val_loss_components.items()}
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 记录历史 (★ 独立记录ANA和ANO)
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        for key in ["ANA", "ANO", "OX", "EFF"]:  # ★
            history[f'train_loss_{key}'].append(avg_train_components[key])
            history[f'val_loss_{key}'].append(avg_val_components[key])
        history['lr_history'].append(optimizer.param_groups[0]['lr'])  # 记录当前学习率
        
        # 打印进度
        if (epoch + 1) % 50 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                  f"LR: {current_lr:.2e}")
            print(f"  Components - ANA: T={avg_train_components['ANA']:.4f}/V={avg_val_components['ANA']:.4f}, "  # ★ 独立输出块
                  f"ANO: T={avg_train_components['ANO']:.4f}/V={avg_val_components['ANO']:.4f}, "  # ★ 独立输出块
                  f"OX: T={avg_train_components['OX']:.4f}/V={avg_val_components['OX']:.4f}, "
                  f"EFF: T={avg_train_components['EFF']:.4f}/V={avg_val_components['EFF']:.4f}")
        
        # Early stopping 和保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型
            current_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(current_dir, 'model', 'gene_model')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'best_gene_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'history': history
            }, save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}. Best validation loss: {best_val_loss:.6f}")
            break
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    
    # ===== 5. 在测试集上评估 =====
    print("\n" + "=" * 60)
    print("Step 4: Evaluating on test set...")
    
    # 加载最佳模型
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_predictions = {
        "ANA": [], "ANO": [], "OX": [], "EFF": []  # ★ 独立输出块
    }
    all_labels = {
        "ANA": [], "ANO": [], "OX": [], "EFF": []  # ★ 独立输出块
    }
    all_fusion_features = {
        "fusion_ana_ano": [], "fusion_ox": []  # ★ 融合特征保持不变
    }
    test_loss_sum = 0.0
    test_loss_components = {"ANA": 0.0, "ANO": 0.0, "OX": 0.0, "EFF": 0.0}  # ★ 独立输出块
    
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
            for key in ["ANA", "ANO", "OX", "EFF"]:  # ★ 独立输出块
                test_loss_components[key] += loss_dict[key].item()
            
            # 收集预测结果和标签 (★ 独立收集ANA和ANO)
            for key in ["ANA", "ANO", "OX", "EFF"]:  # ★ 独立输出块
                all_predictions[key].append(outputs[key].cpu().numpy())
                all_labels[key].append(labels[key].cpu().numpy())
            
            # 收集融合特征
            all_fusion_features["fusion_ana_ano"].append(outputs["fusion_ana_ano"].cpu().numpy())  # ★
            all_fusion_features["fusion_ox"].append(outputs["fusion_ox"].cpu().numpy())
    
    # 拼接所有批次的结果
    predictions = {key: np.concatenate(vals, axis=0) for key, vals in all_predictions.items()}  # ★
    true_labels = {key: np.concatenate(vals, axis=0) for key, vals in all_labels.items()}  # ★
    fusion_features = {key: np.concatenate(vals, axis=0) for key, vals in all_fusion_features.items()}  # ★
    
    # 计算测试集MSE (★ 独立计算ANA和ANO)
    mse_scores = {}
    for key in ["ANA", "ANO", "OX", "EFF"]:  # ★ 独立输出块
        mse_scores[key] = np.mean((predictions[key] - true_labels[key]) ** 2)
    
    avg_test_loss = test_loss_sum / len(test_loader)
    avg_test_components = {k: v / len(test_loader) for k, v in test_loss_components.items()}
    
    print(f"\nTest Set Results:")
    print(f"  Total MSE Loss: {avg_test_loss:.6f}")
    for key in ["ANA", "ANO", "OX", "EFF"]:  # ★ 独立输出块
        print(f"  {key} MSE: {mse_scores[key]:.6f}")
    
    # ===== 6. 保存结果 =====
    print("\n" + "=" * 60)
    print("Step 5: Saving results...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, 'results', 'gene_model')
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存MSE分数
    mse_df = pd.DataFrame([mse_scores])
    mse_df.index = ['MSE']
    mse_df.to_csv(os.path.join(result_dir, 'mse_scores.csv'))
    print(f"MSE scores saved to: {os.path.join(result_dir, 'mse_scores.csv')}")
    
    # 保存训练历史（含学习率）
    training_history_with_lr = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'lr_history': history['lr_history']
    }
    np.save(os.path.join(result_dir, 'training_history_with_lr.npy'), training_history_with_lr)
    print(f"Training history (with LR) saved to: {os.path.join(result_dir, 'training_history_with_lr.npy')}")
    
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
    
    # 绘制预测 vs 真实值对比图 (★ 独立输出块，3×2布局)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # ANA 对比 (★ 独立子图)
    ax = axes[0, 0]
    ax.scatter(true_labels["ANA"][:, 0], predictions["ANA"][:, 0], alpha=0.5, label='DO-ANA', c='blue')
    ax.scatter(true_labels["ANA"][:, 1], predictions["ANA"][:, 1], alpha=0.5, label='AN-ANA', c='green')
    ax.scatter(true_labels["ANA"][:, 2], predictions["ANA"][:, 2], alpha=0.5, label='NN-ANA', c='orange')
    ax.plot([true_labels["ANA"][:, :3].min(), true_labels["ANA"][:, :3].max()],
            [true_labels["ANA"][:, :3].min(), true_labels["ANA"][:, :3].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'ANA Prediction (MSE={mse_scores["ANA"]:.4f})')
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
    ax.set_title(f'ANO Prediction (MSE={mse_scores["ANO"]:.4f})')  # ★ 新增
    ax.legend()
    ax.grid(True, alpha=0.3)

    # OX 对比
    ax = axes[1, 0]
    ax.scatter(true_labels["OX"][:, 0], predictions["OX"][:, 0], alpha=0.5)
    ax.plot([true_labels["OX"][:, 0].min(), true_labels["OX"][:, 0].max()],
            [true_labels["OX"][:, 0].min(), true_labels["OX"][:, 0].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'OX DO Prediction (MSE={mse_scores["OX"]:.4f})')
    ax.grid(True, alpha=0.3)

    # EFF COD 对比
    ax = axes[1, 1]
    ax.scatter(true_labels["EFF"][:, 0], predictions["EFF"][:, 0], alpha=0.5)
    ax.plot([true_labels["EFF"][:, 0].min(), true_labels["EFF"][:, 0].max()],
            [true_labels["EFF"][:, 0].min(), true_labels["EFF"][:, 0].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'EFF COD Prediction (MSE={mse_scores["EFF"]:.4f})')
    ax.grid(True, alpha=0.3)

    # 训练损失曲线
    ax = axes[2, 0]
    ax.plot(history['epoch'], history['train_loss'], label='Train Loss', alpha=0.7)
    ax.plot(history['epoch'], history['val_loss'], label='Val Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 隐藏未使用的子图 [2, 1]
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'prediction_vs_real.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to: {os.path.join(result_dir, 'prediction_vs_real.png')}")
    
    return model, predictions, true_labels, fusion_features


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    model, predictions, true_labels, fusion_features = train_gene_model()
    
    if model is not None:
        print("\n" + "=" * 60)
        print("Gene Model Training Completed Successfully!")
        print("=" * 60)
