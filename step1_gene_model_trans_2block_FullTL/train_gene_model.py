"""
train_gene_model.py - train the two-branch GeneModel (source domain, step1)
=========================================================
Purpose:
    train the main GeneModel on the original dataset (WQ + n_gene + c_gene two-branch input):
    1) Call gene_model.prepare_data() to build the train/val/test splits
    2) Adam optimization plus LR scheduling, saving the best weights by validation loss (early stopping)
    3) Record the train/val loss and LR curves and plot the test-set comparison
Key parameters:
    batch_size = 256, hidden_dim = 64, dropout = 0.3, num_epochs = 300
Output:
    model/gene_model/best_gene_model.pth          best model weights
    model/gene_model/training_history_with_lr.npy training history (loss / lr)
    results/gene_model/prediction_vs_real.png     prediction comparison figure
    results/gene_model/prediction.csv             test-set predictions
Usage:
    cd step1_gene_model_trans_2block_FullTL && python train_gene_model.py
"""

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
    """Train the gene model"""
    # ===== 1. Prepare data =====
    print("=" * 60)
    print("Step 1: Preparing data...")
    
    (wq_data, gene_dynamic, month_col_map, gene_dynamic_ox, month_col_map_ox,
     gene_c_dynamic, month_col_map_c, gene_c_dynamic_ox, month_col_map_c_ox, labels_dict) = prepare_data()
    
    if wq_data is None:
        print("ERROR: Data preparation failed!")
        return None
    
    # Create the dataset
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
    
    # Split the dataset: train 30%, val 10%, test 60%
    total_size = len(dataset)
    train_size = int(total_size * 0.3)   # the first 30% is used for training
    val_size = int(total_size * 0.1)     # the next 10% is used for validation
    test_size = total_size - train_size - val_size  # the remaining 60% is used for testing
    
    # Switch to the same chronological split as LSTM (Subset)
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_size))
    
    # Create the DataLoader (batch_size set to 256 to match LSTM)
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # ===== 2. Initialize the model =====
    print("\n" + "=" * 60)
    print("Step 2: Initializing model...")
    
    model = GeneModel(hidden_dim=64, dropout_rate=0.3)  # same as LSTM
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    
    # ===== 3. Define the optimizer and loss function =====
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )
    
    # ===== 4. Training loop =====
    print("\n" + "=" * 60)
    print("Step 3: Training...")
    
    num_epochs = 300
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 150  # same as LSTM
    
    # Record the training history (separate ANA and ANO)
    history = {
        'epoch': [],
        'train_loss': [], 'val_loss': [],
        'train_loss_ANA': [], 'val_loss_ANA': [],      # 
        'train_loss_ANO': [], 'val_loss_ANO': [],       # 
        'train_loss_OX': [], 'val_loss_OX': [],
        'train_loss_EFF': [], 'val_loss_EFF': [],
        'lr_history': []  # learning rate history
    }
    
    for epoch in range(num_epochs):
        # ---- Training phase ----
        model.train()
        train_loss_sum = 0.0
        train_loss_components = {"ANA": 0.0, "ANO": 0.0, "OX": 0.0, "EFF": 0.0}  # 
        
        for batch_idx, (inf_feat, ana_ano_feat, ox_feat, eff_feat,  # variable names updated
                         gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, labels) in enumerate(train_loader):
            
            # Move to the device
            inf_feat = inf_feat.to(device)
            ana_ano_feat = ana_ano_feat.to(device)  # 
            ox_feat = ox_feat.to(device)
            eff_feat = eff_feat.to(device)
            gene_n_ana_ano = gene_n_ana_ano.to(device)  # 
            gene_n_ox = gene_n_ox.to(device)
            gene_c_ana_ano = gene_c_ana_ano.to(device)  # 
            gene_c_ox = gene_c_ox.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Forward pass (parameter update)
            outputs = model(inf_feat, ana_ano_feat, ox_feat, eff_feat,
                           gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox)
            
            # Compute the loss
            loss_dict = calculate_loss(outputs, labels)
            
            # Back-propagation and optimization
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()
            
            train_loss_sum += loss_dict['total'].item()
            for key in ["ANA", "ANO", "OX", "EFF"]:  # 
                train_loss_components[key] += loss_dict[key].item()
        
        # Mean training loss
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_components = {k: v / len(train_loader) for k, v in train_loss_components.items()}
        
        # ---- Validation phase ----
        model.eval()
        val_loss_sum = 0.0
        val_loss_components = {"ANA": 0.0, "ANO": 0.0, "OX": 0.0, "EFF": 0.0}  # 
        
        with torch.no_grad():
            for batch_idx, (inf_feat, ana_ano_feat, ox_feat, eff_feat,  # 
                             gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, labels) in enumerate(val_loader):
                
                inf_feat = inf_feat.to(device)
                ana_ano_feat = ana_ano_feat.to(device)  # 
                ox_feat = ox_feat.to(device)
                eff_feat = eff_feat.to(device)
                gene_n_ana_ano = gene_n_ana_ano.to(device)  # 
                gene_n_ox = gene_n_ox.to(device)
                gene_c_ana_ano = gene_c_ana_ano.to(device)  # 
                gene_c_ox = gene_c_ox.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                
                outputs = model(inf_feat, ana_ano_feat, ox_feat, eff_feat,
                               gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox)
                
                loss_dict = calculate_loss(outputs, labels)
                
                val_loss_sum += loss_dict['total'].item()
                for key in ["ANA", "ANO", "OX", "EFF"]:  # 
                    val_loss_components[key] += loss_dict[key].item()
        
        # Mean validation loss
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_components = {k: v / len(val_loader) for k, v in val_loss_components.items()}
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Record the history (ANA and ANO recorded separately)
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        for key in ["ANA", "ANO", "OX", "EFF"]:  # 
            history[f'train_loss_{key}'].append(avg_train_components[key])
            history[f'val_loss_{key}'].append(avg_val_components[key])
        history['lr_history'].append(optimizer.param_groups[0]['lr'])  # record the current learning rate
        
        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                  f"LR: {current_lr:.2e}")
            print(f"  Components - ANA: T={avg_train_components['ANA']:.4f}/V={avg_val_components['ANA']:.4f}, "  # separate output blocks
                  f"ANO: T={avg_train_components['ANO']:.4f}/V={avg_val_components['ANO']:.4f}, "  # separate output blocks
                  f"OX: T={avg_train_components['OX']:.4f}/V={avg_val_components['OX']:.4f}, "
                  f"EFF: T={avg_train_components['EFF']:.4f}/V={avg_val_components['EFF']:.4f}")
        
        # Early stopping and best-model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save the best model
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
    
    # ===== 5. Evaluate on the test set =====
    print("\n" + "=" * 60)
    print("Step 4: Evaluating on test set...")
    
    # Load the best model
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_predictions = {
        "ANA": [], "ANO": [], "OX": [], "EFF": []  # separate output blocks
    }
    all_labels = {
        "ANA": [], "ANO": [], "OX": [], "EFF": []  # separate output blocks
    }
    all_fusion_features = {
        "fusion_ana_ano": [], "fusion_ox": []  # fused features stay unchanged
    }
    test_loss_sum = 0.0
    test_loss_components = {"ANA": 0.0, "ANO": 0.0, "OX": 0.0, "EFF": 0.0}  # separate output blocks
    
    with torch.no_grad():
        for batch_idx, (inf_feat, ana_ano_feat, ox_feat, eff_feat,  # 
                         gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, labels) in enumerate(test_loader):
            
            inf_feat = inf_feat.to(device)
            ana_ano_feat = ana_ano_feat.to(device)  # 
            ox_feat = ox_feat.to(device)
            eff_feat = eff_feat.to(device)
            gene_n_ana_ano = gene_n_ana_ano.to(device)  # 
            gene_n_ox = gene_n_ox.to(device)
            gene_c_ana_ano = gene_c_ana_ano.to(device)  # 
            gene_c_ox = gene_c_ox.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            outputs = model(inf_feat, ana_ano_feat, ox_feat, eff_feat,
                           gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox)
            
            loss_dict = calculate_loss(outputs, labels)
            test_loss_sum += loss_dict['total'].item()
            for key in ["ANA", "ANO", "OX", "EFF"]:  # separate output blocks
                test_loss_components[key] += loss_dict[key].item()
            
            # Collect the predictions and labels (ANA and ANO collected separately)
            for key in ["ANA", "ANO", "OX", "EFF"]:  # separate output blocks
                all_predictions[key].append(outputs[key].cpu().numpy())
                all_labels[key].append(labels[key].cpu().numpy())
            
            # Collect the fused features
            all_fusion_features["fusion_ana_ano"].append(outputs["fusion_ana_ano"].cpu().numpy())  # 
            all_fusion_features["fusion_ox"].append(outputs["fusion_ox"].cpu().numpy())
    
    # Concatenate the results of all batches
    predictions = {key: np.concatenate(vals, axis=0) for key, vals in all_predictions.items()}  # 
    true_labels = {key: np.concatenate(vals, axis=0) for key, vals in all_labels.items()}  # 
    fusion_features = {key: np.concatenate(vals, axis=0) for key, vals in all_fusion_features.items()}  # 
    
    # Compute the test-set MSE (ANA and ANO computed separately)
    mse_scores = {}
    for key in ["ANA", "ANO", "OX", "EFF"]:  # separate output blocks
        mse_scores[key] = np.mean((predictions[key] - true_labels[key]) ** 2)
    
    avg_test_loss = test_loss_sum / len(test_loader)
    avg_test_components = {k: v / len(test_loader) for k, v in test_loss_components.items()}
    
    print(f"\nTest Set Results:")
    print(f"  Total MSE Loss: {avg_test_loss:.6f}")
    for key in ["ANA", "ANO", "OX", "EFF"]:  # separate output blocks
        print(f"  {key} MSE: {mse_scores[key]:.6f}")
    
    # ===== 6. Save results =====
    print("\n" + "=" * 60)
    print("Step 5: Saving results...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, 'results', 'gene_model')
    os.makedirs(result_dir, exist_ok=True)
    
    # Save the MSE scores
    mse_df = pd.DataFrame([mse_scores])
    mse_df.index = ['MSE']
    mse_df.to_csv(os.path.join(result_dir, 'mse_scores.csv'))
    print(f"MSE scores saved to: {os.path.join(result_dir, 'mse_scores.csv')}")
    
    # Save the training history (including the learning rate)
    training_history_with_lr = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'lr_history': history['lr_history']
    }
    np.save(os.path.join(result_dir, 'training_history_with_lr.npy'), training_history_with_lr)
    print(f"Training history (with LR) saved to: {os.path.join(result_dir, 'training_history_with_lr.npy')}")
    
    # Save the predictions (separate output blocks)
    pred_df_list = []
    for i in range(len(predictions["ANA"])):  # use the length of ANA
        row = {'sample_index': i}
        for key in ["ANA", "ANO", "OX", "EFF"]:  # separate output blocks
            row.update({f'{key}_pred_{j}': predictions[key][i][j] for j in range(predictions[key].shape[1])})
            row.update({f'{key}_true_{j}': true_labels[key][i][j] for j in range(true_labels[key].shape[1])})
        pred_df_list.append(row)
    
    pred_df = pd.DataFrame(pred_df_list)
    pred_df.to_csv(os.path.join(result_dir, 'prediction.csv'), index=False)
    print(f"Predictions saved to: {os.path.join(result_dir, 'prediction.csv')}")
    
    # Plot predicted vs real values (separate output blocks, 3x2 layout)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # ANA comparison (separate panel)
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

    # ANO comparison (new separate panel)
    ax = axes[0, 1]
    ax.scatter(true_labels["ANO"][:, 0], predictions["ANO"][:, 0], alpha=0.5, label='DO-ANO', c='blue')
    ax.scatter(true_labels["ANO"][:, 1], predictions["ANO"][:, 1], alpha=0.5, label='AN-ANO', c='green')
    ax.scatter(true_labels["ANO"][:, 2], predictions["ANO"][:, 2], alpha=0.5, label='NN-ANO', c='orange')
    ax.plot([true_labels["ANO"][:, :3].min(), true_labels["ANO"][:, :3].max()],
            [true_labels["ANO"][:, :3].min(), true_labels["ANO"][:, :3].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'ANO Prediction (MSE={mse_scores["ANO"]:.4f})')  # new panel
    ax.legend()
    ax.grid(True, alpha=0.3)

    # OX comparison
    ax = axes[1, 0]
    ax.scatter(true_labels["OX"][:, 0], predictions["OX"][:, 0], alpha=0.5)
    ax.plot([true_labels["OX"][:, 0].min(), true_labels["OX"][:, 0].max()],
            [true_labels["OX"][:, 0].min(), true_labels["OX"][:, 0].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'OX DO Prediction (MSE={mse_scores["OX"]:.4f})')
    ax.grid(True, alpha=0.3)

    # EFF COD comparison
    ax = axes[1, 1]
    ax.scatter(true_labels["EFF"][:, 0], predictions["EFF"][:, 0], alpha=0.5)
    ax.plot([true_labels["EFF"][:, 0].min(), true_labels["EFF"][:, 0].max()],
            [true_labels["EFF"][:, 0].min(), true_labels["EFF"][:, 0].max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'EFF COD Prediction (MSE={mse_scores["EFF"]:.4f})')
    ax.grid(True, alpha=0.3)

    # Training loss curve
    ax = axes[2, 0]
    ax.plot(history['epoch'], history['train_loss'], label='Train Loss', alpha=0.7)
    ax.plot(history['epoch'], history['val_loss'], label='Val Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hide the unused subplot [2, 1]
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
