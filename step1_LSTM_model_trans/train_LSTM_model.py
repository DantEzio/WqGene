import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from LSTM_model import LSTMModel, WaterQualityDataset, prepare_data, calculate_loss

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def train_model():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    wq_data, labels_dict = prepare_data()
    if wq_data is None:
        print("Error: Failed to load data")
        return
    
    # Create dataset
    dataset = WaterQualityDataset(wq_data, labels_dict, window_size=40)
    
    # Split dataset using indices
    train_size = int(0.3 * len(dataset))  # 前0%-30%的数据
    val_size = int(0.1 * len(dataset))  # 30%-35%的数据
    test_size = len(dataset) - train_size - val_size
    
    # Use Subset to select data by indices
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))
    
    print(f"Data split:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Test samples: {test_size}")
    
    # Create data loaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    
    # Initialize model
    model = LSTMModel(hidden_dim=64, dropout_rate=0.3).to(device)
    print(f"Model initialized with hidden_dim=64, dropout_rate=0.3")
    
    # Define optimizer and scheduler
    initial_lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    
    # Advanced scheduler with multiple strategies
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # More aggressive reduction
        patience=30,  # Shorter patience
        min_lr=1e-8,  # Lower minimum LR
        verbose=True  # Print LR changes
    )
    
    # Training parameters
    num_epochs = 3000
    patience = 150  # Reduced for faster early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Learning rate history
    lr_history = []
    train_loss_history = []
    val_loss_history = []
    
    # Create model directory if not exists
    model_dir = os.path.join(os.path.dirname(__file__), "model", "LSTM_model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
    
    # Create results directory if not exists
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    # Training loop
    print(f"Starting training with {num_epochs} epochs...")
    print(f"Initial learning rate: {initial_lr}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_loss_components = {"ANA": 0, "ANO": 0, "OX": 0, "EFF": 0}
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Training loop
        for batch_idx, (inf_feat, ana_feat, ano_feat, ox_feat, eff_feat, labels) in enumerate(train_loader):
            # Move data to device
            inf_feat = inf_feat.to(device)
            ana_feat = ana_feat.to(device)
            ano_feat = ano_feat.to(device)
            ox_feat = ox_feat.to(device)
            eff_feat = eff_feat.to(device)
            
            # Move labels to device
            labels_device = {k: v.to(device) for k, v in labels.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inf_feat, ana_feat, ano_feat, ox_feat, eff_feat)
            
            # Calculate loss
            loss_dict = calculate_loss(outputs, labels_device)
            loss = loss_dict["total"]
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
            for key in ["ANA", "ANO", "OX", "EFF"]:
                train_loss_components[key] += loss_dict[key].item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_components = {k: v / len(train_loader) for k, v in train_loss_components.items()}
        train_loss_history.append(avg_train_loss)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_loss_components = {"ANA": 0, "ANO": 0, "OX": 0, "EFF": 0}
        
        with torch.no_grad():
            for batch_idx, (inf_feat, ana_feat, ano_feat, ox_feat, eff_feat, labels) in enumerate(val_loader):
                # Move data to device
                inf_feat = inf_feat.to(device)
                ana_feat = ana_feat.to(device)
                ano_feat = ano_feat.to(device)
                ox_feat = ox_feat.to(device)
                eff_feat = eff_feat.to(device)
                
                # Move labels to device
                labels_device = {k: v.to(device) for k, v in labels.items()}
                
                # Forward pass
                outputs = model(inf_feat, ana_feat, ano_feat, ox_feat, eff_feat)
                
                # Calculate loss
                loss_dict = calculate_loss(outputs, labels_device)
                loss = loss_dict["total"]
                
                # Accumulate loss
                val_loss += loss.item()
                for key in ["ANA", "ANO", "OX", "EFF"]:
                    val_loss_components[key] += loss_dict[key].item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_components = {k: v / len(val_loader) for k, v in val_loss_components.items()}
        val_loss_history.append(avg_val_loss)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Check early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            # Save best model
            model_path = os.path.join(model_dir, "best_LSTM_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Loss components - Train: {avg_train_loss_components}")
        print(f"  Loss components - Val: {avg_val_loss_components}")
        print(f"  Early stop counter: {early_stop_counter}/{patience}")
    
    # Save training history
    history_data = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'lr_history': lr_history
    }
    
    # Save to numpy file
    np.save(os.path.join(model_dir, 'training_history_with_lr.npy'), history_data)
    print(f"\nTraining history saved to: {os.path.join(model_dir, 'training_history_with_lr.npy')}")
    
    # Plot learning rate history
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(lr_history)
    plt.title('Learning Rate History')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_with_lr_history.png'))
    print(f"Training and LR history plot saved to: {os.path.join(results_dir, 'training_with_lr_history.png')}")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved at: {os.path.join(model_dir, 'best_LSTM_model.pth')}")

if __name__ == "__main__":
    train_model()