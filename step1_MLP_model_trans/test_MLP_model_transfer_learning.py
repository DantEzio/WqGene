import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Subset
from MLP_model import MLPModel, WaterQualityDataset, prepare_data, calculate_loss

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def compute_fisher_information(model, dataloader, device):
    """Compute the Fisher information matrix"""
    fisher_dict = {}
    param_names = []
    
    # Collect the names of all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param)
            param_names.append(name)
    
    # Compute the gradients
    model.eval()
    for batch in dataloader:
        t_inf, t_ana, t_ano, t_ox, t_eff, t_labels = batch
        
        # Move data to device
        t_inf = t_inf.to(device)
        t_ana = t_ana.to(device)
        t_ano = t_ano.to(device)
        t_ox = t_ox.to(device)
        t_eff = t_eff.to(device)
        t_labels = {key: value.to(device) for key, value in t_labels.items()}
        
        # Forward pass
        outputs = model(t_inf, t_ana, t_ano, t_ox, t_eff)
        
        # Calculate loss
        loss_dict = calculate_loss(outputs, t_labels)
        total_loss = loss_dict["total"]
        
        # Backward pass to get gradients
        model.zero_grad()
        total_loss.backward()
        
        # Accumulate Fisher information
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.pow(2)
    
    # Average Fisher information
    num_batches = len(dataloader)
    for name in fisher_dict:
        fisher_dict[name] /= num_batches
    
    return fisher_dict

def ewc_loss(model, fisher_dict, optimal_params, ewc_lambda=1000):
    """Compute the EWC loss"""
    ewc_loss_val = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher_dict:
            ewc_loss_val += (fisher_dict[name] * (param - optimal_params[name]).pow(2)).sum()
    
    return ewc_lambda * ewc_loss_val

def test_transfer_learning():
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize model
        model = MLPModel(hidden_dim=64, dropout_rate=0.3).to(device)

        # Load pre-trained MLP model
        model_dir = os.path.join(os.path.dirname(__file__), "model", "MLP_model")
        model_path = os.path.join(model_dir, "best_MLP_model.pth")
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Pre-trained MLP model loaded successfully from: {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load existing model: {e}")
                print("Using newly initialized model (untrained) for testing")
        else:
            print(f"Pre-trained MLP model not found at: {model_path}")
            print("Using newly initialized model (untrained)")

        # Prepare data
        print("Preparing test data...")
        wq_data, labels_dict = prepare_data()
        if wq_data is None:
            print("Error: Failed to load data")
            return
        
        # Create dataset
        dataset = WaterQualityDataset(wq_data, labels_dict, window_size=40)
        
        print(f"Total data size: {len(dataset)} samples")

        # Create results directory
        results_dir = os.path.join(os.path.dirname(__file__), "results", "MLP_model_transfer_learning")
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results directory: {results_dir}")

        # Test model with transfer learning
        print("\nTesting model with transfer learning...")
        predictions = {"ANA": [], "ANO": [], "OX": [], "EFF": []}
        real_values = {"ANA": [], "ANO": [], "OX": [], "EFF": []}
        
        # Add lists to store feedback mechanism data
        absolute_errors = []
        bollinger_bands = []  # List to store Bollinger Bands data
        transfer_triggers = []

        # Define parameters for transfer learning (optimized)
        window_size = 40  # Same as dataset window size
        transfer_window = 40  # Use up to 30 time steps before current for transfer learning (smaller window)
        transfer_epochs = 30  # Number of epochs for transfer learning (increased)
        transfer_batch_size = 64  # Batch size for transfer learning (increased)

        # Define optimizer for transfer learning (only update specific parameters)
        transfer_params = [
            {'params': model.mlp.parameters()},
            {'params': model.inf_proj.parameters()},
            {'params': model.ana_proj.parameters()},
            {'params': model.ano_proj.parameters()},
            {'params': model.ox_proj.parameters()},
            {'params': model.eff_proj.parameters()},
            {'params': model.output_ana.parameters()},
            {'params': model.output_ano.parameters()},
            {'params': model.output_ox.parameters()},
            {'params': model.output_eff.parameters()}
        ]
        optimizer = torch.optim.Adam(transfer_params, lr=5e-5, weight_decay=1e-5)  # Lower learning rate

        # Compute Fisher information matrix for EWC
        print("Computing Fisher information matrix for EWC...")
        # Use a subset of data for faster computation
        ewc_subset_size = min(1000, len(dataset))
        ewc_subset = Subset(dataset, range(ewc_subset_size))
        ewc_loader = DataLoader(ewc_subset, batch_size=64, shuffle=False)
        fisher_dict = compute_fisher_information(model, ewc_loader, device)
        print(f"Fisher information computed using {ewc_subset_size} samples")
        
        # Save optimal parameters for EWC
        optimal_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                optimal_params[name] = param.data.clone()
        
        # EWC lambda parameter
        ewc_lambda = 1
        print(f"EWC lambda: {ewc_lambda}")

        # Split dataset using same indices as test_gene_model
        train_size = int(0.3 * len(dataset))  # the first 0%-30% of the data
        val_size = int(0.1 * len(dataset))  # the 30%-40% of the data
        
        # Test data starts from 30% of data (train_size)
        start_idx = train_size
        test_size = len(dataset) - start_idx  # 70% of the data
        
        print(f"Test data size: {test_size} samples (from {start_idx+1} to {len(dataset)})")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Training samples: {train_size}")
        print(f"  Validation samples: {val_size}")
        print(f"  Test samples: {test_size}")

        for i in range(start_idx, len(dataset)):
            print(f"Processing time step {i+1}/{len(dataset)}...")
            
            # Get current data point
            inf_feat, ana_feat, ano_feat, ox_feat, eff_feat, labels = dataset[i]
            
            # Calculate absolute errors for feedback mechanism
            current_error = 0.0
            if i >= window_size:
                # Calculate errors for last 40 time steps
                recent_errors = []
                for j in range(max(0, i - window_size), i):
                    # Get data for time step j
                    j_inf, j_ana, j_ano, j_ox, j_eff, j_labels = dataset[j]
                    
                    # Make prediction for time step j
                    model.eval()
                    with torch.no_grad():
                        j_inf_tensor = j_inf.unsqueeze(0).to(device)
                        j_ana_tensor = j_ana.unsqueeze(0).to(device)
                        j_ano_tensor = j_ano.unsqueeze(0).to(device)
                        j_ox_tensor = j_ox.unsqueeze(0).to(device)
                        j_eff_tensor = j_eff.unsqueeze(0).to(device)
                        
                        j_outputs = model(j_inf_tensor, j_ana_tensor, j_ano_tensor, j_ox_tensor, j_eff_tensor)
                        
                        # Calculate absolute error for this time step
                        j_error = 0.0
                        for key in ["ANA", "ANO", "OX", "EFF"]:
                            pred = j_outputs[key].cpu().numpy().squeeze()
                            real = j_labels[key].numpy().squeeze()
                            j_error += np.mean(np.abs(pred - real))
                        recent_errors.append(j_error)
                
                # Calculate Bollinger Bands
                if len(recent_errors) >= window_size:
                    # Use only last 40 errors
                    recent_errors = recent_errors[-window_size:]
                    mean_error = np.mean(recent_errors)
                    std_error = np.std(recent_errors)
                    upper_band = mean_error + 3 * std_error
                    middle_band = mean_error
                    lower_band = mean_error - 3 * std_error
                    
                    # Store Bollinger Bands data
                    bollinger_bands.append({
                        "time_step": i,
                        "upper_band": upper_band,
                        "middle_band": middle_band,
                        "lower_band": lower_band
                    })
                    
                    # Calculate current error
                    model.eval()
                    with torch.no_grad():
                        current_inf = inf_feat.unsqueeze(0).to(device)
                        current_ana = ana_feat.unsqueeze(0).to(device)
                        current_ano = ano_feat.unsqueeze(0).to(device)
                        current_ox = ox_feat.unsqueeze(0).to(device)
                        current_eff = eff_feat.unsqueeze(0).to(device)
                        
                        current_outputs = model(current_inf, current_ana, current_ano, current_ox, current_eff)
                        
                        for key in ["ANA", "ANO", "OX", "EFF"]:
                            pred = current_outputs[key].cpu().numpy().squeeze()
                            real = labels[key].numpy().squeeze()
                            current_error += np.mean(np.abs(pred - real))
            
            # Determine if transfer learning is needed
            need_transfer = False
            if i >= window_size and len(bollinger_bands) > 0:
                bb_data = bollinger_bands[-1]
                threshold = bb_data["lower_band"] + 0.005 * (bb_data["middle_band"] - bb_data["lower_band"])
                if current_error > threshold:
                    need_transfer = True
            
            # Perform transfer learning if needed
            if need_transfer or i < window_size:
                # Prepare data for transfer learning (last 40 time steps or all available)
                start_idx_transfer = max(0, i - transfer_window)
                transfer_indices = list(range(start_idx_transfer, i))
                
                if len(transfer_indices) > 0:
                    # Create transfer learning dataset
                    transfer_dataset = Subset(dataset, transfer_indices)
                    transfer_loader = DataLoader(transfer_dataset, batch_size=transfer_batch_size, shuffle=False)
                    
                    # Transfer learning: update only specific parameters
                    model.train()
                    for epoch in range(transfer_epochs):
                        for batch_idx, (t_inf, t_ana, t_ano, t_ox, t_eff, t_labels) in enumerate(transfer_loader):
                            # Move data to device
                            t_inf = t_inf.to(device)
                            t_ana = t_ana.to(device)
                            t_ano = t_ano.to(device)
                            t_ox = t_ox.to(device)
                            t_eff = t_eff.to(device)
                            
                            # Move labels to device
                            t_labels = {
                                key: value.to(device) for key, value in t_labels.items()
                            }

                            # Forward pass
                            outputs = model(t_inf, t_ana, t_ano, t_ox, t_eff)
                            
                            # Calculate loss
                            loss_dict = calculate_loss(outputs, t_labels)
                            task_loss = loss_dict["total"]
                            
                            # Add EWC loss for non-transfer parameters
                            ewc_loss_val = ewc_loss(model, fisher_dict, optimal_params, ewc_lambda)
                            total_loss = task_loss + ewc_loss_val
                            
                            # Backward pass and optimize
                            optimizer.zero_grad()
                            total_loss.backward()
                            optimizer.step()
            
            # Save model after transfer learning if triggered
            if need_transfer:
                model_dir = os.path.join(os.path.dirname(__file__), "model", "MLP_model_transfer_learning")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"model_after_transfer_step_{i+1}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"  Model saved after transfer learning: {model_path}")
            
            transfer_triggers.append(need_transfer)
            
            # Now make prediction with updated model
            model.eval()
            with torch.no_grad():
                # Move current data to device
                current_inf = inf_feat.unsqueeze(0).to(device)
                current_ana = ana_feat.unsqueeze(0).to(device)
                current_ano = ano_feat.unsqueeze(0).to(device)
                current_ox = ox_feat.unsqueeze(0).to(device)
                current_eff = eff_feat.unsqueeze(0).to(device)
                
                # Forward pass to get predictions
                outputs = model(current_inf, current_ana, current_ano, current_ox, current_eff)
                
                # Collect predictions and real values
                for key in ["ANA", "ANO", "OX", "EFF"]:
                    predictions[key].append(outputs[key].cpu().numpy().squeeze())
                    real_values[key].append(labels[key].numpy().squeeze())
                
                # Calculate and print prediction error for current time step
                step_error = 0.0
                for key in ["ANA", "ANO", "OX", "EFF"]:
                    pred = outputs[key].cpu().numpy().squeeze()
                    real = labels[key].numpy().squeeze()
                    step_error += np.mean(np.abs(pred - real))
                print(f"  Time step {i+1}: Prediction error = {step_error:.6f}, Transfer learning triggered: {need_transfer}")

        # Save prediction results
        print("\nSaving prediction results...")
        
        # Prepare data for prediction.csv
        prediction_data = []
        for i in range(len(predictions["ANA"])):
            row = {
                'time_step': i,
                'ANA_real_1': real_values["ANA"][i][0],
                'ANA_real_2': real_values["ANA"][i][1],
                'ANA_real_3': real_values["ANA"][i][2],
                'ANA_pred_1': predictions["ANA"][i][0],
                'ANA_pred_2': predictions["ANA"][i][1],
                'ANA_pred_3': predictions["ANA"][i][2],
                'ANO_real_1': real_values["ANO"][i][0],
                'ANO_real_2': real_values["ANO"][i][1],
                'ANO_real_3': real_values["ANO"][i][2],
                'ANO_pred_1': predictions["ANO"][i][0],
                'ANO_pred_2': predictions["ANO"][i][1],
                'ANO_pred_3': predictions["ANO"][i][2],
                'OX_real_1': real_values["OX"][i][0],
                'OX_real_2': real_values["OX"][i][1],
                'OX_real_3': real_values["OX"][i][2],
                'OX_pred_1': predictions["OX"][i][0],
                'OX_pred_2': predictions["OX"][i][1],
                'OX_pred_3': predictions["OX"][i][2],
                'EFF_real_1': real_values["EFF"][i][0],
                'EFF_real_2': real_values["EFF"][i][1],
                'EFF_real_3': real_values["EFF"][i][2],
                'EFF_pred_1': predictions["EFF"][i][0],
                'EFF_pred_2': predictions["EFF"][i][1],
                'EFF_pred_3': predictions["EFF"][i][2],
                'transfer_trigger': transfer_triggers[i]
            }
            prediction_data.append(row)
        
        prediction_df = pd.DataFrame(prediction_data)
        prediction_path = os.path.join(results_dir, "prediction.csv")
        prediction_df.to_csv(prediction_path, index=False)
        print(f"Predictions saved: {prediction_path}")
        
        # Save Bollinger Bands data to CSV
        if bollinger_bands:
            bb_df = pd.DataFrame(bollinger_bands)
            bb_path = os.path.join(results_dir, "bollinger_bands.csv")
            bb_df.to_csv(bb_path, index=False)
            print(f"Bollinger Bands saved: {bb_path}")

        # Generate prediction vs real regression plot
        print("\nGenerating prediction vs real regression plot...")
        try:
            # Set environment variable to suppress OpenMP error
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            plt.figure(figsize=(12, 10))
            output_keys = ["ANA", "ANO", "OX", "EFF"]
            
            for i, key in enumerate(output_keys, 1):
                plt.subplot(2, 2, i)
                plt.scatter(np.array(real_values[key]).flatten(), np.array(predictions[key]).flatten(), alpha=0.5, label="Predictions")
                # Plot diagonal line for reference
                min_val = min(np.array(real_values[key]).min(), np.array(predictions[key]).min())
                max_val = max(np.array(real_values[key]).max(), np.array(predictions[key]).max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
                plt.xlabel(f"Real {key}")
                plt.ylabel(f"Predicted {key}")
                plt.title(f"Real vs Predicted {key}")
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            
            regression_plot_path = os.path.join(results_dir, "prediction_vs_real.png")
            plt.savefig(regression_plot_path)
            plt.close()
            print(f"Regression plot saved: {regression_plot_path}")
        except Exception as e:
            print(f"Error generating regression plot: {e}")

        # Generate time series prediction plot
        print("\nGenerating time series prediction plot...")
        try:
            plt.figure(figsize=(16, 12))
            output_keys = ["ANA", "ANO", "OX", "EFF"]
            
            for i, key in enumerate(output_keys, 1):
                plt.subplot(4, 1, i)
                # Plot real values
                plt.plot(range(len(real_values[key])), np.array(real_values[key])[:, 0], 'b-', label=f"Real {key}_1")
                plt.plot(range(len(real_values[key])), np.array(real_values[key])[:, 1], 'g-', label=f"Real {key}_2")
                plt.plot(range(len(real_values[key])), np.array(real_values[key])[:, 2], 'r-', label=f"Real {key}_3")
                # Plot predicted values
                plt.plot(range(len(predictions[key])), np.array(predictions[key])[:, 0], 'b--', label=f"Pred {key}_1")
                plt.plot(range(len(predictions[key])), np.array(predictions[key])[:, 1], 'g--', label=f"Pred {key}_2")
                plt.plot(range(len(predictions[key])), np.array(predictions[key])[:, 2], 'r--', label=f"Pred {key}_3")
                plt.ylabel(f"{key} Values")
                plt.title(f"Time Series Prediction for {key}")
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            
            time_series_plot_path = os.path.join(results_dir, "time_series_prediction.png")
            plt.savefig(time_series_plot_path)
            plt.close()
            print(f"Time series plot saved: {time_series_plot_path}")
        except Exception as e:
            print(f"Error generating time series plot: {e}")

        # Calculate and save MSE scores
        print("\nCalculating MSE scores...")
        mse_scores = {}
        for key in ["ANA", "ANO", "OX", "EFF"]:
            mse = np.mean((np.array(real_values[key]) - np.array(predictions[key])) ** 2)
            mse_scores[key] = mse
            print(f"MSE for {key}: {mse:.4f}")

        total_mse = sum(mse_scores.values()) / len(mse_scores)
        print(f"Total MSE: {total_mse:.4f}")

        # Save MSE scores
        mse_data = {
            "Stage": ["ANA", "ANO", "OX", "EFF"],
            "MSE": [mse_scores["ANA"], mse_scores["ANO"], mse_scores["OX"], mse_scores["EFF"]]
        }
        mse_df = pd.DataFrame(mse_data)
        
        mse_path = os.path.join(results_dir, "mse_scores.csv")
        try:
            mse_df.to_csv(mse_path, index=False)
            print(f"MSE scores saved: {mse_path}")
        except Exception as e:
            print(f"Error saving MSE scores: {e}")

        print("\n Transfer learning test completed successfully!")
        print(f"Results summary:")
        print(f"- Model: MLP Model with Transfer Learning and EWC")
        print(f"- Test samples: {test_size} (from {start_idx+1} to {len(dataset)}, 30%-100% of data)")
        print(f"- MSE scores: {mse_scores}")
        print(f"- Total MSE: {total_mse:.4f}")
        print(f"- Output files saved to: {results_dir}")
        
    except Exception as e:
        print(f"Error during transfer learning testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transfer_learning()