import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from LSTM_model import LSTMModel, WaterQualityDataset, prepare_data, calculate_loss

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def test_model():
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize model with the same parameters as training
        model = LSTMModel(hidden_dim=64, dropout_rate=0.3).to(device)

        # Load trained model (same as transfer learning version)
        model_path = os.path.join(os.path.dirname(__file__), "model", "LSTM_model", "best_LSTM_model.pth")
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Trained model loaded successfully from: {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load existing model (architecture mismatch): {e}")
                print(f"Using newly initialized model (untrained) for testing")
        else:
            print(f"Trained model not found at: {model_path}")
            print(f"Using newly initialized model (untrained)")

        # Prepare data
        print("Preparing test data...")
        wq_data, labels_dict = prepare_data()
        if wq_data is None:
            print("Error: Failed to load data")
            return
        
        # Create dataset
        dataset = WaterQualityDataset(wq_data, labels_dict, window_size=40)
        
        # Split dataset using same indices as train_gene_model
        train_size = int(0.3 * len(dataset))  # 前0%-30%的数据
        val_size = int(0.1 * len(dataset))  # 30%-35%的数据
        
        # Test data starts from 30% of data (train_size)
        start_idx = train_size
        test_size = len(dataset) - start_idx  # 70%的数据
        
        print(f"Test data size: {test_size} samples (from {start_idx+1} to {len(dataset)})")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Training samples: {train_size}")
        print(f"  Validation samples: {val_size}")
        print(f"  Test samples: {test_size}")

        # Test model
        print("Testing model...")
        model.eval()
        predictions = {"ANA": [], "ANO": [], "OX": [], "EFF": []}
        real_values = {"ANA": [], "ANO": [], "OX": [], "EFF": []}

        with torch.no_grad():
            for i in range(start_idx, len(dataset)):
                print(f"Processing time step {i+1}/{len(dataset)}...")
                
                # Get current data point
                inf_feat, ana_feat, ano_feat, ox_feat, eff_feat, labels = dataset[i]
                
                # Move data to device
                inf_feat = inf_feat.unsqueeze(0).to(device)
                ana_feat = ana_feat.unsqueeze(0).to(device)
                ano_feat = ano_feat.unsqueeze(0).to(device)
                ox_feat = ox_feat.unsqueeze(0).to(device)
                eff_feat = eff_feat.unsqueeze(0).to(device)
                
                # Forward pass
                outputs = model(inf_feat, ana_feat, ano_feat, ox_feat, eff_feat)
                
                # Collect predictions and real values
                for key in ["ANA", "ANO", "OX", "EFF"]:
                    predictions[key].append(outputs[key].cpu().numpy().squeeze())
                    real_values[key].append(labels[key].numpy().squeeze())

        # Convert to numpy arrays
        for key in ["ANA", "ANO", "OX", "EFF"]:
            predictions[key] = np.array(predictions[key])
            real_values[key] = np.array(real_values[key])

        # Calculate MSE
        print("\nCalculating MSE scores...")
        mse_scores = {}
        for key in ["ANA", "ANO", "OX", "EFF"]:
            mse = np.mean((real_values[key] - predictions[key]) ** 2)
            mse_scores[key] = mse
            print(f"MSE for {key}: {mse:.4f}")

        total_mse = sum(mse_scores.values()) / len(mse_scores)
        print(f"Total MSE: {total_mse:.4f}")

        # Create results directory (same as transfer learning version)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, "results", "LSTM_model")
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results directory: {results_dir}")

        # Save prediction results
        print("\nSaving prediction results...")
        
        # Prepare data for prediction.csv (same format as transfer learning version)
        prediction_data = []
        for i in range(len(predictions["ANA"])):
            # Adjust time step to match transfer learning version
            time_step = start_idx + i
            row = {
                'time_step': time_step,
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
                'EFF_pred_3': predictions["EFF"][i][2]
            }
            prediction_data.append(row)
        
        prediction_df = pd.DataFrame(prediction_data)
        prediction_path = os.path.join(results_dir, "prediction.csv")
        prediction_df.to_csv(prediction_path, index=False)
        print(f"✅ Predictions saved: {prediction_path}")
        
        # Save MSE scores
        mse_data = {
            "Stage": ["ANA", "ANO", "OX", "EFF"],
            "MSE": [mse_scores["ANA"], mse_scores["ANO"], mse_scores["OX"], mse_scores["EFF"]]
        }
        mse_df = pd.DataFrame(mse_data)
        
        mse_path = os.path.join(results_dir, "mse_scores.csv")
        try:
            mse_df.to_csv(mse_path, index=False)
            print(f"✅ MSE scores saved: {mse_path}")
        except Exception as e:
            print(f"Error saving MSE scores: {e}")

        # Plot predictions vs real values
        print("\nGenerating plots...")
        try:
            # Set environment variable to suppress OpenMP error
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            plt.figure(figsize=(12, 10))
            output_keys = ["ANA", "ANO", "OX", "EFF"]
            
            for i, key in enumerate(output_keys, 1):
                plt.subplot(2, 2, i)
                plt.scatter(real_values[key].flatten(), predictions[key].flatten(), alpha=0.5, label="Predictions")
                # Plot diagonal line for reference
                min_val = min(real_values[key].min(), predictions[key].min())
                max_val = max(real_values[key].max(), predictions[key].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
                plt.xlabel(f"Real {key}")
                plt.ylabel(f"Predicted {key}")
                plt.title(f"Real vs Predicted {key}\nMSE: {mse_scores[key]:.4f}")
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            
            plot_path = os.path.join(results_dir, "prediction_vs_real.png")
            plt.savefig(plot_path)
            plt.close()  # Close the plot to free memory
            print(f"✅ Plot saved: {plot_path}")
        except Exception as e:
            print(f"Error generating plot: {e}")

        print("\n🎉 Test completed successfully!")
        print(f"Results summary:")
        print(f"- Model: LSTM Model (component-wise fusion)")
        print(f"- Test samples: {test_size} (from {start_idx+1} to {len(dataset)}, 30%-100% of data)")
        print(f"- MSE scores: {mse_scores}")
        print(f"- Total MSE: {total_mse:.4f}")
        print(f"- Output files saved to: {results_dir}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()