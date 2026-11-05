import torch
import pandas as pd
import numpy as np
import os
from gene_model import GeneModel, WaterQualityDataset, prepare_data
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)


def _forward_one(model, inf, ana_ano, ox, eff, device):
    """Single-sample inference wrapper (ablation AEGeneRemove: water quality input only, no gene input)"""
    return model(
        inf.unsqueeze(0).to(device),
        ana_ano.unsqueeze(0).to(device),
        ox.unsqueeze(0).to(device),
        eff.unsqueeze(0).to(device),
    )


def test_gene_model_transfer_learning():
    # ===== 1. Prepare data (data loading is exactly the same as the original, unchanged) =====
    print("=" * 60)
    print("Step 1: Preparing data...")

    (wq_data, gene_dynamic, month_col_map, gene_dynamic_ox, month_col_map_ox,
     gene_c_dynamic, month_col_map_c, gene_c_dynamic_ox, month_col_map_c_ox,
     labels_dict) = prepare_data()

    if wq_data is None:
        print("ERROR: Data preparation failed!")
        return None

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
    train_size = int(0.3 * total_size)
    val_size = int(0.1 * total_size)
    start_idx = train_size  # the test window starts at 30% (chronological slicing, same as LSTM trans)
    print(f"Total samples: {total_size} | train: {train_size} | val: {val_size} | test from: {start_idx}")

    # ===== 2. Load the ablation model (AEGeneRemove: wq input only) =====
    print("\n" + "=" * 60)
    print("Step 2: Loading ablated model (AEGeneRemove, wq-only)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Note: a checkpoint check shows best_gene_model.pth is actually hidden_dim=64
    model = GeneModel(hidden_dim=64, dropout_rate=0.3).to(device)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the pretrained baseline weights first (strict=False: ignore gene-related weights)
    pretrained_path = os.path.join(current_dir, 'model', 'gene_model', 'best_gene_model.pth')
    checkpoint = torch.load(pretrained_path, map_location=device)
    # Ablation (AEGeneRemove): after dropping the gene part, load the trained weights with strict=False and ignore gene-related weights
    incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print(f"Pre-trained model loaded from: {pretrained_path}")
    print(f"  Missing keys (random init): {list(incompatible.missing_keys)}")
    print(f"  Ignored gene-related keys: {list(incompatible.unexpected_keys)}")

    # Snapshot directory of the per-step transfer learning (model parameters come from here)
    snapshot_dir = os.path.join(current_dir, 'model', 'gene_model_transfer_learning')

    # ===== 3. Result containers =====
    predictions = {"ANA": [], "ANO": [], "OX": [], "EFF": []}
    real_values = {"ANA": [], "ANO": [], "OX": [], "EFF": []}
    fusion_vectors = {"fusion_ana_ano": [], "fusion_ox": []}
    transfer_triggers = []

    # ===== 4. Per-step loop: load the model of each time step and predict =====
    print("\n" + "=" * 60)
    print("Step 3: Per-step prediction (loading per-step snapshot models)...")

    for i in range(start_idx, len(dataset)):
        inf_feat, ana_ano_feat, ox_feat, eff_feat, \
            gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, labels = dataset[i]

        # ---- Per-step model loading (loading method unchanged) ----
        # The original transfer learning saves model_after_transfer_step_{i+1}.pth once triggered,
        # This snapshot is the model state used to predict step i.
        snap_path = os.path.join(snapshot_dir, f"model_after_transfer_step_{i+1}.pth")
        snapshot_loaded = False
        if os.path.exists(snap_path):
            state = torch.load(snap_path, map_location=device)
            # Ablation (AEGeneRemove): strict=False, load only the wq part and later modules, ignore gene-related weights
            model.load_state_dict(state, strict=False)
            model.eval()
            snapshot_loaded = True
        # Steps without a snapshot reuse the model loaded at the previous step (same as the original behaviour when no transfer fired)

        transfer_triggers.append(snapshot_loaded)

        # ---- Predict the current step (ablation: water quality input only, no gene input) ----
        model.eval()
        with torch.no_grad():
            outputs = _forward_one(model, inf_feat, ana_ano_feat, ox_feat, eff_feat, device)

        for k in ["ANA", "ANO", "OX", "EFF"]:
            predictions[k].append(outputs[k].cpu().numpy().squeeze())
            real_values[k].append(labels[k].numpy().squeeze())
        for k in ["fusion_ana_ano", "fusion_ox"]:
            if k in outputs:
                fusion_vectors[k].append(outputs[k].cpu().numpy().squeeze())

        step_error = 0.0
        for k in ["ANA", "ANO", "OX", "EFF"]:
            step_error += np.mean(np.abs(
                outputs[k].cpu().numpy().squeeze() - labels[k].numpy().squeeze()))
        print(f"  Time step {i+1}/{len(dataset)}: error = {step_error:.6f}, "
              f"snapshot loaded: {snapshot_loaded}")

    # ===== 5. Write results to disk (same format as the original) =====
    print("\n" + "=" * 60)
    print("Step 4: Saving results...")

    result_dir = os.path.join(current_dir, 'results', 'gene_model_transfer_learning')
    os.makedirs(result_dir, exist_ok=True)

    # prediction.csv (including the transfer_trigger column)
    rows = []
    for i_ in range(len(predictions["ANA"])):
        row = {'time_step': i_}
        for k in ["ANA", "ANO", "OX", "EFF"]:
            for j in range(3):
                row[f'{k}_real_{j+1}'] = real_values[k][i_][j]
                row[f'{k}_pred_{j+1}'] = predictions[k][i_][j]
        row['transfer_trigger'] = bool(transfer_triggers[i_])
        rows.append(row)
    pred_path = os.path.join(result_dir, 'prediction.csv')
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    print(f"Predictions saved: {pred_path}")

    # fusion vectors .npy
    for k, v in fusion_vectors.items():
        if v:
            npy_path = os.path.join(result_dir, f'{k}.npy')
            np.save(npy_path, np.array(v))
            print(f"Fusion vectors saved: {npy_path}")

    # 2x2 scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, k in zip(axes.flatten(), ["ANA", "ANO", "OX", "EFF"]):
        r = np.array(real_values[k])
        p = np.array(predictions[k])
        ax.scatter(r.flatten(), p.flatten(), alpha=0.5, label="Predictions")
        mn = float(min(r.min(), p.min()))
        mx = float(max(r.max(), p.max()))
        ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label="Perfect Prediction")
        ax.set_xlabel(f'Real {k}')
        ax.set_ylabel(f'Predicted {k}')
        ax.set_title(f'Real vs Predicted {k}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(result_dir, 'prediction_vs_real.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Regression plot saved: {fig_path}")

    # 4-row time series figure
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    for ax, k in zip(axes, ["ANA", "ANO", "OX", "EFF"]):
        ts = np.arange(len(predictions[k]))
        r = np.array(real_values[k])
        p = np.array(predictions[k])
        for j in range(3):
            ax.plot(ts, r[:, j], '-', label=f'Real {k}_{j+1}', alpha=0.7)
            ax.plot(ts, p[:, j], '--', label=f'Predicted {k}_{j+1}', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(k)
        ax.set_title(f'{k} Time Series Prediction (Transfer Learning, AEGeneRemove)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ts_path = os.path.join(result_dir, 'time_series_prediction.png')
    plt.savefig(ts_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Time series plot saved: {ts_path}")

    # MSE
    print("\nCalculating MSE scores...")
    mse_scores = {}
    for k in ["ANA", "ANO", "OX", "EFF"]:
        mse_scores[k] = float(np.mean((np.array(real_values[k]) - np.array(predictions[k])) ** 2))
        print(f"  MSE for {k}: {mse_scores[k]:.10f}")
    total_mse = sum(mse_scores.values()) / len(mse_scores)
    print(f"  Total MSE: {total_mse:.10f}")

    mse_path = os.path.join(result_dir, 'mse_scores.csv')
    pd.DataFrame({'Stage': list(mse_scores.keys()),
                  'MSE': list(mse_scores.values())}).to_csv(mse_path, index=False)
    print(f"MSE scores saved: {mse_path}")

    n_triggered = sum(1 for t in transfer_triggers if t)
    print(f"\nPer-step snapshot models loaded on {n_triggered}/{len(transfer_triggers)} steps "
          f"({n_triggered / max(len(transfer_triggers), 1) * 100:.1f}%)")

    print("\nTransfer learning test completed successfully (AEGeneRemove ablation, wq-only)!")
    return {
        'predictions': predictions,
        'real_values': real_values,
        'fusion_vectors': fusion_vectors,
        'mse_scores': mse_scores,
        'transfer_triggers': transfer_triggers,
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    test_gene_model_transfer_learning()
