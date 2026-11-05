"""
test_gene_model_transfer_learning.py - GeneModel per-step transfer learning (EWC)
=========================================================
Purpose:
    simulates an online / incremental setting by splitting the test segment chronologically into several transfer steps and fine-tuning step by step:
    1) Compute the Fisher information matrix and the parameter snapshot theta* on the full training data as the EWC reference
    2) Whenever a new window arrives, fine-tune for transfer_epochs epochs on that window,
       loss = task loss + EWC_LAMBDA x EWC penalty (keeps the important parameters from drifting)
    3) Save a checkpoint at every transfer step together with that step's predictions and fusion vectors,
       consumed by the evolution analyses in results4_* (SHAP/PDP) and results5~9
Key parameters:
    EWC_LAMBDA = 1000, FISHER_N_BATCHES = 10, FISHER_BATCH_SIZE = 32
    transfer_window = 40 (data window length of each fine-tune), transfer_epochs = 30
Input:
    model/gene_model/best_gene_model.pth (source-domain pretrained weights, actually hidden_dim=64)
Output:
    model/gene_model/model_after_transfer_step_{i}.pth   weights of each transfer step
    results/gene_model_transfer_learning/prediction.csv  predictions after transfer learning
    results/gene_model_transfer_learning/*.png           regression plot and time series plot
Usage:
    cd step1_gene_model_trans_2block_FullTL && python test_gene_model_transfer_learning.py
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from gene_model import GeneModel, WaterQualityDataset, prepare_data, calculate_loss
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

# ===== EWC setup (FullTL: nothing frozen, all parameters participate in transfer learning) =====
EWC_LAMBDA = 1000           # EWC penalty coefficient
FISHER_N_BATCHES = 10       # number of batches used for the Fisher computation
FISHER_BATCH_SIZE = 32      # batch size used for the Fisher computation


def _forward_one(model, inf, ana_ano, ox, eff,
                 gn_aa, gn_ox, gc_aa, gc_ox, device):
    """Single-sample inference wrapper: the 8 feature tensors of dataset[i] -> model output dict"""
    return model(
        inf.unsqueeze(0).to(device),
        ana_ano.unsqueeze(0).to(device),
        ox.unsqueeze(0).to(device),
        eff.unsqueeze(0).to(device),
        gn_aa.unsqueeze(0).to(device),
        gn_ox.unsqueeze(0).to(device),
        gc_aa.unsqueeze(0).to(device),
        gc_ox.unsqueeze(0).to(device),
    )


def compute_fisher_and_star(model, dataset, device):
    """Empirical Fisher (diagonal): F_i ~= E[(dL/dtheta_i)^2], theta* = current parameter snapshot.

    FullTL: every parameter is trainable, so Fisher and theta* cover all parameters,
    sampled from the training segment (the first 30%).
    """
    fisher = {n: torch.zeros_like(p, device='cpu')
              for n, p in model.named_parameters() if p.requires_grad}
    theta_star = {n: p.detach().clone().cpu()
                  for n, p in model.named_parameters() if p.requires_grad}

    n_train = int(0.3 * len(dataset))
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_train)
    sample_indices = perm[:FISHER_N_BATCHES * FISHER_BATCH_SIZE].tolist()

    sub = Subset(dataset, sample_indices)
    loader = DataLoader(sub, batch_size=FISHER_BATCH_SIZE, shuffle=False)

    # Note: cuDNN RNN backward can only run in train mode (eval mode raises
    # "cudnn RNN backward can only be called in training mode"), 
    # so Fisher computation keeps model.train(); the model has no BatchNorm, only Dropout is active,
    # Averaging the Fisher estimate over 10 batches has a negligible effect.
    model.train()
    seen = 0
    for batch in loader:
        if seen >= FISHER_N_BATCHES:
            break
        inf, ana_ano, ox, eff, gn_aa, gn_ox, gc_aa, gc_ox, labels = batch
        inf = inf.to(device)
        ana_ano = ana_ano.to(device)
        ox = ox.to(device)
        eff = eff.to(device)
        gn_aa = gn_aa.to(device)
        gn_ox = gn_ox.to(device)
        gc_aa = gc_aa.to(device)
        gc_ox = gc_ox.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        outputs = model(inf, ana_ano, ox, eff, gn_aa, gn_ox, gc_aa, gc_ox)
        # Sum of the four MSEs used as the Fisher objective
        loss = sum(((outputs[k] - labels[k]) ** 2).mean()
                   for k in ['ANA', 'ANO', 'OX', 'EFF'])
        model.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None and n in fisher:
                fisher[n] += (p.grad.detach().cpu() ** 2)
        seen += 1

    for n in fisher:
        fisher[n] /= max(seen, 1)
    print(f"  Fisher computed over {seen} batches x {FISHER_BATCH_SIZE} samples.")
    return fisher, theta_star


def ewc_penalty(model, fisher, theta_star):
    """EWC penalty: lambda/2 * sum F_i * (theta_i - theta*_i)^2 (trainable parameters only)"""
    pen = 0.0
    for n, p in model.named_parameters():
        if n in fisher and p.requires_grad:
            f = fisher[n].to(p.device)
            ts = theta_star[n].to(p.device)
            pen = pen + (f * (p - ts) ** 2).sum()
    return EWC_LAMBDA / 2.0 * pen


def test_gene_model_transfer_learning():
    # ===== 1. Prepare data =====
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

    # ===== 2. Load the pretrained model =====
    print("\n" + "=" * 60)
    print("Step 2: Loading pre-trained model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Note: a checkpoint check shows best_gene_model.pth is actually hidden_dim=64
    # (lstm.weight_ih_l0=(256,19)=4*64, output_ana.weight=(3,64),
    #  gene_c_proj.0.weight=(128,1205)=2*64)
    model = GeneModel(hidden_dim=64, dropout_rate=0.3).to(device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, 'model', 'gene_model', 'best_gene_model.pth')
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Pre-trained model loaded from: {pretrained_path}")

    # ===== 3. FullTL: every parameter takes part in transfer learning (nothing frozen) =====
    print("\n" + "=" * 60)
    print("Step 3: Enabling all parameters for training (FullTL, no freeze)...")

    for name, p in model.named_parameters():
        p.requires_grad = True

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={n_total:,} | trainable={n_trainable:,} (all trainable, no freeze)")

    # ===== 4. EWC: compute the Fisher matrix and theta* =====
    print("\n" + "=" * 60)
    print("Step 4: Computing EWC Fisher information and reference parameters...")
    fisher, theta_star = compute_fisher_and_star(model, dataset, device)

    # ===== 5. Hyperparameters (identical to LSTM trans) =====
    window_size = 40           # Bollinger history window
    transfer_window = 40       # fine-tune on the latest 40 steps
    transfer_epochs = 30       # epochs per trigger
    transfer_batch_size = 64
    lr = 5e-5
    weight_decay = 1e-5

    # FullTL: fine-tune every parameter with the EWC penalty added
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    print(f"EWC full fine-tuning | lambda={EWC_LAMBDA} | lr={lr} | weight_decay={weight_decay} "
          f"| epochs/trigger={transfer_epochs} | batch={transfer_batch_size}")

    # ===== 6. Result containers =====
    predictions = {"ANA": [], "ANO": [], "OX": [], "EFF": []}
    real_values = {"ANA": [], "ANO": [], "OX": [], "EFF": []}
    bollinger_bands = []
    fusion_vectors = {"fusion_ana_ano": [], "fusion_ox": []}
    transfer_triggers = []

    snapshot_dir = os.path.join(current_dir, 'model', 'gene_model_transfer_learning')
    os.makedirs(snapshot_dir, exist_ok=True)

    # ===== 7. Per-step loop (Bollinger-triggered transfer learning) =====
    print("\n" + "=" * 60)
    print("Step 5: Per-step transfer learning testing (EWC, full fine-tuning)...")

    for i in range(start_idx, len(dataset)):
        inf_feat, ana_ano_feat, ox_feat, eff_feat, \
            gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, labels = dataset[i]

        # ---- 7.1 Compute historical errors and the Bollinger band ----
        current_error = 0.0
        if i >= window_size:
            recent_errors = []
            for j in range(max(0, i - window_size), i):
                j_inf, j_ana_ano, j_ox, j_eff, j_g_n_aa, j_g_n_ox, j_g_c_aa, j_g_c_ox, j_labels = dataset[j]
                model.eval()
                with torch.no_grad():
                    j_out = _forward_one(model, j_inf, j_ana_ano, j_ox, j_eff,
                                         j_g_n_aa, j_g_n_ox, j_g_c_aa, j_g_c_ox, device)
                err = 0.0
                for k in ["ANA", "ANO", "OX", "EFF"]:
                    err += np.mean(np.abs(
                        j_out[k].cpu().numpy().squeeze() - j_labels[k].numpy().squeeze()))
                recent_errors.append(err)

            if len(recent_errors) >= window_size:
                recent_errors = recent_errors[-window_size:]
                mean_e = float(np.mean(recent_errors))
                std_e = float(np.std(recent_errors))
                upper_band = mean_e + 3 * std_e
                middle_band = mean_e
                lower_band = mean_e - 3 * std_e
                bollinger_bands.append({
                    "time_step": i,
                    "upper_band": upper_band,
                    "middle_band": middle_band,
                    "lower_band": lower_band,
                })

                # Predict the current step and compute current_error
                model.eval()
                with torch.no_grad():
                    cur_out = _forward_one(model, inf_feat, ana_ano_feat, ox_feat, eff_feat,
                                           gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, device)
                for k in ["ANA", "ANO", "OX", "EFF"]:
                    current_error += np.mean(np.abs(
                        cur_out[k].cpu().numpy().squeeze() - labels[k].numpy().squeeze()))

        # ---- 7.2 Trigger decision (same threshold semantics as LSTM trans) ----
        need_transfer = False
        if i >= window_size and len(bollinger_bands) > 0:
            bb = bollinger_bands[-1]
            threshold = bb["lower_band"] + 0.005 * (bb["middle_band"] - bb["lower_band"])
            if current_error > threshold:
                need_transfer = True

        # ---- 7.3 Trigger transfer learning: fine-tune 30 epochs on the latest 40 steps (loss includes the EWC penalty) ----
        if need_transfer or i < window_size:
            tl_start = max(0, i - transfer_window)  # avoid shadowing the outer start_idx
            transfer_indices = list(range(tl_start, i))
            if len(transfer_indices) > 0:
                transfer_dataset = Subset(dataset, transfer_indices)
                transfer_loader = DataLoader(transfer_dataset,
                                             batch_size=transfer_batch_size, shuffle=False)

                model.train()
                for epoch in range(transfer_epochs):
                    for batch_idx, (t_inf, t_ana_ano, t_ox, t_eff,
                                    t_g_n_aa, t_g_n_ox, t_g_c_aa, t_g_c_ox,
                                    t_labels) in enumerate(transfer_loader):
                        t_inf = t_inf.to(device)
                        t_ana_ano = t_ana_ano.to(device)
                        t_ox = t_ox.to(device)
                        t_eff = t_eff.to(device)
                        t_g_n_aa = t_g_n_aa.to(device)
                        t_g_n_ox = t_g_n_ox.to(device)
                        t_g_c_aa = t_g_c_aa.to(device)
                        t_g_c_ox = t_g_c_ox.to(device)
                        t_labels = {k: v.to(device) for k, v in t_labels.items()}

                        outputs = model(t_inf, t_ana_ano, t_ox, t_eff,
                                        t_g_n_aa, t_g_n_ox, t_g_c_aa, t_g_c_ox)
                        loss_dict = calculate_loss(outputs, t_labels)
                        loss = loss_dict['total'] + ewc_penalty(model, fisher, theta_star)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

        # ---- 7.4 Save a model snapshot when a trigger fires ----
        if need_transfer:
            mp = os.path.join(snapshot_dir, f"model_after_transfer_step_{i+1}.pth")
            torch.save(model.state_dict(), mp)

        transfer_triggers.append(need_transfer)

        # ---- 7.5 Predict the current step (with the fine-tuned / current model) ----
        model.eval()
        with torch.no_grad():
            outputs = _forward_one(model, inf_feat, ana_ano_feat, ox_feat, eff_feat,
                                   gene_n_ana_ano, gene_n_ox, gene_c_ana_ano, gene_c_ox, device)

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
              f"transfer triggered: {need_transfer}")

    # ===== 8. Write results to disk =====
    print("\n" + "=" * 60)
    print("Step 6: Saving results...")

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

    # bollinger_bands.csv
    if bollinger_bands:
        bb_path = os.path.join(result_dir, 'bollinger_bands.csv')
        pd.DataFrame(bollinger_bands).to_csv(bb_path, index=False)
        print(f"Bollinger Bands saved: {bb_path}")

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
        ax.set_title(f'{k} Time Series Prediction (Transfer Learning, EWC Full Fine-tuning)')
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
    print(f"\nTransfer triggered on {n_triggered}/{len(transfer_triggers)} steps "
          f"({n_triggered / max(len(transfer_triggers), 1) * 100:.1f}%)")

    print("\nTransfer learning test completed successfully (EWC, all params fine-tuned)!")
    return {
        'predictions': predictions,
        'real_values': real_values,
        'fusion_vectors': fusion_vectors,
        'mse_scores': mse_scores,
        'bollinger_bands': bollinger_bands,
        'transfer_triggers': transfer_triggers,
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    test_gene_model_transfer_learning()
