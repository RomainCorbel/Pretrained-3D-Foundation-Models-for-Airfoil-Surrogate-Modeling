import torch
import numpy as np
import os.path as osp
from torch_geometric.loader import DataLoader
import random
import torch_geometric.nn as nng
import json
from dataset import Dataset
import os

def Results_test(device, model, hparams, coef_norm, path_in='../Dataset', path_out='scores',
                 s='full_test', test_dataset=None):

    # ------------------------------------------------------
    # Build test dataset ONCE (no subsampling)
    # ------------------------------------------------------
    test_dataset_sampled = []

    use_edges = ('r' in hparams and hparams['r'] is not None)

    for data in test_dataset:
        data_sampled = data.clone()

        # Build edges once if needed
        if use_edges:
            data_sampled.edge_index = nng.radius_graph(
                x=data_sampled.pos.to(device),
                r=hparams['r'],
                loop=True,
                max_num_neighbors=int(hparams['max_neighbors'])
            ).to(device)

        test_dataset_sampled.append(data_sampled)

    test_loader = DataLoader(test_dataset_sampled, batch_size=1, shuffle=False)

    # ------------------------------------------------------
    # Run inference
    # ------------------------------------------------------
    true_list = []
    pred_list = []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            pred = model(data)
            y = data.y

            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            if y.dim() == 1:
                y = y.unsqueeze(1)

            m_surf = data.surf
            if m_surf.any():
                pred_list.append(pred[m_surf].cpu().numpy())
                true_list.append(y[m_surf].cpu().numpy())

    true_norm = np.concatenate(true_list)
    pred_norm = np.concatenate(pred_list)
    # ----------------------
    # Test loss (MSE)
    # ----------------------
    test_mse = np.mean((pred_norm - true_norm) ** 2)
    print(f"[TEST] MSE (normalized): {test_mse:.6e}")

    # ----------------------
    # De-normalization
    # ----------------------
    mean_x, std_x, mean_y, std_y, mean_g, std_g = coef_norm

    mean_y = np.array(mean_y).reshape(1,)
    std_y  = np.array(std_y).reshape(1,)

    true_denorm = true_norm * std_y + mean_y
    pred_denorm = pred_norm * std_y + mean_y

    return true_norm, pred_norm, true_denorm, pred_denorm, float(test_mse)