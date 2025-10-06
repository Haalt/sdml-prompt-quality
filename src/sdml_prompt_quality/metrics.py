import torch
import numpy as np
from sklearn.metrics import r2_score


def collate(batch):
    # stack every field along dim=0
    keys = batch[0].keys()
    collated = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    return collated


def batch_mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def epoch_metrics(all_pred, all_target):
    pred = torch.cat(all_pred).cpu().numpy()
    target = torch.cat(all_target).cpu().numpy()
    mae = abs(pred - target).mean()
    mse = ((pred - target) ** 2).mean()
    r2 = r2_score(target, pred)
    return mae, mse, r2
