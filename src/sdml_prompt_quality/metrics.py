import torch
import numpy as np
from sklearn.metrics import r2_score
import logging

logger = logging.getLogger(__name__)


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
    
    # Log detailed statistics
    logger.info(f"Predictions - shape: {pred.shape}, min: {pred.min():.6f}, max: {pred.max():.6f}, mean: {pred.mean():.6f}, std: {pred.std():.6f}")
    logger.info(f"Targets - shape: {target.shape}, min: {target.min():.6f}, max: {target.max():.6f}, mean: {target.mean():.6f}, std: {target.std():.6f}")
    
    # Check for NaN/Inf in predictions
    pred_nan_count = np.isnan(pred).sum()
    pred_inf_count = np.isinf(pred).sum()
    if pred_nan_count > 0:
        logger.error(f"Found {pred_nan_count} NaN values in predictions!")
        logger.error(f"NaN indices: {np.where(np.isnan(pred))[0][:20]}")  # Show first 20 indices
    if pred_inf_count > 0:
        logger.error(f"Found {pred_inf_count} Inf values in predictions!")
        logger.error(f"Inf indices: {np.where(np.isinf(pred))[0][:20]}")
    
    # Check for NaN/Inf in targets
    target_nan_count = np.isnan(target).sum()
    target_inf_count = np.isinf(target).sum()
    if target_nan_count > 0:
        logger.error(f"Found {target_nan_count} NaN values in targets!")
        logger.error(f"NaN indices: {np.where(np.isnan(target))[0][:20]}")
    if target_inf_count > 0:
        logger.error(f"Found {target_inf_count} Inf values in targets!")
        logger.error(f"Inf indices: {np.where(np.isinf(target))[0][:20]}")
    
    mae = abs(pred - target).mean()
    mse = ((pred - target) ** 2).mean()
    
    logger.info(f"MAE: {mae:.6f}, MSE: {mse:.6f}")
    
    r2 = r2_score(target, pred)
    return mae, mse, r2
