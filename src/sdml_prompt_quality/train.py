from multiprocessing import freeze_support
import argparse
import os
import json
import copy
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

logger = logging.getLogger(__name__)

from .data.datasets import PromptDataset
from .models.v3 import PromptQualityModelV3
from .metrics import collate, epoch_metrics
from .data.preprocess import preprocess
from .data.load_dataset import load_dataset_model_scores, split_dataset

# TOKENIZER_FILE = str(Path(__file__).parent / "tokenizer" / "weighted_tokenizer.json")
# TOKENIZER_FILE = str(Path(__file__).parent / "tokenizer" / "xl_tokenizer.json")
TOKENIZER_FILE = str(Path(__file__).parent / "tokenizer" / "combined_tokenizer.json")


def _set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def _build_param_groups_stage(model, stage: str):
    groups = []
    if stage == "A":
        groups.append(
            {"params": [model.sampler_embed.weight, model.upsc_embed.weight], "lr": 5e-3})
        groups.append({"params": [model.token_embed.weight], "lr": 1e-3})
        groups.append(
            {"params": [model.head[-1].weight, model.head[-1].bias], "lr": 1e-3})
    elif stage == "B":
        groups.append(
            {"params": [model.sampler_embed.weight, model.upsc_embed.weight], "lr": 5e-4})
        groups.append({"params": [model.token_embed.weight], "lr": 5e-4})
        groups.append(
            {"params": [*model.samp_fuse.parameters(), *model.up_fuse.parameters()], "lr": 5e-4})
        groups.append(
            {"params": [p for p in model.head.parameters()], "lr": 5e-4})
    elif stage == "C":
        groups.append(
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-4})
    else:
        # Lower base LR for stability
        groups.append(
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-4})
    return groups


def _freeze_for_stage(model, stage: str, freeze_bucket_embed: bool):
    _set_requires_grad(model, False)
    _set_requires_grad(model.lora_embed, False)
    _set_requires_grad(model.lora_fuse, False)

    if stage == "A":
        _set_requires_grad(model.token_embed, True)
        _set_requires_grad(model.sampler_embed, True)
        _set_requires_grad(model.upsc_embed, True)
        model.head[-1].weight.requires_grad = True
        model.head[-1].bias.requires_grad = True
        if not freeze_bucket_embed:
            _set_requires_grad(model.bucket_embed, True)
    elif stage == "B":
        _set_requires_grad(model.token_embed, True)
        _set_requires_grad(model.sampler_embed, True)
        _set_requires_grad(model.upsc_embed, True)
        _set_requires_grad(model.samp_fuse, True)
        _set_requires_grad(model.up_fuse, True)
        _set_requires_grad(model.head, True)
        if not freeze_bucket_embed:
            _set_requires_grad(model.bucket_embed, True)
    elif stage == "C":
        _set_requires_grad(model, True)
        _set_requires_grad(model.lora_embed, False)
        _set_requires_grad(model.lora_fuse, False)
        if freeze_bucket_embed:
            _set_requires_grad(model.bucket_embed, False)


def train_model(
    train_samples,
    val_samples,
    vocab_tokens,
    vocab_loras,
    vocab_samplers,
    vocab_upscalers,
    num_epochs=150,
    patience=8,
    batch_size=256,
    lr=1e-3,
    hidden_dim=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
    init_state: str = None,
    finetune: bool = False,
    epochs_a: int = 6,
    epochs_b: int = 10,
    epochs_c: int = 0,
    l2sp_lambda: float = 0.0,
    freeze_bucket_embed: bool = True,
):

    # data loaders
    train_loader = DataLoader(
        PromptDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PromptDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    # model = PromptQualityModelV2(
    #     vocab_tokens=vocab_tokens,
    #     vocab_loras=vocab_loras,
    #     d_t=256,
    #     d_L=128,
    #     head_h1=512,
    #     #    n_heads=8,
    #     dropout_p=0.4,
    # ).to(device)

    model = PromptQualityModelV3(
        vocab_tokens=vocab_tokens,
        vocab_loras=vocab_loras,
        vocab_samplers=vocab_samplers,
        vocab_upscalers=vocab_upscalers,
        bucket_size=20,
        d_t=256,
        d_L=128,
        head_h1=512,
        dropout_p=0.2,
    ).to(device)

    # model = PromptQualityModelV6(
    #     vocab_tokens=vocab_tokens,
    #     vocab_loras=vocab_loras,
    #     vocab_samplers=vocab_samplers,
    #     vocab_upscalers=vocab_upscalers,
    #     bucket_size=20,
    #     d_t=384,
    #     d_L=128,
    #     head_h1=768,
    #     dropout_p=0.4,
    #     n_layers=2,
    #     n_heads=8,
    #     d_ff=1536
    # ).to(device)

    # model = PromptQualityModelV5(
    #     vocab_tokens=vocab_tokens,
    #     vocab_loras=vocab_loras,
    #     vocab_samplers=vocab_samplers,
    #     vocab_upscalers=vocab_upscalers,
    #     bucket_size=20,
    #     d_t=256,
    #     d_L=128,
    #     head_h1=512,
    #     dropout_p=0.5,
    # ).to(device)

    # model = PromptQualityModelV3_6(
    #     vocab_tokens=vocab_tokens,
    #     vocab_loras=vocab_loras,
    #     vocab_samplers=vocab_samplers,
    #     vocab_upscalers=vocab_upscalers,
    #     d_token_hidden=256,
    #     d_t=256,
    #     d_L=128,
    #     head_h1=512,
    #     dropout_p=0.15,
    # ).to(device)

    # model = PromptQualityModelV3_5(
    #     vocab_tokens=vocab_tokens,
    #     vocab_loras=vocab_loras,
    #     vocab_samplers=vocab_samplers,
    #     vocab_upscalers=vocab_upscalers,
    #     d_token_hidden=256,
    #     d_t=256,
    #     d_L=128,
    #     head_h1=512,
    #     dropout_p=0.15,
    # ).to(device)

    # model = PromptQualityModelV5(
    #     vocab_tokens=vocab_tokens,
    #     vocab_loras=vocab_loras,
    #     vocab_samplers=vocab_samplers,
    #     vocab_upscalers=vocab_upscalers,
    #     bucket_size=20,
    #     d_t=512,
    #     d_L=128,
    #     head_h1=512,
    #     dropout_p=0.2,
    # ).to(device)

    # model = PromptQualityModelV4(
    #     vocab_tokens=vocab_tokens,
    #     vocab_loras=vocab_loras,
    #     vocab_samplers=vocab_samplers,
    #     vocab_upscalers=vocab_upscalers,
    #     d_t=256,
    #     d_L=128,
    #     head_h1=512,
    #     dropout_p=0.1,
    # ).to(device)

    l2sp_anchor = None
    if init_state is not None and os.path.isfile(init_state):
        state = torch.load(init_state, map_location="cpu")
        model.load_state_dict(state, strict=False)
        if l2sp_lambda > 0.0:
            l2sp_anchor = {k: v.to(device)
                           for k, v in model.state_dict().items()}

    print("train set size:", len(train_loader.dataset))
    print("val   set size:", len(val_loader.dataset))

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss()

    best_val_mse = float("inf")
    best_state = None
    epochs_no_improve = 0

    saved_train_mae = float("inf")
    saved_train_mse = float("inf")
    saved_val_mse = float("inf")
    saved_val_mae = float("inf")
    saved_val_r2 = float("inf")

    def run_stage(stage_name: str, stage_epochs: int, optimiser, scheduler):
        nonlocal best_val_mse, best_state, epochs_no_improve
        nonlocal saved_train_mae, saved_train_mse, saved_val_mse, saved_val_mae, saved_val_r2

        for _ in range(stage_epochs):
            model.train()
            train_mae = 0.0
            train_mse = 0.0
            train_samples = 0
            for batch_idx, batch in enumerate(train_loader):
                batch = {k: v.to(device, non_blocking=True)
                         for k, v in batch.items()}
                optimiser.zero_grad()
                logits = model(
                    batch["tokens"],
                    batch["token_mask"],
                    batch["lora_ids"],
                    batch["lora_w"],
                    batch["cfg"],
                    batch["n_loras"],
                    batch["sampler_id"],
                    batch["steps_log"],
                    batch["steps_bucket"],
                    batch["upscaler_id"],
                    batch["up_has"],
                    batch["up_steps"],
                    batch["denoise"],
                    batch["model_id"]
                )
                
                # Check logits for NaN/Inf during training
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.error(f"NaN/Inf detected in logits at training batch {batch_idx}!")
                    logger.error(f"  Logits stats - min: {logits.min().item():.6f}, max: {logits.max().item():.6f}, mean: {logits.mean().item():.6f}")
                    if torch.isnan(logits).any():
                        logger.error(f"  NaN count: {torch.isnan(logits).sum().item()} / {logits.numel()}")
                    if torch.isinf(logits).any():
                        logger.error(f"  Inf count: {torch.isinf(logits).sum().item()} / {logits.numel()}")
                
                # Removed sigmoid here since targets are now normalized (unbounded), not 0-1 probabilities
                y_hat = torch.sigmoid(logits) 
                # y_hat = logits
                
                # Check predictions for NaN/Inf during training
                if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
                    logger.error(f"NaN/Inf detected in predictions at training batch {batch_idx}!")
                    logger.error(f"  Predictions stats - min: {y_hat.min().item():.6f}, max: {y_hat.max().item():.6f}, mean: {y_hat.mean().item():.6f}")
                
                # Changed from BCEWithLogitsLoss (for classification/0-1 regression) to MSELoss (for general regression)
                loss = criterion(logits, batch["target"].squeeze(1))
                # loss = F.mse_loss(y_hat, batch["target"].squeeze(1))

                # Weighted MSE Loss
                # loss = (y_hat - batch["target"].squeeze(1)) ** 2
                # loss = (loss * batch["weight"].squeeze(1)).mean()
                
                # Check loss for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf detected in loss at training batch {batch_idx}!")
                    logger.error(f"  Loss value: {loss.item()}")
                    logger.error(f"  Target stats - min: {batch['target'].min().item():.6f}, max: {batch['target'].max().item():.6f}, mean: {batch['target'].mean().item():.6f}")
                
                if l2sp_anchor is not None and l2sp_lambda > 0.0:
                    l2sp = 0.0
                    for n, p in model.named_parameters():
                        if p.requires_grad and n in l2sp_anchor:
                            l2sp = l2sp + torch.sum((p - l2sp_anchor[n]) ** 2)
                    loss = loss + l2sp_lambda * l2sp
                loss.backward()
                
                # Clip gradients to prevent exploding gradients and NaN propagation
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimiser.step()

                train_mae += F.l1_loss(y_hat,
                                       batch["target"].squeeze(1), reduction="sum").item()
                train_mse += F.mse_loss(y_hat,
                                        batch["target"].squeeze(1), reduction="sum").item()
                train_samples += batch["target"].shape[0]

            model.eval()
            all_pred, all_tgt = [], []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    batch = {k: v.to(device, non_blocking=True)
                             for k, v in batch.items()}
                    
                    # Check inputs for NaN/Inf before model forward pass
                    for key, value in batch.items():
                        if torch.isnan(value).any():
                            logger.error(f"NaN detected in input '{key}' at validation batch {batch_idx}!")
                            logger.error(f"  {key} - NaN count: {torch.isnan(value).sum().item()} / {value.numel()}")
                            nan_indices = torch.where(torch.isnan(value))
                            logger.error(f"  NaN at indices: {nan_indices}")
                        if torch.isinf(value).any():
                            logger.error(f"Inf detected in input '{key}' at validation batch {batch_idx}!")
                            logger.error(f"  {key} - Inf count: {torch.isinf(value).sum().item()} / {value.numel()}")
                    
                    logits = model(
                        batch["tokens"],
                        batch["token_mask"],
                        batch["lora_ids"],
                        batch["lora_w"],
                        batch["cfg"],
                        batch["n_loras"],
                        batch["sampler_id"],
                        batch["steps_log"],
                        batch["steps_bucket"],
                        batch["upscaler_id"],
                        batch["up_has"],
                        batch["up_steps"],
                        batch["denoise"],
                        batch["model_id"]
                    )
                    
                    # Check logits for NaN/Inf before sigmoid
                    if torch.isnan(logits).any():
                        nan_mask = torch.isnan(logits)
                        nan_sample_indices = torch.where(nan_mask)[0]
                        logger.error(f"NaN detected in logits at validation batch {batch_idx}!")
                        logger.error(f"  Logits stats - min: {logits[~nan_mask].min().item() if (~nan_mask).any() else 'all NaN':.6f}, max: {logits[~nan_mask].max().item() if (~nan_mask).any() else 'all NaN':.6f}, mean: {logits[~nan_mask].mean().item() if (~nan_mask).any() else 'all NaN':.6f}")
                        logger.error(f"  NaN count: {torch.isnan(logits).sum().item()} / {logits.numel()}")
                        logger.error(f"  Sample indices in batch with NaN: {nan_sample_indices.cpu().tolist()}")
                        
                        # Log the problematic inputs
                        for idx in nan_sample_indices.cpu().tolist()[:3]:  # Log first 3 problematic samples
                            logger.error(f"  --- Problematic sample {idx} in batch ---")
                            logger.error(f"    tokens: {batch['tokens'][idx].cpu().tolist()[:20]}...")  # First 20 tokens
                            logger.error(f"    token_mask: {batch['token_mask'][idx].cpu().tolist()[:20]}...")
                            logger.error(f"    lora_ids: {batch['lora_ids'][idx].cpu().tolist()}")
                            logger.error(f"    lora_w: {batch['lora_w'][idx].cpu().tolist()}")
                            logger.error(f"    cfg: {batch['cfg'][idx].item():.6f}")
                            logger.error(f"    n_loras: {batch['n_loras'][idx].item()}")
                            logger.error(f"    sampler_id: {batch['sampler_id'][idx].item()}")
                            logger.error(f"    steps_log: {batch['steps_log'][idx].item():.6f}")
                            logger.error(f"    steps_bucket: {batch['steps_bucket'][idx].item()}")
                            logger.error(f"    upscaler_id: {batch['upscaler_id'][idx].item()}")
                            logger.error(f"    up_has: {batch['up_has'][idx].item()}")
                            logger.error(f"    up_steps: {batch['up_steps'][idx].item():.6f}")
                            logger.error(f"    denoise: {batch['denoise'][idx].item():.6f}")
                            logger.error(f"    model_id: {batch['model_id'][idx].item()}")
                            logger.error(f"    target: {batch['target'][idx].item():.6f}")
                    if torch.isinf(logits).any():
                        logger.error(f"Inf detected in logits at validation batch {batch_idx}!")
                        logger.error(f"  Inf count: {torch.isinf(logits).sum().item()} / {logits.numel()}")
                    
                    y_hat = torch.sigmoid(logits)
                    # y_hat = logits
                    
                    # Check predictions for NaN/Inf after sigmoid
                    if torch.isnan(y_hat).any():
                        logger.error(f"NaN detected in predictions at validation batch {batch_idx}!")
                        logger.error(f"  Predictions stats - min: {y_hat.min().item():.6f}, max: {y_hat.max().item():.6f}, mean: {y_hat.mean().item():.6f}")
                        logger.error(f"  NaN count: {torch.isnan(y_hat).sum().item()} / {y_hat.numel()}")
                    if torch.isinf(y_hat).any():
                        logger.error(f"Inf detected in predictions at validation batch {batch_idx}!")
                        logger.error(f"  Inf count: {torch.isinf(y_hat).sum().item()} / {y_hat.numel()}")
                    
                    # Check targets for NaN/Inf
                    targets = batch["target"].squeeze(1)
                    if torch.isnan(targets).any():
                        logger.error(f"NaN detected in targets at validation batch {batch_idx}!")
                        logger.error(f"  Target stats - min: {targets.min().item():.6f}, max: {targets.max().item():.6f}, mean: {targets.mean().item():.6f}")
                        logger.error(f"  NaN count: {torch.isnan(targets).sum().item()} / {targets.numel()}")
                    if torch.isinf(targets).any():
                        logger.error(f"Inf detected in targets at validation batch {batch_idx}!")
                        logger.error(f"  Inf count: {torch.isinf(targets).sum().item()} / {targets.numel()}")
                    
                    all_pred.append(y_hat)
                    all_tgt.append(targets)

            val_mae, val_mse, val_r2 = epoch_metrics(all_pred, all_tgt)
            y_true_cont = torch.cat(all_tgt).cpu().numpy().ravel()
            y_scores = torch.cat(all_pred).cpu().numpy().ravel()
            
            # Use 0.0 as threshold since data is Z-score normalized
            # thr = 0.0 
            thr = float(np.median(y_true_cont))
            
            y_true_bin = (y_true_cont >= thr).astype(int)
            if np.unique(y_true_bin).size >= 2:
                val_auc = roc_auc_score(y_true_bin, y_scores)
            else:
                val_auc = float("nan")
            if scheduler is not None:
                # scheduler.step()
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_mse)
                else:
                    scheduler.step()

            train_mae_avg = train_mae / \
                train_samples if train_samples > 0 else float("nan")
            train_mse_avg = train_mse / \
                train_samples if train_samples > 0 else float("nan")

            print(
                f"Stage {stage_name} | train_MSE={train_mse_avg:.5f}  "
                f"train_MAE={train_mae_avg:.5f}  val_MSE={val_mse:.5f}  "
                f"val_MAE={val_mae:.5f}  val_R2={val_r2:.4f}  val_AUROC={val_auc:.5f}"
            )
            print("LR:", optimiser.param_groups[0]["lr"])

            if val_mse + 1e-6 < best_val_mse:
                best_val_mse = val_mse
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                torch.save(best_state, "best_model.pt")
                print("  ↳ new best model saved.")
                saved_train_mae = train_mae_avg
                saved_train_mse = train_mse_avg
                saved_val_mse = val_mse
                saved_val_mae = val_mae
                saved_val_r2 = val_r2
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping in Stage {stage_name} after patience exhausted. Best val_MSE={best_val_mse:.5f}")
                    return True
        return False

    if finetune:
        _freeze_for_stage(model, "A", freeze_bucket_embed=freeze_bucket_embed)
        pg = _build_param_groups_stage(model, "A")
        optimiser = optim.AdamW(pg, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max(1, epochs_a))
        stopped = run_stage("A", epochs_a, optimiser, scheduler)
        if not stopped and epochs_b > 0:
            _freeze_for_stage(
                model, "B", freeze_bucket_embed=freeze_bucket_embed)
            pg = _build_param_groups_stage(model, "B")
            optimiser = optim.AdamW(pg, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=max(1, epochs_b))
            stopped = run_stage("B", epochs_b, optimiser, scheduler)
        if not stopped and epochs_c > 0:
            _freeze_for_stage(
                model, "C", freeze_bucket_embed=freeze_bucket_embed)
            pg = _build_param_groups_stage(model, "C")
            optimiser = optim.AdamW(pg, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=max(1, epochs_c))
            run_stage("C", epochs_c, optimiser, scheduler)
    else:
        pg = _build_param_groups_stage(model, "base")
        # Increase weight decay for stronger regularization
        optimiser = optim.AdamW(pg, weight_decay=1e-3, lr=1e-4)
        
        # Use ReduceLROnPlateau for adaptive decay
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        
        # warm_up = torch.optim.lr_scheduler.LinearLR(
        #     optimiser, start_factor=1e-6, end_factor=1.0, total_iters=5)
        # cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimiser, T_max=num_epochs - 3)
        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimiser, schedulers=[warm_up, cosine], milestones=[3])
        run_stage("Base", num_epochs, optimiser, scheduler)

    # load & return the best model
    model.load_state_dict(best_state)
    return (
        model,
        saved_train_mae,
        saved_train_mse,
        saved_val_mse,
        saved_val_mae,
        saved_val_r2,
    )


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(
        description="Train V3 with optional staged fine-tuning")
    parser.add_argument("--init_state", type=str, default=None,
                        help="Path to initialization state_dict (e.g., from transplant)")
    parser.add_argument("--finetune", action="store_true",
                        help="Enable staged fine-tuning (A/B/C)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Total epochs for base training when not finetuning")
    parser.add_argument("--epochs_a", type=int,
                        default=6, help="Stage A epochs")
    parser.add_argument("--epochs_b", type=int,
                        default=10, help="Stage B epochs")
    parser.add_argument("--epochs_c", type=int,
                        default=0, help="Stage C epochs")
    parser.add_argument("--patience", type=int, default=12,
                        help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--l2sp", type=float, default=0.0,
                        help="L2-SP regularization strength")
    parser.add_argument("--no_freeze_bucket", action="store_true",
                        help="Do not freeze bucket_embed in Stage A/B")
    args = parser.parse_args()

    dataset_input, dataset_output = load_dataset_model_scores(combine=True, normalize=True)
    # (train_samples, val_samples) = split_dataset(dataset_input, dataset_output)

    (train_samples, val_samples, vocab_tokens, vocab_loras,
     vocab_samplers, vocab_upscalers) = preprocess(
        dataset_input, dataset_output, TOKENIZER_FILE
    )

    best_model, _, _, _, _, _ = train_model(
        train_samples,
        val_samples,
        vocab_tokens=vocab_tokens,
        vocab_loras=vocab_loras,
        vocab_samplers=vocab_samplers,
        vocab_upscalers=vocab_upscalers,
        num_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        init_state=args.init_state,
        finetune=args.finetune,
        epochs_a=args.epochs_a,
        epochs_b=args.epochs_b,
        epochs_c=args.epochs_c,
        l2sp_lambda=args.l2sp,
        freeze_bucket_embed=not args.no_freeze_bucket,
    )
