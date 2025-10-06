from multiprocessing import freeze_support
import argparse
import os
import json
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

from .data.datasets import PromptDataset
from .models.v3 import PromptQualityModelV3
# from .models.v4 import PromptQualityModelV4
from .metrics import collate, epoch_metrics
from .data.preprocess import preprocess
from .data.load_dataset import load_dataset, split_dataset

TOKENIZER_FILE = str(Path(__file__).parent /
                     "tokenizer" / "weighted_tokenizer.json")


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
):

    # data loaders
    train_loader = DataLoader(
        PromptDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PromptDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
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
        d_t=256,
        d_L=128,
        head_h1=512,
        dropout_p=0.4,
    ).to(device)

    print("train set size:", len(train_loader.dataset))
    print("val   set size:", len(val_loader.dataset))

    criterion = nn.BCEWithLogitsLoss()

    base = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("self_attn") and not n.startswith("attn_ln")
    ]
    attn = [
        p
        for n, p in model.named_parameters()
        if n.startswith("self_attn") or n.startswith("attn_ln")
    ]
    optimiser = optim.AdamW(
        [{"params": base, "lr": 3e-4}, {"params": attn, "lr": 1e-3}], weight_decay=1e-4
    )

    warm_up = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1e-6, end_factor=1.0, total_iters=3
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=num_epochs - 3)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser, schedulers=[warm_up, cosine], milestones=[3]
    )

    best_val_mse = float("inf")
    best_state = None
    epochs_no_improve = 0

    saved_train_mae = float("inf")
    saved_train_mse = float("inf")
    saved_val_mse = float("inf")
    saved_val_mae = float("inf")
    saved_val_r2 = float("inf")

    for epoch in range(1, num_epochs + 1):
        # training
        model.train()
        train_mae = 0.0
        train_mse = 0.0
        for batch in train_loader:
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
            )
            y_hat = torch.sigmoid(logits)
            loss = criterion(logits, batch["target"].squeeze(1))
            loss.backward()
            optimiser.step()

            train_mae += F.l1_loss(
                y_hat, batch["target"].squeeze(1), reduction="sum"
            ).item()
            train_mse += F.mse_loss(
                y_hat, batch["target"].squeeze(1), reduction="sum"
            ).item()

        # validation
        model.eval()
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True)
                         for k, v in batch.items()}

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
                )

                y_hat = torch.sigmoid(logits)
                all_pred.append(y_hat)
                all_tgt.append(batch["target"].squeeze(1))

        val_mae, val_mse, val_r2 = epoch_metrics(all_pred, all_tgt)

        y_true_cont = torch.cat(all_tgt).cpu().numpy().ravel()
        y_scores = torch.cat(all_pred).cpu().numpy().ravel()
        thr = float(np.median(y_true_cont))
        y_true_bin = (y_true_cont >= thr).astype(int)
        if np.unique(y_true_bin).size >= 2:
            val_auc = roc_auc_score(y_true_bin, y_scores)
        else:
            val_auc = float("nan")
        scheduler.step()

        print(
            f"Epoch {epoch:03d}  | train_MSE={(train_mse / len(train_loader)):.5f}  train_MAE={(train_mae / len(train_loader)):.5f}  val_MSE={val_mse:.5f}  "
            f"val_MAE={val_mae:.5f}  val_R2={val_r2:.4f}  val_AUROC={val_auc:.5f}"
        )
        print("LR:", optimiser.param_groups[0]["lr"])

        if val_mse + 1e-6 < best_val_mse:  # tiny epsilon to avoid float noise
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_state, "best_model.pt")
            print("  ↳ new best model saved.")
            saved_train_mae = train_mae
            saved_train_mse = train_mse
            saved_val_mse = val_mse
            saved_val_mae = val_mae
            saved_val_r2 = val_r2
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping after {epoch} epochs. "
                    f"Best val_MSE={best_val_mse:.5f}"
                )
                break

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

    dataset_input, dataset_output = load_dataset()
    (train_samples, val_samples) = split_dataset(dataset_input, dataset_output)

    (train_samples, val_samples, vocab_size, lora_vocab_size, sampler_vocab_size,
     upscaler_vocab_size) = preprocess(train_samples, val_samples, TOKENIZER_FILE)

    best_model, _, _, _, _, _ = train_model(
        train_samples,
        val_samples,
        vocab_tokens=vocab_size,
        vocab_loras=lora_vocab_size,
        vocab_samplers=sampler_vocab_size,
        vocab_upscalers=upscaler_vocab_size,
    )
