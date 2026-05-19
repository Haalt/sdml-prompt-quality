from multiprocessing import freeze_support
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .data.datasets import PromptDataset
from .data.load_dataset import load_dataset_model_scores
from .data.preprocess import preprocess
from .metrics import collate
from .models.v3 import PromptQualityModelV3Quantile


TOKENIZER_FILE = str(Path(__file__).parent / "tokenizer" / "combined_tokenizer.json")


def pinball_loss(pred, target, tau):
    err = target - pred
    return torch.mean(torch.maximum(tau * err, (tau - 1) * err))


def _forward_model(model, batch):
    return model(
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
        batch["model_id"],
    )


def _epoch_stats(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    mae = np.mean(np.abs(pred_np - target_np), axis=0)
    mse = np.mean((pred_np - target_np) ** 2, axis=0)
    return mae, mse


def train_model(
    train_samples,
    val_samples,
    vocab_tokens,
    vocab_loras,
    vocab_samplers,
    vocab_upscalers,
    num_epochs=120,
    patience=12,
    batch_size=2048,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
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

    model = PromptQualityModelV3Quantile(
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

    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_preds = []
        train_targets = []
        train_pin50 = 0.0
        train_pin90 = 0.0
        train_count = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            y = batch["target"]
            if y.dim() != 2 or y.size(1) != 2:
                raise ValueError(
                    f"Quantile training expects target shape [B,2], got {tuple(y.shape)}"
                )

            optimiser.zero_grad()
            pred = _forward_model(model, batch)

            pred_q50 = pred[:, 0]
            pred_q90 = pred[:, 1]
            y_q50 = y[:, 0]
            y_q90 = y[:, 1]

            loss_q50 = pinball_loss(pred_q50, y_q50, 0.5)
            loss_q90 = pinball_loss(pred_q90, y_q90, 0.9)
            loss = loss_q50 + loss_q90
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            bsz = y.size(0)
            train_pin50 += loss_q50.item() * bsz
            train_pin90 += loss_q90.item() * bsz
            train_count += bsz
            train_preds.append(pred.detach())
            train_targets.append(y.detach())

        model.eval()
        val_preds = []
        val_targets = []
        val_pin50 = 0.0
        val_pin90 = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                y = batch["target"]
                pred = _forward_model(model, batch)

                pred_q50 = pred[:, 0]
                pred_q90 = pred[:, 1]
                y_q50 = y[:, 0]
                y_q90 = y[:, 1]

                loss_q50 = pinball_loss(pred_q50, y_q50, 0.5)
                loss_q90 = pinball_loss(pred_q90, y_q90, 0.9)

                bsz = y.size(0)
                val_pin50 += loss_q50.item() * bsz
                val_pin90 += loss_q90.item() * bsz
                val_count += bsz
                val_preds.append(pred)
                val_targets.append(y)

        train_pred_tensor = torch.cat(train_preds, dim=0)
        train_target_tensor = torch.cat(train_targets, dim=0)
        val_pred_tensor = torch.cat(val_preds, dim=0)
        val_target_tensor = torch.cat(val_targets, dim=0)

        train_mae, train_mse = _epoch_stats(train_pred_tensor, train_target_tensor)
        val_mae, val_mse = _epoch_stats(val_pred_tensor, val_target_tensor)

        train_pin50_avg = train_pin50 / max(train_count, 1)
        train_pin90_avg = train_pin90 / max(train_count, 1)
        val_pin50_avg = val_pin50 / max(val_count, 1)
        val_pin90_avg = val_pin90 / max(val_count, 1)
        val_loss = val_pin50_avg + val_pin90_avg

        print(
            f"Epoch {epoch:03d} | "
            f"train_pin50={train_pin50_avg:.5f} train_pin90={train_pin90_avg:.5f} "
            f"train_mae50={train_mae[0]:.5f} train_mae90={train_mae[1]:.5f} "
            f"train_mse50={train_mse[0]:.5f} train_mse90={train_mse[1]:.5f} || "
            f"val_pin50={val_pin50_avg:.5f} val_pin90={val_pin90_avg:.5f} "
            f"val_mae50={val_mae[0]:.5f} val_mae90={val_mae[1]:.5f} "
            f"val_mse50={val_mse[0]:.5f} val_mse90={val_mse[1]:.5f}"
        )
        print("LR:", optimiser.param_groups[0]["lr"])

        scheduler.step(val_loss)
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_state, "best_model_quantile.pt")
            print("  ↳ new best quantile model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping after patience exhausted. Best val loss={best_val_loss:.5f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="Train V3 quantile model (q50/q90)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable per-target Z-score normalization before training.",
    )
    args = parser.parse_args()

    # dataset_input, dataset_output = load_dataset_model_scores(
    #     combine=True,
    #     normalize=args.normalize,
    #     target_mode="quantiles",
    # )

    
    dataset_input, dataset_output = load_dataset_model_scores(normalize=False,
        target_mode="quantiles")
    dataset_input_xl, dataset_output_xl = load_dataset_model_scores(is_xl=True, normalize=False,
        target_mode="quantiles")

    dataset_input += dataset_input_xl
    dataset_output += dataset_output_xl

    (
        train_samples,
        val_samples,
        vocab_tokens,
        vocab_loras,
        vocab_samplers,
        vocab_upscalers,
    ) = preprocess(dataset_input, dataset_output, TOKENIZER_FILE)

    vocab_tokens = 1376

    _ = train_model(
        train_samples,
        val_samples,
        vocab_tokens=vocab_tokens,
        vocab_loras=vocab_loras,
        vocab_samplers=vocab_samplers,
        vocab_upscalers=vocab_upscalers,
        num_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
    )
