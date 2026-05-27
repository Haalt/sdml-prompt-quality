from multiprocessing import freeze_support
import argparse
import copy
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

logger = logging.getLogger(__name__)

from .models.v3_clip_embed import PromptQualityModelV3ClipEmbed
from .metrics import collate, epoch_metrics
from .data.preprocess_clip_embed import preprocess_clip_embed
from .data.load_dataset import load_dataset_model_scores

# Use the existing tokenizer file for LoRAs/etc.
TOKENIZER_FILE = str(Path(__file__).parent / "tokenizer" / "combined_tokenizer.json")


class PromptDatasetClipEmbed(Dataset):
    """
    Dataset for CLIP-embedding based model.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return self.samples[0].shape[0]

    def __getitem__(self, idx):
        item = {
            "clip_emb": torch.as_tensor(self.samples[0][idx], dtype=torch.float32),
            "lora_ids": torch.as_tensor(self.samples[1][idx], dtype=torch.long),
            "lora_w": torch.as_tensor(self.samples[2][idx], dtype=torch.float32),
            "cfg": torch.as_tensor([self.samples[3][idx]], dtype=torch.float32),
            "n_loras": torch.as_tensor([self.samples[4][idx]], dtype=torch.float32),
            "sampler_id": torch.as_tensor([self.samples[5][idx]], dtype=torch.long),
            "steps_log": torch.as_tensor([self.samples[6][idx]], dtype=torch.float32),
            "steps_bucket": torch.as_tensor([self.samples[7][idx]], dtype=torch.long),
            "upscaler_id": torch.as_tensor([self.samples[8][idx]], dtype=torch.long),
            "up_has": torch.as_tensor([self.samples[9][idx]], dtype=torch.float32),
            "up_steps": torch.as_tensor([self.samples[10][idx]], dtype=torch.float32),
            "denoise": torch.as_tensor([self.samples[11][idx]], dtype=torch.float32),
            "model_id": torch.as_tensor([self.samples[12][idx]], dtype=torch.long),
            "target": torch.as_tensor([self.samples[13][idx]], dtype=torch.float32),
            "weight": torch.as_tensor([9.45 if self.samples[12][idx] == 1 else 1.0], dtype=torch.float32),
        }
        return item


def train_model(
    train_samples,
    val_samples,
    vocab_loras,
    vocab_samplers,
    vocab_upscalers,
    num_epochs=150,
    patience=8,
    batch_size=256,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    # data loaders
    train_loader = DataLoader(
        PromptDatasetClipEmbed(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PromptDatasetClipEmbed(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    model = PromptQualityModelV3ClipEmbed(
        vocab_loras=vocab_loras,
        vocab_samplers=vocab_samplers,
        vocab_upscalers=vocab_upscalers,
        bucket_size=20,
        d_L=128,
        head_h1=512,
        dropout_p=0.2,
    ).to(device)

    print("train set size:", len(train_loader.dataset))
    print("val   set size:", len(val_loader.dataset))

    criterion = nn.BCEWithLogitsLoss()

    best_val_mse = float("inf")
    best_state = None
    epochs_no_improve = 0

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimiser = optim.AdamW(params_to_update, lr=5e-4, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    for epoch in range(num_epochs):
        model.train()
        train_mae = 0.0
        train_mse = 0.0
        train_samples_count = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True)
                     for k, v in batch.items()}
            optimiser.zero_grad()

            logits = model(
                batch["clip_emb"],
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

            loss = criterion(logits, batch["target"].squeeze(1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            y_hat = torch.sigmoid(logits)
            train_mae += F.l1_loss(y_hat, batch["target"].squeeze(1), reduction="sum").item()
            train_mse += F.mse_loss(y_hat, batch["target"].squeeze(1), reduction="sum").item()
            train_samples_count += batch["target"].shape[0]

        model.eval()
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = {k: v.to(device, non_blocking=True)
                         for k, v in batch.items()}

                logits = model(
                    batch["clip_emb"],
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

                y_hat = torch.sigmoid(logits)
                all_pred.append(y_hat)
                all_tgt.append(batch["target"].squeeze(1))

        val_mae, val_mse, val_r2 = epoch_metrics(all_pred, all_tgt)

        # Calculate AUC (binarized)
        y_true_cont = torch.cat(all_tgt).cpu().numpy().ravel()
        y_scores = torch.cat(all_pred).cpu().numpy().ravel()
        thr = float(np.median(y_true_cont))
        y_true_bin = (y_true_cont >= thr).astype(int)

        if np.unique(y_true_bin).size >= 2:
            val_auc = roc_auc_score(y_true_bin, y_scores)
        else:
            val_auc = float("nan")

        scheduler.step(val_mse)

        train_mae_avg = train_mae / train_samples_count if train_samples_count > 0 else float("nan")
        train_mse_avg = train_mse / train_samples_count if train_samples_count > 0 else float("nan")

        print(
            f"Epoch {epoch+1}/{num_epochs} | train_MSE={train_mse_avg:.5f}  "
            f"train_MAE={train_mae_avg:.5f}  val_MSE={val_mse:.5f}  "
            f"val_MAE={val_mae:.5f}  val_R2={val_r2:.4f}  val_AUROC={val_auc:.5f}"
        )
        print("LR:", optimiser.param_groups[0]["lr"])

        if val_mse + 1e-6 < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_state, "best_model_clip_embed.pt")
            print("  ↳ new best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after patience exhausted. Best val_MSE={best_val_mse:.5f}")
                break

    model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="Train V3 with pooled CLIP embeddings")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--clip_batch_size", type=int, default=256)
    parser.add_argument("--clip_cache", type=str, default=None)
    args = parser.parse_args()

    dataset_input, dataset_output = load_dataset_model_scores(normalize=False)
    dataset_input_xl, dataset_output_xl = load_dataset_model_scores(is_xl=True, normalize=False)

    dataset_input += dataset_input_xl
    dataset_output += dataset_output_xl

    (train_samples, val_samples, vocab_loras,
     vocab_samplers, vocab_upscalers) = preprocess_clip_embed(
        dataset_input,
        dataset_output,
        TOKENIZER_FILE,
        clip_model_name=args.clip_model,
        device=args.device,
        cache_path=args.clip_cache,
        clip_batch_size=args.clip_batch_size,
    )

    train_model(
        train_samples,
        val_samples,
        vocab_loras=vocab_loras,
        vocab_samplers=vocab_samplers,
        vocab_upscalers=vocab_upscalers,
        num_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )

