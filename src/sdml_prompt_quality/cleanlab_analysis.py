"""
Cleanlab analysis script for identifying label issues in the dataset.
Outputs the top N most problematic entries to a JSON file.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
except ImportError:
    print("ERROR: cleanlab is not installed. Please run: pip install cleanlab")
    exit(1)

from .data.datasets import PromptDataset
from .data.load_dataset import load_dataset_model_scores
from .data.preprocess import preprocess
from .models.v4 import load_model
from .metrics import collate

TOKENIZER_FILE = str(Path(__file__).parent / "tokenizer" / "weighted_tokenizer.json")


def get_predictions(model, data_loader, device):
    """Get model predictions for all samples"""
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
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
            
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch["target"].squeeze(1).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels)
    
    return np.concatenate(all_probs), np.concatenate(all_labels)


def convert_to_binary_for_cleanlab(labels, threshold=0.5):
    """
    Convert continuous labels to binary for cleanlab analysis.
    Cleanlab works best with discrete class labels.
    """
    return (labels >= threshold).astype(int)


def analyze_with_cleanlab(pred_probs, labels, threshold=0.5):
    """
    Analyze dataset using cleanlab to find label issues.
    
    Args:
        pred_probs: Model predicted probabilities (continuous)
        labels: True labels (continuous)
        threshold: Threshold to convert continuous labels to binary
    
    Returns:
        label_issues_idx: Indices of samples with label issues
        label_quality_scores: Quality scores for each sample (lower = more problematic)
    """
    # Convert continuous labels to binary
    binary_labels = convert_to_binary_for_cleanlab(labels, threshold)
    
    # Ensure 1D
    pred_probs = np.asarray(pred_probs).reshape(-1)
    # Convert probabilities to 2D array for binary classification
    # Shape: (n_samples, 2) for [prob_class_0, prob_class_1]
    pred_probs_2d = np.column_stack([1 - pred_probs, pred_probs])
    
    # Compute cleanlab label quality scores (higher = better quality)
    label_quality_scores = get_label_quality_scores(
        labels=binary_labels,
        pred_probs=pred_probs_2d,
        method="self_confidence",
    )

    # Rank indices from most problematic (lowest quality score) to least
    ranked_indices = np.argsort(label_quality_scores)  # ascending

    return ranked_indices, label_quality_scores


def save_problematic_entries(
    output_file, 
    raw_inputs, 
    raw_labels, 
    pred_probs, 
    label_issues_idx, 
    label_quality_scores,
    top_n=100
):
    """
    Save the top N most problematic entries to a JSON file.
    
    Args:
        output_file: Path to output JSON file
        raw_inputs: Original input data (before preprocessing)
        raw_labels: Original labels
        pred_probs: Model predictions
        label_issues_idx: Indices of samples with label issues (ranked)
        label_quality_scores: Quality scores for each sample
        top_n: Number of top problematic entries to save
    """
    # Build all entries paired with their quality score so we can sort by it
    entries = []
    for idx in label_issues_idx:
        idx = int(idx)
        entry = {
            "index": idx,
            "sequence": raw_inputs[idx]["sequence"],
            "cfg_scale": float(raw_inputs[idx]["cfg_scale"]),
            "sampler": raw_inputs[idx]["sampler"],
            "steps": int(raw_inputs[idx]["steps"]),
            "upscaler": raw_inputs[idx]["upscaler"],
            "upscaler_steps": int(raw_inputs[idx]["upscaler_steps"]),
            "denoising_strength": float(raw_inputs[idx]["denoising_strength"]),
            "true_label": float(raw_labels[idx]),
            "predicted_score": float(pred_probs[idx]),
            "label_quality_score": float(label_quality_scores[idx]),
            "prediction_error": abs(float(raw_labels[idx]) - float(pred_probs[idx])),
        }
        entries.append(entry)

    # Sort by label_quality_score ascending (most problematic first)
    entries.sort(key=lambda e: e["label_quality_score"])  # ascending

    # Take top N
    problematic_entries = entries[:top_n]
    
    # Save to JSON
    output_data = {
        "metadata": {
            "total_samples": len(raw_labels),
            "total_issues_found": len(label_issues_idx),
            "top_n_saved": top_n,
            "description": "Top N most problematic entries identified by cleanlab. Lower label_quality_score indicates more problematic labels."
        },
        "problematic_entries": problematic_entries
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved top {top_n} problematic entries to: {output_file}")
    print(f"Total issues found: {len(label_issues_idx)} out of {len(raw_labels)} samples")


def run_analysis(
    model_path="best_model.pt",
    output_file="cleanlab_issues.json",
    top_n=100,
    batch_size=256,
    threshold=0.5,
    device=None,
):
    """
    Run cleanlab analysis on the dataset.
    
    Args:
        model_path: Path to trained model checkpoint
        output_file: Path to output JSON file
        top_n: Number of top problematic entries to save
        batch_size: Batch size for inference
        threshold: Threshold for converting continuous labels to binary
        device: Device for inference (cuda/cpu)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("Cleanlab Analysis for Label Issues")
    print("="*80)
    print(f"Model path: {model_path}")
    print(f"Output file: {output_file}")
    print(f"Top N: {top_n}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Binary threshold: {threshold}")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_input, dataset_output = load_dataset_model_scores(include_null=False)
    
    # Keep raw inputs for later
    raw_inputs = dataset_input.copy()
    raw_labels = np.array(dataset_output)
    
    print(f"Loaded {len(dataset_input)} samples")
    
    # Preprocess (same as training)
    print("\nPreprocessing data...")
    # Note: preprocess expects split_dataset format, so we need to handle this
    # For cleanlab, we want to analyze the training set
    from .data.load_dataset import split_dataset
    
    # Split the data the same way as training
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(
        dataset_input, dataset_output
    )
    
    # Concatenate train and test (as done in preprocess)
    x_train_full = np.concatenate([x_train, x_test], axis=0)
    y_train_full = np.concatenate([y_train, y_test], axis=0)
    
    # Store raw data before preprocessing
    raw_train_inputs = list(x_train_full)
    raw_train_labels = y_train_full
    
    # Now preprocess using the standard pipeline
    (train_samples, val_samples, vocab_size, lora_vocab_size, 
     sampler_vocab_size, upscaler_vocab_size) = preprocess(
        dataset_input, dataset_output, TOKENIZER_FILE
    )
    
    print(f"Training set size: {len(raw_train_labels)}")
    print(f"Vocab size: {vocab_size}")
    print(f"Lora vocab size: {lora_vocab_size}")
    
    # Create data loader
    print("\nCreating data loader...")
    train_loader = DataLoader(
        PromptDataset(train_samples),
        batch_size=batch_size,
        shuffle=False,  # Important: don't shuffle for cleanlab
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
    
    # Get predictions
    print("\nGenerating predictions...")
    pred_probs, labels = get_predictions(model, train_loader, device)
    print(f"Generated predictions for {len(pred_probs)} samples")
    
    # Analyze with cleanlab
    print("\nAnalyzing with cleanlab...")
    label_issues_idx, label_quality_scores = analyze_with_cleanlab(
        pred_probs, labels, threshold=threshold
    )
    
    print(f"Found {len(label_issues_idx)} potential label issues")
    
    # Calculate statistics
    issue_rate = len(label_issues_idx) / len(labels) * 100
    print(f"Issue rate: {issue_rate:.2f}%")
    
    # Show some statistics about the most problematic entries
    top_10_indices = label_issues_idx[:10]
    print("\nTop 10 most problematic entries:")
    print("-" * 80)
    print(f"{'Index':<8} {'True Label':<12} {'Predicted':<12} {'Quality Score':<15} {'Error':<8}")
    print("-" * 80)
    for idx in top_10_indices:
        idx = int(idx)
        print(f"{idx:<8} {labels[idx]:<12.4f} {pred_probs[idx]:<12.4f} "
              f"{label_quality_scores[idx]:<15.4f} {abs(labels[idx] - pred_probs[idx]):<8.4f}")
    
    # Save results
    print("\nSaving results...")
    save_problematic_entries(
        output_file,
        raw_train_inputs,
        raw_train_labels,
        pred_probs,
        label_issues_idx,
        label_quality_scores,
        top_n=top_n,
    )
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Perform cleanlab analysis on the dataset to find label issues"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="best_model.pt",
        help="Path to trained model checkpoint (default: best_model.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cleanlab_issues.json",
        help="Output JSON file for problematic entries (default: cleanlab_issues.json)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top problematic entries to save (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for converting continuous labels to binary (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (default: cuda if available, else cpu)",
    )
    
    args = parser.parse_args()
    
    run_analysis(
        model_path=args.model_path,
        output_file=args.output,
        top_n=args.top_n,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()

