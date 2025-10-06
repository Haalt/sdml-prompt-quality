#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def cmd_train(args) -> int:
    logger = logging.getLogger(__name__)

    try:
        from .train import train_model, TOKENIZER_FILE
        print(TOKENIZER_FILE)
        from .data.load_dataset import load_dataset_model_scores, split_dataset
        from .data.preprocess import preprocess

        logger.info("Loading dataset...")
        dataset_input, dataset_output = load_dataset_model_scores()

        logger.info("Preprocessing data...")
        (train_samples, val_samples, vocab_tokens, vocab_loras,
         vocab_samplers, vocab_upscalers) = preprocess(
            dataset_input, dataset_output, TOKENIZER_FILE
        )

        logger.info("Starting training...")
        model, train_mae, train_mse, val_mse, val_mae, val_r2 = train_model(
            train_samples=train_samples,
            val_samples=val_samples,
            vocab_tokens=vocab_tokens,
            vocab_loras=vocab_loras,
            vocab_samplers=vocab_samplers,
            vocab_upscalers=vocab_upscalers,
        )

        logger.info(
            f"Training completed! Final metrics - Val MSE: {val_mse:.5f}, Val MAE: {val_mae:.5f}")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


def cmd_export_onnx(args) -> int:
    logger = logging.getLogger(__name__)

    try:
        from .export_onnx import convert_v3

        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        logger.info(f"Exporting model from {args.checkpoint} to {args.output}")

        convert_v3(
            out_path=args.output,
            model_path=args.checkpoint,
            device=args.device,
            fp16=args.fp16,
            keep_io_types=args.keep_io_types,
        )

        logger.info(f"ONNX export completed: {args.output}")
        return 0

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SDML Prompt Quality CLI",
        prog="sdpq"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train the prompt quality model")

    train_parser.set_defaults(func=cmd_train)

    # Export ONNX command
    export_parser = subparsers.add_parser(
        "export-onnx", help="Export PyTorch model to ONNX")
    export_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint"
    )
    export_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output ONNX file path"
    )
    export_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for export"
    )
    export_parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert to FP16 after export"
    )
    export_parser.add_argument(
        "--keep-io-types",
        action="store_true",
        help="Keep I/O as FP32 when using FP16"
    )
    export_parser.set_defaults(func=cmd_export_onnx)

    args = parser.parse_args()

    setup_logging(args.verbose)

    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
