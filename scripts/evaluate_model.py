#!/usr/bin/env python3
"""
evaluate_model.py

A script to evaluate a fine-tuned language model on a validation dataset using Hugging Face Transformers.

Usage:
    python evaluate_model.py --model_dir path/to/model_dir --val_dataset path/to/val_dataset

Example:
    python evaluate_model.py --model_dir models/plato-gpt2 --val_dataset data/datasets/val_dataset

Dependencies:
    - transformers
    - datasets
    - torch
    - numpy

Install dependencies using:
    pip install transformers datasets torch numpy
"""

import os
import argparse
import logging
import math
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_from_disk
import torch

def setup_logging(log_file='evaluate_model.log'):
    """
    Sets up logging for the script.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    # Also log to stdout
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned language model.')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path to the fine-tuned model directory.'
    )
    parser.add_argument(
        '--val_dataset',
        type=str,
        required=True,
        help='Path to the validation dataset.'
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=1,
        help='Batch size per GPU/TPU core/CPU for evaluation.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory where evaluation results will be saved.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for initialization.'
    )
    return parser.parse_args()

def main():
    # Setup logging
    setup_logging()

    # Parse command-line arguments
    args = parse_arguments()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    # Load the validation dataset
    val_dataset = load_from_disk(args.val_dataset)
    logging.info(f"Validation dataset loaded from {args.val_dataset}")

    # Set up evaluation arguments
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        seed=args.seed,
        dataloader_drop_last=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val_dataset,
    )

    # Evaluate the model
    logging.info("Starting evaluation...")
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results: {eval_results}")

    # Calculate perplexity
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss)
    logging.info(f"Perplexity: {perplexity:.2f}")

    # Save evaluation results to a file
    results_file = os.path.join(args.output_dir, 'eval_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results:\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Perplexity: {perplexity:.2f}\n")
    logging.info(f"Evaluation results saved to {results_file}")

if __name__ == "__main__":
    main()
