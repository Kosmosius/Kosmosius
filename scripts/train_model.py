#!/usr/bin/env python3
"""
train_model.py

A script to fine-tune a pre-trained language model on your dataset using Hugging Face Transformers.

Usage:
    python train_model.py --train_dataset path/to/train_dataset --val_dataset path/to/val_dataset --output_dir path/to/output_model

Example:
    python train_model.py --train_dataset data/datasets/train_dataset --val_dataset data/datasets/val_dataset --output_dir models/plato-gpt2

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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_from_disk
import torch

def setup_logging(log_file='train_model.log'):
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
    parser = argparse.ArgumentParser(description='Fine-tune a language model on your dataset.')
    parser.add_argument(
        '--train_dataset',
        type=str,
        required=True,
        help='Path to the training dataset.'
    )
    parser.add_argument(
        '--val_dataset',
        type=str,
        required=True,
        help='Path to the validation dataset.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/plato-gpt2',
        help='Directory where the fine-tuned model will be saved.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt2',
        help='Pre-trained model to fine-tune (default: gpt2).'
    )
    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=3,
        help='Total number of training epochs to perform.'
    )
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=1,
        help='Batch size per GPU/TPU core/CPU for training.'
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=1,
        help='Batch size per GPU/TPU core/CPU for evaluation.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Initial learning rate for AdamW optimizer.'
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=100,
        help='Log every X updates steps.'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help='Save checkpoint every X updates steps.'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help='Run evaluation every X steps.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for initialization.'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use mixed precision training.'
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Load the datasets
    train_dataset = load_from_disk(args.train_dataset)
    val_dataset = load_from_disk(args.val_dataset)

    logging.info(f"Training dataset loaded from {args.train_dataset}")
    logging.info(f"Validation dataset loaded from {args.val_dataset}")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        seed=args.seed,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Start training
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed.")

    # Evaluate the model
    logging.info("Starting evaluation...")
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results: {eval_results}")

    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
