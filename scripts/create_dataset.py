#!/usr/bin/env python3
"""
create_dataset.py

A script to create Hugging Face datasets for language modeling from preprocessed text data.

Usage:
    python create_dataset.py --input_file path/to/preprocessed_text.txt --output_dir path/to/output_dataset

Example:
    python create_dataset.py --input_file data/processed/cleaned_plato.txt --output_dir data/datasets/

Dependencies:
    - transformers
    - datasets

Install dependencies using:
    pip install transformers datasets
"""

import os
import argparse
import logging
from datasets import Dataset
from transformers import AutoTokenizer

def setup_logging(log_file='create_dataset.log'):
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
    parser = argparse.ArgumentParser(description='Create Hugging Face datasets for language modeling.')
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to the preprocessed text file.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/datasets/',
        help='Directory where the datasets will be saved.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt2',
        help='Pre-trained model tokenizer to use (default: gpt2).'
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=1024,
        help='Optional input sequence length after tokenization.'
    )
    return parser.parse_args()

def read_text_file(file_path):
    """
    Reads the preprocessed text file.
    """
    if not os.path.exists(file_path):
        logging.error(f"Input file not found at {file_path}. Exiting.")
        exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logging.info(f"Loaded preprocessed text from {file_path}.")
        return text
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        exit(1)

def tokenize_function(examples, tokenizer):
    """
    Tokenizes the text using the specified tokenizer.
    """
    return tokenizer(examples['text'], return_special_tokens_mask=True)

def group_texts(examples, block_size):
    """
    Concatenates texts and groups them into blocks of a fixed size.
    """
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # Create labels
    result['labels'] = result['input_ids'].copy()
    return result

def save_dataset(dataset, output_dir, filename):
    """
    Saves the dataset to disk.
    """
    output_path = os.path.join(output_dir, filename)
    dataset.save_to_disk(output_path)
    logging.info(f"Dataset saved to {output_path}")

def main():
    # Setup logging
    setup_logging()

    # Parse command-line arguments
    args = parse_arguments()

    input_file = args.input_file
    output_dir = args.output_dir
    model_name = args.model_name
    block_size = args.block_size

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the preprocessed text
    text = read_text_file(input_file)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create the dataset
    dataset = Dataset.from_dict({'text': [text]})

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=['text']
    )

    # Group texts into blocks
    lm_dataset = tokenized_dataset.map(
        lambda examples: group_texts(examples, block_size),
        batched=True
    )

    # Split into train and validation datasets
    split_datasets = lm_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['test']

    # Save the datasets
    save_dataset(train_dataset, output_dir, 'train_dataset')
    save_dataset(val_dataset, output_dir, 'val_dataset')

    logging.info("Dataset creation completed successfully.")

if __name__ == "__main__":
    main()
