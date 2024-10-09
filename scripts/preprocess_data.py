#!/usr/bin/env python3
"""
preprocess_data.py

A script to preprocess raw text data for language model training using Hugging Face Transformers and Datasets.

Usage:
    python preprocess_data.py --input_file path/to/raw_text.txt --output_dir path/to/output_dir

Example:
    python preprocess_data.py --input_file data/raw/plato.txt --output_dir data/processed/

Dependencies:
    - transformers
    - datasets
    - nltk

Install dependencies using:
    pip install transformers datasets nltk
"""

import argparse
import logging
import os
import re
import nltk
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Ensure necessary NLTK data files are downloaded
nltk.download('punkt', quiet=True)

def setup_logging(log_file='preprocess_data.log'):
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
    parser = argparse.ArgumentParser(description='Preprocess raw text data for language model training.')
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to the raw text file.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/',
        help='Directory where processed data will be saved.'
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

def read_and_clean_text(file_path):
    """
    Reads and cleans the raw text file.
    """
    if not os.path.exists(file_path):
        logging.error(f"Input file not found at {file_path}. Exiting.")
        exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logging.info(f"Loaded raw text from {file_path}.")
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        exit(1)

    # Remove Project Gutenberg headers/footers if present
    text = re.sub(r'(?s).*START OF THIS PROJECT GUTENBERG EBOOK.*?\*{3}', '', text)
    text = re.sub(r'(?s)End of the Project Gutenberg.*', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    logging.info("Text cleaning completed.")
    return text

def tokenize_function(examples, tokenizer):
    """
    Tokenizes the text using the specified tokenizer.
    """
    return tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, max_length=block_size)

def group_texts(examples, block_size):
    """
    Concatenates texts and groups them into blocks of a fixed size.
    """
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples['input_ids'])

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

    # Read and clean the text
    cleaned_text = read_and_clean_text(input_file)

    # Create a dataset from the text
    dataset = load_dataset('text', data_files={'train': input_file})

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    def tokenize_batch(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, max_length=block_size)

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=['text']
    )

    # Group texts into blocks
    lm_dataset = tokenized_dataset.map(
        lambda examples: group_texts(examples, block_size),
        batched=True
    )

    # Split into train and validation datasets
    split_datasets = lm_dataset['train'].train_test_split(test_size=0.1, seed=42)
    data_dict = DatasetDict({
        'train': split_datasets['train'],
        'validation': split_datasets['test']
    })

    # Save the datasets
    data_dict.save_to_disk(output_dir)
    logging.info(f"Processed datasets saved to {output_dir}")

    logging.info("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
