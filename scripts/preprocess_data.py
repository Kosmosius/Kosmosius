#!/usr/bin/env python3
"""
preprocess_data.py

A script to preprocess raw text data for language model training using Hugging Face libraries.

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
from datasets import Dataset
from transformers import AutoTokenizer
import nltk

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

def read_text_file(file_path):
    """
    Reads the raw text file.
    """
    if not os.path.exists(file_path):
        logging.error(f"Input file not found at {file_path}. Exiting.")
        exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logging.info(f"Loaded raw text from {file_path}.")
        return text
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        exit(1)

def clean_text(text):
    """
    Cleans the raw text by removing unwanted characters and headers/footers.
    """
    # Remove Project Gutenberg headers/footers if present
    text = re.sub(r'(?s).*START OF THIS PROJECT GUTENBERG EBOOK.*?\*{3}', '', text)
    text = re.sub(r'(?s)End of the Project Gutenberg.*', '', text)

    # Remove any non-textual elements or artifacts (e.g., digits, special characters)
    # text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    logging.info("Text cleaning completed.")
    return text

def tokenize_function(examples, tokenizer):
    """
    Tokenizes the text using the specified tokenizer.
    """
    return tokenizer(examples['text'])

def group_texts(examples, block_size):
    """
    Concatenates texts and groups them into blocks of a fixed size.
    """
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated['input_ids'])

    # Drop the small remainder, we could add padding if the model supported it instead of this drop
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
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
    raw_text = read_text_file(input_file)
    cleaned_text = clean_text(raw_text)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Split text into sentences (optional)
    # sentences = nltk.tokenize.sent_tokenize(cleaned_text)
    # dataset = Dataset.from_dict({'text': sentences})

    # For language modeling, we can use the entire text as a single example
    dataset = Dataset.from_dict({'text': [cleaned_text]})

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

    logging.info("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
