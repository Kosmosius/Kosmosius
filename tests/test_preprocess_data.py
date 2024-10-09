#!/usr/bin/env python3
"""
test_preprocessing.py

Unit tests for preprocess_data.py using the pytest framework.

Run the tests:
    pytest tests/test_preprocessing.py

Dependencies:
    - pytest
    - datasets
    - transformers
    - nltk

Ensure that the 'preprocess_data.py' script is in the 'scripts' directory and that 'scripts' is a Python package.
"""

import pytest
import os
import shutil
import tempfile
from datasets import Dataset
from transformers import AutoTokenizer
import nltk

# Import functions from preprocess_data.py
# Adjust the import path based on your project structure
# Ensure that 'scripts' directory contains an __init__.py file
from scripts.preprocess_data import (
    read_and_clean_text,
    tokenize_function,
    group_texts,
    save_dataset,
)

# Ensure necessary NLTK data files are downloaded for testing
nltk.download('punkt', quiet=True)

@pytest.fixture
def temp_environment():
    """
    Fixture to set up a temporary environment for testing.
    Creates temporary input and output directories/files.
    Cleans up after the test.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    input_file = os.path.join(temp_dir, 'test_input.txt')
    output_dir = os.path.join(temp_dir, 'output')
    block_size = 128  # Define a block size for testing

    # Sample raw text for testing
    raw_text = """
    *** START OF THIS PROJECT GUTENBERG EBOOK THE TEST BOOK ***
    This is a sample text for testing purposes.
    It contains multiple sentences.

    End of the Project Gutenberg EBook.
    """

    # Write the raw text to a temporary input file
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(raw_text)

    yield {
        'temp_dir': temp_dir,
        'input_file': input_file,
        'output_dir': output_dir,
        'raw_text': raw_text,
        'block_size': block_size,
    }

    # Clean up the temporary directory after the test
    shutil.rmtree(temp_dir)

@pytest.fixture
def tokenizer():
    """
    Fixture to initialize and provide the tokenizer.
    """
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def test_read_and_clean_text(temp_environment):
    """
    Test the read_and_clean_text function to ensure it correctly reads and cleans the raw text file.
    """
    cleaned_text = read_and_clean_text(temp_environment['input_file'])
    assert isinstance(cleaned_text, str), "read_and_clean_text should return a string."
    assert len(cleaned_text) > 0, "read_and_clean_text should not return an empty string."
    assert 'This is a sample text' in cleaned_text, "read_and_clean_text did not read the correct content."
    assert 'START OF THIS PROJECT GUTENBERG EBOOK' not in cleaned_text, "read_and_clean_text did not remove Gutenberg header."
    assert 'End of the Project Gutenberg EBook' not in cleaned_text, "read_and_clean_text did not remove Gutenberg footer."
    assert cleaned_text.startswith('This is a sample text'), "cleaned text does not start with expected content."
    assert cleaned_text.endswith('multiple sentences.'), "cleaned text does not end with expected content."

def test_tokenize_function(tokenizer):
    """
    Test the tokenize_function to ensure it correctly tokenizes input text.
    """
    text = 'This is a test.'
    dataset = Dataset.from_dict({'text': [text]})
    tokenized = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    assert 'input_ids' in tokenized.features, "Tokenized dataset should contain 'input_ids'."
    assert 'attention_mask' in tokenized.features, "Tokenized dataset should contain 'attention_mask'."
    assert isinstance(tokenized['input_ids'][0], list), "'input_ids' should be a list."
    assert len(tokenized['input_ids'][0]) > 0, "'input_ids' should not be empty."

def test_group_texts(tokenizer, temp_environment):
    """
    Test the group_texts function to ensure it correctly groups tokenized texts into fixed-size blocks.
    """
    block_size = temp_environment['block_size']
    text = ' '.join(['This is a test sentence.'] * 100)  # Create a long text
    dataset = Dataset.from_dict({'text': [text]})
    tokenized = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    lm_dataset = tokenized.map(
        lambda examples: group_texts(examples, block_size),
        batched=True
    )
    assert 'input_ids' in lm_dataset.features, "Grouped dataset should contain 'input_ids'."
    assert 'labels' in lm_dataset.features, "Grouped dataset should contain 'labels'."
    assert isinstance(lm_dataset['input_ids'][0], list), "'input_ids' should be a list."
    assert len(lm_dataset['input_ids'][0]) == block_size, f"'input_ids' length should be {block_size}."

def test_save_dataset(temp_environment):
    """
    Test the save_dataset function to ensure it correctly saves the dataset to disk.
    """
    text = 'This is a test.'
    dataset = Dataset.from_dict({'text': [text]})
    output_filename = 'test_dataset'
    save_dataset(dataset, temp_environment['output_dir'], output_filename)
    output_path = os.path.join(temp_environment['output_dir'], output_filename)
    assert os.path.exists(output_path), "save_dataset did not create the expected output directory."
    # Additional check: load the dataset and verify content
    loaded_dataset = Dataset.load_from_disk(output_path)
    assert 'input_ids' in loaded_dataset.features, "Loaded dataset should contain 'input_ids'."
    assert len(loaded_dataset) == 1, "Loaded dataset should contain one example."

def test_full_preprocessing_pipeline(temp_environment, tokenizer):
    """
    Test the full preprocessing pipeline from reading the file to saving the datasets.
    """
    # Read and clean the text
    cleaned_text = read_and_clean_text(temp_environment['input_file'])
    assert 'This is a sample text' in cleaned_text, "read_and_clean_text did not process the text correctly."

    # Create dataset
    dataset = Dataset.from_dict({'text': [cleaned_text]})
    assert len(dataset) == 1, "Dataset should contain one example."

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    assert 'input_ids' in tokenized_dataset.features, "Tokenized dataset should contain 'input_ids'."
    assert 'attention_mask' in tokenized_dataset.features, "Tokenized dataset should contain 'attention_mask'."

    # Group texts
    block_size = temp_environment['block_size']
    lm_dataset = tokenized_dataset.map(
        lambda examples: group_texts(examples, block_size),
        batched=True
    )
    assert 'labels' in lm_dataset.features, "Grouped dataset should contain 'labels'."
    assert len(lm_dataset['input_ids'][0]) == block_size, "Grouped 'input_ids' length mismatch."

    # Split datasets
    split_datasets = lm_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['test']

    assert len(train_dataset) > 0, "Train dataset should not be empty."
    assert len(val_dataset) > 0, "Validation dataset should not be empty."
    assert 'input_ids' in train_dataset.features, "Train dataset should contain 'input_ids'."
    assert 'labels' in train_dataset.features, "Train dataset should contain 'labels'."
    assert len(train_dataset['input_ids'][0]) == block_size, "Train 'input_ids' length mismatch."

def test_tokenizer_padding(tokenizer):
    """
    Ensure the tokenizer has a pad_token set.
    """
    assert tokenizer.pad_token is not None, "Tokenizer should have a pad_token set."

def test_block_size():
    """
    Ensure block_size is a positive integer.
    """
    block_size = 128
    assert isinstance(block_size, int), "block_size should be an integer."
    assert block_size > 0, "block_size should be a positive integer."

def test_nltk_download():
    """
    Ensure NLTK 'punkt' tokenizer is downloaded.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        pytest.fail("NLTK 'punkt' tokenizer not found.")
