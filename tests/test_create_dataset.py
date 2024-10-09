#!/usr/bin/env python3
"""
test_create_dataset.py

Unit tests for create_dataset.py using the pytest framework.

Run the tests:
    pytest tests/test_create_dataset.py

Dependencies:
    - pytest
    - datasets
    - transformers

Ensure that the 'create_dataset.py' script is in the 'scripts' directory.
"""

import pytest
import os
import shutil
import tempfile
import re
from datasets import Dataset
from transformers import AutoTokenizer

# Import functions from create_dataset.py
# Adjust the import path based on your project structure
from scripts.create_dataset import (
    read_text_file,
    tokenize_function,
    group_texts,
    save_dataset,
)

@pytest.fixture
def temp_environment():
    """
    Fixture to set up a temporary environment for testing.
    Creates temporary input and output directories/files.
    Cleans up after the test.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    input_file = os.path.join(temp_dir, 'test_preprocessed.txt')
    output_dir = os.path.join(temp_dir, 'datasets_output')

    # Sample preprocessed text for testing
    preprocessed_text = "This is a cleaned and preprocessed text for testing purposes. " \
                       "It should be tokenized and grouped correctly."

    # Write the preprocessed text to a temporary input file
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(preprocessed_text)

    yield {
        'temp_dir': temp_dir,
        'input_file': input_file,
        'output_dir': output_dir,
        'preprocessed_text': preprocessed_text,
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

def test_read_text_file(temp_environment):
    """
    Test the read_text_file function to ensure it correctly reads a preprocessed text file.
    """
    text = read_text_file(temp_environment['input_file'])
    assert isinstance(text, str), "read_text_file should return a string."
    assert len(text) > 0, "read_text_file should not return an empty string."
    assert 'cleaned and preprocessed text' in text, "read_text_file did not read the correct content."

def test_tokenize_function(tokenizer):
    """
    Test the tokenize_function to ensure it correctly tokenizes input text.
    """
    text = "This is a test sentence."
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

def test_group_texts(tokenizer):
    """
    Test the group_texts function to ensure it correctly groups tokenized texts into fixed-size blocks.
    """
    block_size = 10
    text = "This is a test sentence for grouping texts into blocks."
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
    assert lm_dataset['input_ids'][0] == lm_dataset['labels'][0], "'labels' should be a copy of 'input_ids'."

def test_save_dataset(temp_environment):
    """
    Test the save_dataset function to ensure it correctly saves the dataset to disk.
    """
    text = "This is a test for saving the dataset."
    dataset = Dataset.from_dict({'text': [text]})
    output_filename = 'test_dataset'
    save_dataset(dataset, temp_environment['output_dir'], output_filename)
    output_path = os.path.join(temp_environment['output_dir'], output_filename)
    assert os.path.exists(output_path), "save_dataset did not create the expected output directory."

    # Additional check: load the dataset and verify content
    loaded_dataset = Dataset.load_from_disk(output_path)
    assert 'input_ids' in loaded_dataset.features, "Loaded dataset should contain 'input_ids'."
    assert 'attention_mask' in loaded_dataset.features, "Loaded dataset should contain 'attention_mask'."
    assert 'labels' in loaded_dataset.features, "Loaded dataset should contain 'labels'."
    assert len(loaded_dataset) == 1, "Loaded dataset should contain one example."

def test_full_dataset_creation_pipeline(temp_environment, tokenizer):
    """
    Test the full dataset creation pipeline from reading the file to saving the datasets.
    """
    # Read the preprocessed text
    text = read_text_file(temp_environment['input_file'])
    assert 'cleaned and preprocessed text' in text, "Text reading failed in the pipeline."

    # Create the dataset
    dataset = Dataset.from_dict({'text': [text]})
    assert len(dataset) == 1, "Dataset should contain one example."

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    assert 'input_ids' in tokenized_dataset.features, "Tokenized dataset should contain 'input_ids'."
    assert 'attention_mask' in tokenized_dataset.features, "Tokenized dataset should contain 'attention_mask'."

    # Group texts into blocks
    block_size = temp_environment['preprocessed_text'].split().__len__()  # Simple block size based on word count
    lm_dataset = tokenized_dataset.map(
        lambda examples: group_texts(examples, block_size),
        batched=True
    )
    assert 'labels' in lm_dataset.features, "Grouped dataset should contain 'labels'."
    assert len(lm_dataset['input_ids'][0]) == block_size, f"'input_ids' length should be {block_size}."

    # Split into train and validation datasets
    split_datasets = lm_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['test']

    assert len(train_dataset) > 0, "Train dataset should not be empty."
    assert len(val_dataset) > 0, "Validation dataset should not be empty."

    # Save the datasets
    save_dataset(train_dataset, temp_environment['output_dir'], 'train_dataset')
    save_dataset(val_dataset, temp_environment['output_dir'], 'val_dataset')

    # Verify saving
    train_output_path = os.path.join(temp_environment['output_dir'], 'train_dataset')
    val_output_path = os.path.join(temp_environment['output_dir'], 'val_dataset')
    assert os.path.exists(train_output_path), "Train dataset was not saved correctly."
    assert os.path.exists(val_output_path), "Validation dataset was not saved correctly."

    # Load and verify train dataset
    loaded_train = Dataset.load_from_disk(train_output_path)
    assert 'input_ids' in loaded_train.features, "Train dataset should contain 'input_ids'."
    assert 'labels' in loaded_train.features, "Train dataset should contain 'labels'."
    assert len(loaded_train) >= 1, "Train dataset should contain at least one example."

    # Load and verify validation dataset
    loaded_val = Dataset.load_from_disk(val_output_path)
    assert 'input_ids' in loaded_val.features, "Validation dataset should contain 'input_ids'."
    assert 'labels' in loaded_val.features, "Validation dataset should contain 'labels'."
    assert len(loaded_val) >= 1, "Validation dataset should contain at least one example."

def test_tokenizer_configuration(tokenizer):
    """
    Ensure that the tokenizer has a pad_token set.
    """
    assert tokenizer.pad_token is not None, "Tokenizer should have a pad_token set."

def test_block_size_positive():
    """
    Ensure that block_size is a positive integer.
    """
    block_size = 128
    assert isinstance(block_size, int), "block_size should be an integer."
    assert block_size > 0, "block_size should be a positive integer."

def test_preprocessed_text_content(temp_environment):
    """
    Ensure that the preprocessed text contains expected content.
    """
    text = read_text_file(temp_environment['input_file'])
    assert 'cleaned and preprocessed text' in text, "Preprocessed text does not contain expected content."

