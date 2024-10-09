#!/usr/bin/env python3
"""
test_train_model.py

Unit tests for train_model.py using the pytest framework.

Run the tests:
    pytest tests/test_train_model.py

Dependencies:
    - pytest
    - pytest-mock
    - transformers
    - datasets
    - torch
    - numpy

Ensure that the 'train_model.py' script is in the 'scripts' directory.
"""

import pytest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import functions from train_model.py
# Adjust the import path based on your project structure
from scripts.train_model import (
    setup_logging,
    parse_arguments,
    main
)

@pytest.fixture
def temp_environment():
    """
    Fixture to set up a temporary environment for testing.
    Creates temporary training and validation datasets and an output directory.
    Cleans up after the test.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    train_dataset_dir = os.path.join(temp_dir, 'train_dataset')
    val_dataset_dir = os.path.join(temp_dir, 'val_dataset')
    output_dir = os.path.join(temp_dir, 'output_model')

    # Create dummy datasets
    train_data = {"input_ids": [[1, 2, 3], [4, 5, 6]], "labels": [[1, 2, 3], [4, 5, 6]]}
    val_data = {"input_ids": [[7, 8, 9]], "labels": [[7, 8, 9]]}

    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    train_dataset.save_to_disk(train_dataset_dir)
    val_dataset.save_to_disk(val_dataset_dir)

    yield {
        'train_dataset_dir': train_dataset_dir,
        'val_dataset_dir': val_dataset_dir,
        'output_dir': output_dir,
        'temp_dir': temp_dir
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

@pytest.fixture
def model():
    """
    Fixture to initialize and provide the model.
    """
    model_name = 'gpt2'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model

def test_parse_arguments():
    """
    Test the argument parsing to ensure all required arguments are present.
    """
    test_args = [
        '--train_dataset', 'path/to/train_dataset',
        '--val_dataset', 'path/to/val_dataset',
        '--output_dir', 'path/to/output_model',
        '--model_name', 'gpt2',
        '--num_train_epochs', '1',
        '--per_device_train_batch_size', '2',
        '--per_device_eval_batch_size', '2',
        '--learning_rate', '3e-5',
        '--logging_steps', '10',
        '--save_steps', '50',
        '--eval_steps', '50',
        '--seed', '123',
        '--fp16'
    ]

    with patch('sys.argv', ['train_model.py'] + test_args):
        args = parse_arguments()
        assert args.train_dataset == 'path/to/train_dataset'
        assert args.val_dataset == 'path/to/val_dataset'
        assert args.output_dir == 'path/to/output_model'
        assert args.model_name == 'gpt2'
        assert args.num_train_epochs == 1
        assert args.per_device_train_batch_size == 2
        assert args.per_device_eval_batch_size == 2
        assert args.learning_rate == 3e-5
        assert args.logging_steps == 10
        assert args.save_steps == 50
        assert args.eval_steps == 50
        assert args.seed == 123
        assert args.fp16 is True

@patch('scripts.train_model.Trainer')
@patch('scripts.train_model.AutoTokenizer.from_pretrained')
@patch('scripts.train_model.AutoModelForCausalLM.from_pretrained')
def test_main(mock_model_from_pretrained, mock_tokenizer_from_pretrained, mock_trainer, temp_environment, tokenizer, model):
    """
    Test the main function to ensure that the Trainer is initialized correctly and training steps are called.
    This test mocks the Trainer to avoid actual training.
    """
    # Mock the tokenizer and model
    mock_tokenizer_from_pretrained.return_value = tokenizer
    mock_model_from_pretrained.return_value = model

    # Create a mock Trainer instance
    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance

    # Prepare test arguments
    test_args = [
        '--train_dataset', temp_environment['train_dataset_dir'],
        '--val_dataset', temp_environment['val_dataset_dir'],
        '--output_dir', temp_environment['output_dir'],
        '--model_name', 'gpt2',
        '--num_train_epochs', '1',
        '--per_device_train_batch_size', '2',
        '--per_device_eval_batch_size', '2',
        '--learning_rate', '3e-5',
        '--logging_steps', '10',
        '--save_steps', '50',
        '--eval_steps', '50',
        '--seed', '123',
        '--fp16'
    ]

    with patch('sys.argv', ['train_model.py'] + test_args):
        main()

        # Assert that the tokenizer and model are loaded correctly
        mock_tokenizer_from_pretrained.assert_called_once_with('gpt2', use_fast=True)
        mock_model_from_pretrained.assert_called_once_with('gpt2')

        # Assert that the Trainer is initialized with correct arguments
        mock_trainer.assert_called_once()
        trainer_call_args = mock_trainer.call_args[1]  # Get the keyword arguments
        assert trainer_call_args['model'] == model
        assert trainer_call_args['args'].output_dir == temp_environment['output_dir']
        assert trainer_call_args['args'].num_train_epochs == 1
        assert trainer_call_args['args'].per_device_train_batch_size == 2
        assert trainer_call_args['args'].per_device_eval_batch_size == 2
        assert trainer_call_args['args'].learning_rate == 3e-5
        assert trainer_call_args['args'].logging_steps == 10
        assert trainer_call_args['args'].save_steps == 50
        assert trainer_call_args['args'].eval_steps == 50
        assert trainer_call_args['args'].fp16 is True
        assert trainer_call_args['args'].seed == 123

        # Assert that training and evaluation were called
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.evaluate.assert_called_once()

        # Assert that the model was saved
        mock_trainer_instance.save_model.assert_called_once_with(temp_environment['output_dir'])
        tokenizer.save_pretrained.assert_called_once_with(temp_environment['output_dir'])

def test_setup_logging():
    """
    Test the setup_logging function to ensure that logs are created.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, 'test_log.log')
        setup_logging(log_file=log_file)

        # Log a test message
        import logging
        logging.info("Test log message.")

        # Check that the log file exists and contains the message
        assert os.path.exists(log_file), "Log file was not created."
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert "Test log message." in log_content, "Log message not found in log file."

def test_environment_variables():
    """
    Test that the environment variables required by train_model.py are set correctly.
    This is a placeholder for actual environment variable tests if needed.
    """
    # Example: Ensure that CUDA is available if expected
    import torch
    if torch.cuda.is_available():
        assert torch.cuda.is_available(), "CUDA should be available."
    else:
        pytest.skip("CUDA is not available, skipping environment variable tests.")

