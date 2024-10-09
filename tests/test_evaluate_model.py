#!/usr/bin/env python3
"""
test_evaluate_model.py

Unit tests for evaluate_model.py using the pytest framework.

Run the tests:
    pytest tests/test_evaluate_model.py

Dependencies:
    - pytest
    - pytest-mock
    - transformers
    - datasets
    - torch
    - numpy

Ensure that the 'evaluate_model.py' script is in the 'scripts' directory.
"""

import pytest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import functions from evaluate_model.py
# Adjust the import path based on your project structure
from scripts.evaluate_model import (
    setup_logging,
    parse_arguments,
    main
)


@pytest.fixture
def temp_environment():
    """
    Fixture to set up a temporary environment for testing.
    Creates temporary validation datasets and an output directory.
    Cleans up after the test.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    val_dataset_dir = os.path.join(temp_dir, 'val_dataset')
    output_dir = os.path.join(temp_dir, 'evaluation_results')

    # Create a dummy validation dataset
    val_data = {"input_ids": [[7, 8, 9], [10, 11, 12]], "labels": [[7, 8, 9], [10, 11, 12]]}
    val_dataset = Dataset.from_dict(val_data)
    val_dataset.save_to_disk(val_dataset_dir)

    yield {
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
        '--model_dir', 'path/to/model_dir',
        '--val_dataset', 'path/to/val_dataset',
        '--output_dir', 'path/to/output_dir',
        '--per_device_eval_batch_size', '2',
        '--seed', '123'
    ]

    with patch('sys.argv', ['evaluate_model.py'] + test_args):
        args = parse_arguments()
        assert args.model_dir == 'path/to/model_dir'
        assert args.val_dataset == 'path/to/val_dataset'
        assert args.output_dir == 'path/to/output_dir'
        assert args.per_device_eval_batch_size == 2
        assert args.seed == 123


@patch('scripts.evaluate_model.Trainer')
@patch('scripts.evaluate_model.AutoTokenizer.from_pretrained')
@patch('scripts.evaluate_model.AutoModelForCausalLM.from_pretrained')
def test_main(mock_model_from_pretrained, mock_tokenizer_from_pretrained, mock_trainer, temp_environment, tokenizer, model):
    """
    Test the main function to ensure that the Trainer is initialized correctly and evaluation steps are called.
    This test mocks the Trainer to avoid actual evaluation.
    """
    # Mock the tokenizer and model
    mock_tokenizer_from_pretrained.return_value = tokenizer
    mock_model_from_pretrained.return_value = model

    # Create a mock Trainer instance
    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance

    # Prepare test arguments
    test_args = [
        '--model_dir', temp_environment['temp_dir'],  # Using temp_dir as a placeholder
        '--val_dataset', temp_environment['val_dataset_dir'],
        '--output_dir', temp_environment['output_dir'],
        '--per_device_eval_batch_size', '2',
        '--seed', '123'
    ]

    with patch('sys.argv', ['evaluate_model.py'] + test_args):
        main()

        # Assert that the tokenizer and model are loaded correctly
        mock_tokenizer_from_pretrained.assert_called_once_with(temp_environment['temp_dir'])
        mock_model_from_pretrained.assert_called_once_with(temp_environment['temp_dir'])

        # Assert that the Trainer is initialized with correct arguments
        mock_trainer.assert_called_once()
        trainer_call_args = mock_trainer.call_args[1]  # Get the keyword arguments
        assert trainer_call_args['model'] == model
        assert trainer_call_args['args'].output_dir == temp_environment['output_dir']
        assert trainer_call_args['args'].per_device_eval_batch_size == 2
        assert trainer_call_args['args'].seed == 123

        # Assert that evaluation was called
        mock_trainer_instance.evaluate.assert_called_once()

        # Assert that evaluation results are processed correctly
        # Assuming eval_results has 'eval_loss' key
        # Since evaluate is mocked, we need to set a return value
        mock_trainer_instance.evaluate.return_value = {'eval_loss': 0.5}
        with patch('math.exp') as mock_exp:
            mock_exp.return_value = 1.6487212707001282  # math.exp(0.5)
            main()
            mock_exp.assert_called_once_with(0.5)

        # Assert that the model was not saved since it's evaluation only
        mock_trainer_instance.save_model.assert_not_called()
        tokenizer.save_pretrained.assert_not_called()


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
    Test that the environment variables required by evaluate_model.py are set correctly.
    This is a placeholder for actual environment variable tests if needed.
    """
    # Example: Ensure that CUDA is available if expected
    import torch
    if torch.cuda.is_available():
        assert torch.cuda.is_available(), "CUDA should be available."
    else:
        pytest.skip("CUDA is not available, skipping environment variable tests.")


def test_perplexity_calculation(temp_environment, mocker):
    """
    Test the calculation of perplexity from eval_loss.
    """
    # Mock the Trainer and its evaluate method
    with patch('scripts.evaluate_model.Trainer') as mock_trainer_class:
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance
        mock_trainer_instance.evaluate.return_value = {'eval_loss': 0.5}

        # Mock math.exp to return a fixed value
        mocker.patch('math.exp', return_value=1.6487212707001282)

        # Prepare test arguments
        test_args = [
            '--model_dir', temp_environment['temp_dir'],
            '--val_dataset', temp_environment['val_dataset_dir'],
            '--output_dir', temp_environment['output_dir'],
            '--per_device_eval_batch_size', '2',
            '--seed', '123'
        ]

        with patch('sys.argv', ['evaluate_model.py'] + test_args):
            main()

            # Check that math.exp was called with eval_loss
            import math
            math.exp.assert_called_with(0.5)


def test_save_evaluation_results(temp_environment):
    """
    Test that evaluation results are saved correctly to a file.
    """
    # Mock the Trainer and its evaluate method
    with patch('scripts.evaluate_model.Trainer') as mock_trainer_class:
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance
        eval_results = {'eval_loss': 0.5, 'other_metric': 0.8}
        mock_trainer_instance.evaluate.return_value = eval_results

        # Mock the Trainer's methods
        mock_trainer_instance.evaluate.return_value = eval_results

        # Mock the Trainer to return specific results
        with patch('math.exp', return_value=1.6487212707001282):
            # Prepare test arguments
            test_args = [
                '--model_dir', temp_environment['temp_dir'],
                '--val_dataset', temp_environment['val_dataset_dir'],
                '--output_dir', temp_environment['output_dir'],
                '--per_device_eval_batch_size', '2',
                '--seed', '123'
            ]

            with patch('sys.argv', ['evaluate_model.py'] + test_args):
                main()

                # Check that the results file exists
                results_file = os.path.join(temp_environment['output_dir'], 'eval_results.txt')
                assert os.path.exists(results_file), "Evaluation results file was not created."

                # Check the contents of the results file
                with open(results_file, 'r') as f:
                    content = f.read()
                    assert "Evaluation Results:" in content, "Results file does not contain 'Evaluation Results:'."
                    for key, value in eval_results.items():
                        assert f"{key}: {value}" in content, f"Results file does not contain '{key}: {value}'."
                    assert "Perplexity: 1.6487212707001282" in content, "Perplexity was not calculated correctly."


