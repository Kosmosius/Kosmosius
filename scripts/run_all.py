#!/usr/bin/env python3
"""
run_all.py

A master script to automate the entire process:
1. Preprocess raw text data.
2. Create datasets for language model training.
3. Fine-tune a language model.
4. Evaluate the fine-tuned model.
5. Generate sample texts.

Usage:
    python run_all.py --config configs/pipeline_config.yaml

Dependencies:
    - PyYAML
    - All dependencies from individual scripts.

Install dependencies using:
    pip install PyYAML

Note:
    Ensure all individual scripts (preprocess_data.py, create_dataset.py, train_model.py, evaluate_model.py, generate_samples.py) are present in the 'scripts' directory.
"""

import argparse
import logging
import os
import subprocess
import sys
import yaml

def setup_logging(log_file='run_all.log'):
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
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Automate the entire NLP pipeline.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file.'
    )
    return parser.parse_args()

def load_config(config_path):
    """
    Loads the configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at {config_path}. Exiting.")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}.")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        sys.exit(1)

def run_script(command):
    """
    Runs a script as a subprocess.
    """
    try:
        logging.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, text=True)
        logging.info(f"Command completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running the command: {e}")
        sys.exit(1)

def main():
    # Setup logging
    setup_logging()

    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Step 1: Preprocess Data
    logging.info("Step 1: Preprocessing data.")
    preprocess_config = config['preprocess']
    preprocess_command = [
        'python', 'scripts/preprocess_data.py',
        '--input_file', preprocess_config['input_file'],
        '--output_dir', preprocess_config['output_dir'],
        '--model_name', preprocess_config.get('model_name', 'gpt2'),
        '--block_size', str(preprocess_config.get('block_size', 1024))
    ]
    run_script(preprocess_command)

    # Step 2: Create Dataset
    logging.info("Step 2: Creating dataset.")
    dataset_config = config['create_dataset']
    create_dataset_command = [
        'python', 'scripts/create_dataset.py',
        '--input_file', dataset_config['input_file'],
        '--output_dir', dataset_config['output_dir'],
        '--model_name', dataset_config.get('model_name', 'gpt2'),
        '--block_size', str(dataset_config.get('block_size', 1024))
    ]
    run_script(create_dataset_command)

    # Step 3: Train Model
    logging.info("Step 3: Training model.")
    train_config = config['train']
    train_command = [
        'python', 'scripts/train_model.py',
        '--train_dataset', train_config['train_dataset'],
        '--val_dataset', train_config['val_dataset'],
        '--output_dir', train_config['output_dir'],
        '--model_name', train_config.get('model_name', 'gpt2'),
        '--num_train_epochs', str(train_config.get('num_train_epochs', 3)),
        '--per_device_train_batch_size', str(train_config.get('per_device_train_batch_size', 1)),
        '--per_device_eval_batch_size', str(train_config.get('per_device_eval_batch_size', 1)),
        '--learning_rate', str(train_config.get('learning_rate', 5e-5)),
        '--logging_steps', str(train_config.get('logging_steps', 100)),
        '--save_steps', str(train_config.get('save_steps', 500)),
        '--eval_steps', str(train_config.get('eval_steps', 500)),
        '--seed', str(train_config.get('seed', 42))
    ]
    if train_config.get('fp16', False):
        train_command.append('--fp16')
    run_script(train_command)

    # Step 4: Evaluate Model
    logging.info("Step 4: Evaluating model.")
    evaluate_config = config['evaluate']
    evaluate_command = [
        'python', 'scripts/evaluate_model.py',
        '--model_dir', evaluate_config['model_dir'],
        '--val_dataset', evaluate_config['val_dataset'],
        '--output_dir', evaluate_config['output_dir'],
        '--per_device_eval_batch_size', str(evaluate_config.get('per_device_eval_batch_size', 1)),
        '--seed', str(evaluate_config.get('seed', 42))
    ]
    run_script(evaluate_command)

    # Step 5: Generate Samples
    logging.info("Step 5: Generating samples.")
    generate_config = config['generate']
    generate_command = [
        'python', 'scripts/generate_samples.py',
        '--model_dir', generate_config['model_dir'],
        '--output_file', generate_config['output_file'],
        '--max_length', str(generate_config.get('max_length', 100)),
        '--num_return_sequences', str(generate_config.get('num_return_sequences', 1)),
        '--temperature', str(generate_config.get('temperature', 1.0)),
        '--top_k', str(generate_config.get('top_k', 50)),
        '--top_p', str(generate_config.get('top_p', 0.95)),
        '--seed', str(generate_config.get('seed', 42))
    ]
    if generate_config.get('prompt'):
        generate_command.extend(['--prompt', generate_config['prompt']])
    elif generate_config.get('input_file'):
        generate_command.extend(['--input_file', generate_config['input_file']])
    else:
        logging.error("No prompt or input_file specified for text generation.")
        sys.exit(1)
    run_script(generate_command)

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
