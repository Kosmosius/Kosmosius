# Kosmosius

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![GitHub Actions](https://github.com/yourusername/Kosmosius/workflows/CI/badge.svg)

**Kosmosius** is a project dedicated to fine-tuning a Large Language Model (LLM) on a curated corpus of great literature using HuggingFace's Transformers and PEFT (Parameter-Efficient Fine-Tuning) techniques. Leveraging a single NVIDIA GeForce RTX 3070 Ti GPU, Kosmosius aims to capture the stylistic nuances and thematic elements of classical and modern literary works, making advanced language modeling accessible and efficient.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Setting Up the Environment](#setting-up-the-environment)
- [Data Handling](#data-handling)
  - [Downloading Data](#downloading-data)
  - [Data Preprocessing](#data-preprocessing)
- [Model Fine-Tuning](#model-fine-tuning)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Generating Text Samples](#generating-text-samples)
- [Usage](#usage)
  - [Loading the Fine-Tuned Model](#loading-the-fine-tuned-model)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The goal of Kosmosius is to fine-tune a pre-trained LLM on a diverse set of literary works, spanning ancient texts to modern literature. By employing PEFT techniques like LoRA (Low-Rank Adaptation), Kosmosius efficiently adapts large models within the constraints of a single GPU setup. This project not only serves as a practical example of LLM fine-tuning but also provides insights into handling literary corpora for advanced language modeling tasks.

## Repository Structure

```
Kosmosius/
├── data/
│   ├── raw/                     # Placeholder for raw data (not hosted)
│   ├── processed/               # Placeholder for processed data (not hosted)
│   ├── scripts/
│   │   └── download_data.py     # Script to download and organize data locally
│   └── README.md                # Documentation about data sources and usage
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Notebook for exploring the raw data
│   ├── 02_preprocessing.ipynb       # Notebook for data cleaning and preprocessing
│   └── 03_model_training.ipynb      # Notebook for experimenting with model training
├── scripts/
│   ├── preprocess_data.py           # Script to preprocess raw data
│   ├── train_model.py               # Script to fine-tune the LLM
│   ├── evaluate_model.py            # Script to evaluate the fine-tuned model
│   ├── generate_samples.py          # Script to generate text samples from the model
│   └── utils.py                     # Utility functions used across scripts
├── models/
│   ├── fine-tuned-model/            # Directory to store the fine-tuned model
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer/
│   │   │   ├── tokenizer_config.json
│   │   │   ├── vocab.json
│   │   │   └── merges.txt
│   │   └── README.md                # Documentation about the fine-tuned model
│   └── README.md                    # Overview of all fine-tuned models
├── configs/
│   ├── training_config.yaml         # Configuration for model training
│   ├── preprocessing_config.yaml    # Configuration for data preprocessing
│   └── README.md                    # Documentation about configuration files
├── datasets/
│   ├── dataset_script.py            # HuggingFace dataset loading script
│   └── README.md                    # Documentation about the dataset script
├── tests/
│   ├── test_preprocessing.py        # Tests for the preprocessing script
│   ├── test_training.py             # Tests for the training script
│   ├── test_evaluation.py           # Tests for the evaluation script
│   └── test_generate_samples.py     # Tests for the sample generation script
├── docker/
│   ├── Dockerfile                   # Docker configuration for environment reproducibility
│   └── README.md                    # Documentation about Docker setup
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions workflow for CI/CD
├── .gitignore                        # Specifies intentionally untracked files to ignore
├── README.md                         # Main project documentation
├── LICENSE                           # Project license (MIT)
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment configuration
├── setup.py                          # Setup script for installing the project as a package
└── CONTRIBUTING.md                    # Guidelines for contributing to the project
```

## Features

- **Parameter-Efficient Fine-Tuning (PEFT):** Utilizes LoRA to adapt large language models efficiently within limited GPU memory.
- **Modular Scripts:** Organized scripts for data downloading, preprocessing, training, evaluation, and sample generation.
- **Comprehensive Notebooks:** Interactive Jupyter notebooks for data exploration, preprocessing, and model training.
- **Automated Testing:** Ensures reliability and correctness of scripts through automated tests using `pytest`.
- **Docker Support:** Facilitates environment reproducibility and ease of setup using Docker containers.
- **Continuous Integration (CI):** GitHub Actions workflows automate testing and linting on every commit.
- **HuggingFace Integration:** Seamless compatibility with HuggingFace’s Transformers and Datasets libraries, enabling easy model and dataset sharing.

## Getting Started

### Prerequisites

- **Hardware:**
  - NVIDIA GeForce RTX 3070 Ti with 8 GB VRAM
- **Software:**
  - Python 3.10 or higher
  - Git
  - Docker (optional, for containerization)
  - Conda (optional, for environment management)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Kosmosius.git
   cd Kosmosius

2. **Set Up the Environment:**

You can choose between venv or conda for environment management.

- Using venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
  
- Using Conda:
    ```bash
   conda env create -f environment.yml
   conda activate Kosmosius

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

### Setting Up the Environment
Ensure all dependencies are correctly installed. If you encounter any issues, refer to the `environment.yml` or `requirements.txt` for required packages and their versions.

## Data Handling
### Downloading Data
Since Kosmosius does not host the data, you need to download it locally using the provided scripts.

1. Navigate to the Data Scripts Directory:

    ```bash
    cd data/scripts
2. Run the Data Download Script:

    ```bash
    python download_data.py

- **Function:** Automates the downloading of literature works from sources like Project Gutenberg and Internet Archive.
- **Output:** Downloads and organizes data into the data/raw/ directory.

3. Verify Downloaded Data:

Check the `data/raw/` directory to ensure that the data has been downloaded and organized correctly.

### Data Preprocessing
After downloading the data, preprocess it to prepare for model training.

1. Run the Preprocessing Script:

    ```bash
    cd ../../scripts
    python preprocess_data.py --input_dir ../data/raw/ --output_dir ../data/processed/
    
- **Function:** Cleans and preprocesses raw data into a format suitable for training.
- **Output:** Processed data stored in the data/processed/ directory.

2. Alternative: Use Jupyter Notebook

You can also use the interactive notebook for preprocessing:
    
    ```
    jupyter notebook ../notebooks/02_preprocessing.ipynb

## Model Fine-Tuning
### Training the Model
Fine-tune the selected LLM using the provided training scripts and configurations.

1. Configure Training Parameters:

Modify the configs/training_config.yaml file to adjust hyperparameters as needed.

2. Run the Training Script:

    ```bash
    python scripts/train_model.py --config configs/training_config.yaml
- **Function:** Fine-tunes the LLM using HuggingFace’s Transformers and PEFT techniques.
- **Output:** Fine-tuned model saved in the models/fine-tuned-model/ directory.

3. Alternative: Use Jupyter Notebook

You can also use the interactive notebook for training:

    ```
    jupyter notebook ../notebooks/03_model_training.ipynb

### Evaluating the Model
Assess the performance of the fine-tuned model using evaluation scripts.

1. Run the Evaluation Script:

    ```bash
    python scripts/evaluate_model.py --model_dir models/fine-tuned-model/ --data_dir data/processed/test/
- **Function:** Evaluates the fine-tuned model using metrics like perplexity.
- **Output:** Evaluation results displayed in the console.

## Generating Text Samples
Generate text samples to qualitatively assess the fine-tuned model's capabilities.

1. Run the Sample Generation Script:

    ```bash
    python scripts/generate_samples.py --model_dir models/fine-tuned-model/ --prompt "Once upon a time" --max_length 150 --num_samples 3

**Parameters:**
--model_dir: Path to the fine-tuned model directory.
--prompt: Text prompt to initiate generation.
--max_length: Maximum length of the generated text.
--num_samples: Number of samples to generate.

**Output:** Generated text samples displayed in the console.

### Usage
#### Loading the Fine-Tuned Model
You can load and utilize the fine-tuned model in your own scripts or applications using HuggingFace’s Transformers library.

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('models/fine-tuned-model/')
    model = AutoModelForCausalLM.from_pretrained('models/fine-tuned-model/')

    # Encode prompt
    prompt = "In a distant future,"
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

## Contributing
Contributions are welcome! Please read our Contributing Guide for guidelines on how to get started.

## License
This project is licensed under the MIT License.

## Acknowledgements
[HuggingFace Transformers](https://github.com/huggingface/transformers)\
[HuggingFace Datasets](https://github.com/huggingface/datasets)\
[Parameter-Efficient Fine-Tuning (PEFT)](https://github.com/huggingface/peft)\
[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)\
[Project Gutenberg](https://www.gutenberg.org/)\
[Internet Archive](https://archive.org/)
