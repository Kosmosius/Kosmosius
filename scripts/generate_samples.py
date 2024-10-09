#!/usr/bin/env python3
"""
generate_samples.py

A script to generate text samples using a fine-tuned language model with Hugging Face Transformers.

Usage:
    Interactive mode:
        python generate_samples.py --model_dir path/to/model_dir

    Generate from prompt:
        python generate_samples.py --model_dir path/to/model_dir --prompt "Your prompt here"

    Generate from prompts in a file:
        python generate_samples.py --model_dir path/to/model_dir --input_file path/to/prompts.txt --output_file path/to/output.txt

Dependencies:
    - transformers
    - torch

Install dependencies using:
    pip install transformers torch
"""

import os
import argparse
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    set_seed,
)
import torch

def setup_logging(log_file='generate_samples.log'):
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
    parser = argparse.ArgumentParser(description='Generate text samples using a fine-tuned language model.')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path to the fine-tuned model directory.'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='Prompt text for text generation.'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Path to a file containing prompts (one per line).'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Path to save generated texts.'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of generated text.'
    )
    parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=1,
        help='Number of generated sequences to return per prompt.'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature.'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling.'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help='Nucleus sampling (top-p).'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.'
    )
    return parser.parse_args()

def generate_text(pipeline, prompt, args):
    """
    Generates text using the provided pipeline and prompt.
    """
    outputs = pipeline(
        prompt,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        pad_token_id=pipeline.tokenizer.eos_token_id,
    )
    return outputs

def main():
    # Setup logging
    setup_logging()

    # Parse command-line arguments
    args = parse_arguments()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    # Initialize the text generation pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    logging.info(f"Model loaded from {args.model_dir}")

    # Interactive mode
    if not args.prompt and not args.input_file:
        logging.info("Entering interactive mode. Type your prompt and press Enter.")
        try:
            while True:
                prompt = input("Prompt: ")
                if prompt.strip().lower() in ['exit', 'quit']:
                    print("Exiting.")
                    break
                outputs = generate_text(pipeline, prompt, args)
                for i, output in enumerate(outputs):
                    print(f"\nGenerated Text {i+1}:\n{output['generated_text']}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
    else:
        # Generate from prompt
        if args.prompt:
            logging.info(f"Generating text for prompt: {args.prompt}")
            outputs = generate_text(pipeline, args.prompt, args)
            for i, output in enumerate(outputs):
                print(f"\nGenerated Text {i+1}:\n{output['generated_text']}\n")

            # Save outputs if output_file is specified
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for i, output in enumerate(outputs):
                        f.write(f"Prompt: {args.prompt}\n")
                        f.write(f"Generated Text {i+1}:\n{output['generated_text']}\n\n")
                logging.info(f"Generated texts saved to {args.output_file}")

        # Generate from prompts in a file
        if args.input_file:
            if not os.path.exists(args.input_file):
                logging.error(f"Input file not found at {args.input_file}. Exiting.")
                exit(1)
            with open(args.input_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(prompts)} prompts from {args.input_file}")

            all_outputs = []
            for idx, prompt in enumerate(prompts):
                logging.info(f"Generating text for prompt {idx+1}/{len(prompts)}: {prompt}")
                outputs = generate_text(pipeline, prompt, args)
                all_outputs.append({'prompt': prompt, 'outputs': outputs})

                # Print outputs
                for i, output in enumerate(outputs):
                    print(f"\nPrompt {idx+1}: {prompt}")
                    print(f"Generated Text {i+1}:\n{output['generated_text']}\n")

            # Save outputs if output_file is specified
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for item in all_outputs:
                        f.write(f"Prompt: {item['prompt']}\n")
                        for i, output in enumerate(item['outputs']):
                            f.write(f"Generated Text {i+1}:\n{output['generated_text']}\n\n")
                logging.info(f"Generated texts saved to {args.output_file}")

    logging.info("Text generation completed.")

if __name__ == "__main__":
    main()
