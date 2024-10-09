#!/usr/bin/env python3
"""
internet_archive_download.py

A script to download text files of literary works from the Internet Archive based on authors and their works specified in a JSON file.

Usage:
    python internet_archive_download.py --works_file path/to/works.json --download_dir path/to/download_directory

Dependencies:
    - internetarchive
    - tqdm

Install dependencies using:
    pip install internetarchive tqdm
"""

import internetarchive as ia
import os
import json
import argparse
import logging
from tqdm import tqdm
import sys

def setup_logging(log_file='internet_archive_download.log'):
    """
    Sets up logging for the script.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',
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
    parser = argparse.ArgumentParser(description='Download literary works from the Internet Archive.')
    parser.add_argument(
        '--works_file',
        type=str,
        default='data/scripts/works.json',
        help='Path to the works.json file containing authors and their works.'
    )
    parser.add_argument(
        '--download_dir',
        type=str,
        default='./downloaded_books',
        help='Directory where downloaded books will be saved.'
    )
    return parser.parse_args()

def load_works(file_path):
    """
    Loads the works from a JSON file.
    """
    if not os.path.exists(file_path):
        logging.error(f"works.json file not found at {file_path}. Exiting.")
        sys.exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            works_data = json.load(f)
        logging.info(f"Loaded works from {file_path}.")
        return works_data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        sys.exit(1)

def create_directory(path):
    """
    Creates a directory if it doesn't exist.
    """
    try:
        os.makedirs(path, exist_ok=True)
        logging.debug(f"Directory ensured at: {path}")
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        sys.exit(1)

def sanitize_filename(filename):
    """
    Sanitizes the filename by removing or replacing invalid characters.
    """
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()

def search_item(author, title):
    """
    Searches for an item in the Internet Archive based on author and title.

    Returns:
        The first matching item's identifier or None if not found.
    """
    query = f'creator:"{author}" AND title:"{title}" AND mediatype:texts AND language:"English"'
    search_results = ia.search_items(query, fields=['identifier'], max_results=5)
    results = list(search_results)
    if results:
        # Return the first matching identifier
        return results[0]['identifier']
    else:
        logging.warning(f"No results found for '{title}' by {author}.")
        return None

def download_text(identifier, author, title, save_path):
    """
    Downloads text files for a given identifier from the Internet Archive.

    Args:
        identifier (str): The Internet Archive identifier.
        author (str): Author's name.
        title (str): Title of the work.
        save_path (str): Path where the text file will be saved.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    try:
        # Search for available files with .txt extension
        item = ia.get_item(identifier)
        files = item.get_files()
        txt_files = [file for file in files if file['format'] == 'Text']

        if not txt_files:
            logging.warning(f"No .txt files found for '{title}' by {author} (ID: {identifier}).")
            return False

        for txt_file in txt_files:
            # Download each .txt file
            file_name = txt_file['name']
            logging.info(f"Downloading '{file_name}' for '{title}' by {author}...")
            item.download(files=[file_name], destdir=os.path.dirname(save_path), ignore_existing=True)
        return True
    except Exception as e:
        logging.error(f"Error downloading '{title}' by {author} (ID: {identifier}): {e}")
        return False

def main():
    # Setup logging
    setup_logging()

    # Parse command-line arguments
    args = parse_arguments()

    works_file = args.works_file
    download_directory = args.download_dir

    # Load works data
    works_data = load_works(works_file)

    # Ensure download directory exists
    create_directory(download_directory)

    # Prepare a list of all author-work pairs
    author_work_list = []
    for author, works in works_data.items():
        for work in works:
            author_work_list.append({
                'author': author,
                'title': work['title']
            })

    logging.info(f"Starting download of {len(author_work_list)} works.")

    # Iterate through each author and work with a progress bar
    for work in tqdm(author_work_list, desc="Downloading Works"):
        author = work['author']
        title = work['title']
        logging.info(f"Processing '{title}' by {author}.")

        # Search for the item's identifier
        identifier = search_item(author, title)

        if identifier:
            # Define author-specific directory
            author_dir = os.path.join(download_directory, sanitize_filename(author))
            create_directory(author_dir)

            # Define save path for the text file
            sanitized_title = sanitize_filename(title)
            save_filename = f"{sanitized_title}.txt"
            save_path = os.path.join(author_dir, save_filename)

            # Download the text
            success = download_text(identifier, author, title, save_path)

            if success:
                logging.info(f"Successfully downloaded '{title}' by {author}.")
            else:
                logging.warning(f"Failed to download '{title}' by {author}.")
        else:
            logging.warning(f"Identifier not found for '{title}' by {author}. Skipping download.")

    logging.info("Download process completed.")

if __name__ == "__main__":
    main()
