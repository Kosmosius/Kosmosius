def main():
    print("Starting the download process for Kosmosius...")

    # Define the base data directory
    base_data_dir = os.path.join(os.getcwd(), "data", "raw")
    create_directory(base_data_dir)

    # Iterate through each author and their works
    for author, works in AUTHORS_WORKS.items():
        author_dir = os.path.join(base_data_dir, sanitize_filename(author))
        create_directory(author_dir)

        for work in works:
            title = work["title"]
            gutenberg_id = work["gutenberg_id"]
            sanitized_title = sanitize_filename(title)
            save_filename = f"{sanitized_title}.txt"
            save_path = os.path.join(author_dir, save_filename)

            if gutenberg_id:
                # Proceed to download
                success = download_text(gutenberg_id, author, title, save_path)
            else:
                print(f"Gutenberg ID not available for '{title}' by {author}. Skipping download.")
                logging.warning(f"Gutenberg ID not available for '{title}' by {author}. Skipping download.")
                continue

            if not success:
                print(f"Failed to download: {title} by {author}. Check log for details.")
            else:
                print(f"Downloaded: {title} by {author}.")

            # Respectful delay to avoid overwhelming the server
            time.sleep(1)  # 1-second delay

    print("Download process completed. Check 'download_data.log' for details.")
