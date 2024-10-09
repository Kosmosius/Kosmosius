import internetarchive as ia
import os

# List of author names you're interested in
authors_list = ["Plato"]  # Example authors

# Directory where you want to save the downloaded books
download_directory = "./downloaded_books"

# Check if the download directory exists, create it if it doesn't
if not os.path.exists(download_directory):
    os.makedirs(download_directory)
    print(f"Created directory: {download_directory}")

for author in authors_list:
    # Construct the query string with filters for texts in English
    query_string = f'creator:({author}) AND mediatype:"texts" AND language:"English"'
    
    # Search for items that match the query
    search_results = ia.search_items(query_string)
    
    for result in search_results:
        # Each result is a dictionary. Get the identifier of the item.
        identifier = result['identifier']
        
        # Download only text files (.txt) for the item based on its identifier.
        print(f"Downloading English text files by {author} with identifier {identifier}...")
        ia.download(identifier, destdir=download_directory, ignore_existing=True, glob_pattern='*.txt')

print("Download completed.")
