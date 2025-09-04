import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import wget

sys.path.append(".")
from paths import DATA_FOLDER

basepath = str(DATA_FOLDER)

print("Looking for files in", basepath, "\n")

zenodo_doi = "17058315"

zenodo_record = f"https://zenodo.org/records/{zenodo_doi}"
zenodo_file_api = f"https://zenodo.org/api/records/{zenodo_doi}/files-archive"
zenodo_individual_files_api = f"https://zenodo.org/records/{zenodo_doi}/files"

zipfiles = [
    'converted_data.zip',
    'depletion_voltage_width.zip',
    'drive_sweep.zip',
    'long_time_record.zip',
    'processed_data.zip',
    'qdmzm.zip',
    'wpwp_0.zip',
    'wpwp_1.zip',
    'wpwp_2.zip',
    'wpwp_3.zip',
    'wpwp_4.zip',
    'wpwp_5.zip',
    'wpwp_6.zip',
    'wpwp_7.zip',
    'xmpr.zip',
    'zmpr.zip'
]

minimal_files = [
    "converted_data.zip",
    "processed_data.zip",
    "depletion_voltage_width.zip"
]

# Edit this list to customize which files to download when using option 3
custom_files = [
]

# Ensure custom_files and minimal_files are subsets of zipfiles
invalid_minimal = set(minimal_files) - set(zipfiles)
invalid_custom = set(custom_files) - set(zipfiles)
if invalid_minimal:
    print(f"⚠️  Error: minimal_files contains invalid entries: {', '.join(invalid_minimal)}")
    sys.exit(1)
if invalid_custom:
    print(f"⚠️  Error: custom_files contains invalid entries: {', '.join(invalid_custom)}")
    sys.exit(1)

def print_header():
    """Print formatted header with dataset information."""
    print("=" * 80)
    print("QUANTUM TETRON LIFETIMES - DATA PREPARATION")
    print("=" * 80)
    print(f"\nDataset available at: {zenodo_record}")
    print("You can view all available files and their descriptions at the link above.\n")

def print_file_options():
    """Print the available download options."""
    print("DOWNLOAD OPTIONS:")
    print("-" * 40)
    print("0. Skip download (I already have the files I need)")
    print("1. Download Minimal Dataset (~5GB)")
    print(f"   Files: {', '.join(minimal_files)}")
    print("2. Download All Datasets (~100GB)")
    print(f"   Files: All {len(zipfiles)} available zip files")
    print("3. Download Custom Set")
    print(f"   Files: {', '.join(custom_files)}")
    print("   (Edit 'custom_files' variable in prepare_data.py to customize)")
    print("-" * 40)

def get_user_choice():
    """Get and validate user's download choice."""
    while True:
        try:
            choice = input("\nPlease enter your choice (0-3): ").strip()
            choice_int = int(choice)
            if 0 <= choice_int <= 3:
                return choice_int
            else:
                print("Invalid choice. Please enter a number between 0 and 3.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 3.")

def download_files(files_to_download, description):
    """Download the specified files."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING {description.upper()}")
    print(f"{'='*60}")
    
    downloaded_count = 0
    skipped_count = 0
    
    for file_name in files_to_download:
        file_path = f"{basepath}/{file_name}"
        if not is_zipfile(file_path):
            print(f"\n[{downloaded_count + 1}/{len(files_to_download)}] Downloading {file_name}...")
            try:
                _ = wget.download(
                    f"{zenodo_individual_files_api}/{file_name}",
                    out=file_path,
                )
                print()  # New line after wget progress bar
                downloaded_count += 1
            except Exception as e:
                print(f"\nError downloading {file_name}: {e}")
        else:
            print(f"✓ Found {file_name}, skipping download")
            skipped_count += 1
    
    print(f"\nDownload complete! Downloaded: {downloaded_count}, Skipped: {skipped_count}")

def process_file(filename):
    full_path = os.path.join(basepath, filename)
    if not is_zipfile(full_path):
        if filename in minimal_files:
            print(f"⚠️  Cannot find {filename} (required for paper figures)")
        else:
            print(f"⚠️  Skip optional {filename} (needed for Cq converted data)")
    else:
        print(f"📦 Extracting {filename}...")
        with ZipFile(full_path, "r") as myzip:
            myzip.extractall(path=basepath)
        print(f"✓ Extracted {filename}")

if __name__ == "__main__":
    # Handle command line arguments
    if "--download-all" in sys.argv:
        choice = 2
    elif "--download-minimal" in sys.argv:
        choice = 1
    elif "--download-custom" in sys.argv:
        choice = 3
    elif "--no-download" in sys.argv:
        choice = 0
    else:
        # Interactive mode
        print_header()
        print("To run the code in this repository and reproduce the paper figures,")
        print("you need access to the measurement and simulation data.\n")
        
        print_file_options()
        choice = get_user_choice()

    # Process the user's choice
    if choice == 0:
        print("\n✓ Skipping download. Proceeding with file extraction...")
    else:
        # Create data directory if it doesn't exist
        Path(basepath).mkdir(exist_ok=True)
        
        if choice == 1:
            download_files(minimal_files, "Minimal Dataset")
        elif choice == 2:
            download_files(zipfiles, "All Datasets")
        elif choice == 3:
            if not custom_files:
                print("\n⚠️  Custom files list is empty. Please edit 'custom_files' variable in prepare_data.py")
                sys.exit(1)
            download_files(custom_files, "Custom Dataset")

    # Extract files
    print(f"\n{'='*60}")
    print("EXTRACTING FILES AND SETTING UP DIRECTORY STRUCTURE")
    print(f"{'='*60}")

    with ProcessPoolExecutor() as executor:
        executor.map(process_file, zipfiles)
    
    print(f"\n{'='*60}")
    print("✓ DATA PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Data location: {basepath}")
    print(f"For more information, visit: {zenodo_record}")