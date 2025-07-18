# unzip.py
# This script unpacks the dataset from the downloaded .zip files.

# --- Source and Pattern Citations ---
#
# 1. Use of os, zipfile, and shutil modules as per their standard documentation:
#    Reference: Python Official Documentation:
#      - "os — Miscellaneous operating system interfaces" https://docs.python.org/3/library/os.html
#      - "zipfile — Work with ZIP archives" https://docs.python.org/3/library/zipfile.html
#      - "shutil — High-level file operations" https://docs.python.org/3/library/shutil.html
#
# 2. Use of `if __name__ == "__main__":`:
#    Reference: Python Official Documentation, "Executing modules as scripts"
#    https://docs.python.org/3/tutorial/modules.html#executing-modules-as-scripts
#
# 3. Iterating over files and checking extensions using string methods are standard as documented here:
#    Reference: Python Standard Library Documentation, str.endswith
#    https://docs.python.org/3/library/stdtypes.html#str.endswith
#
# 4. Checking file existence and moving files using os.path.exists(), shutil.move():
#    Reference: Python Official Documentation
#      - https://docs.python.org/3/library/os.path.html#os.path.exists
#      - https://docs.python.org/3/library/shutil.html#shutil.move
#
# 5. The overall unzip/extract pattern and handling nested ZIPs has been adapted from common approaches
#    found in Stack Overflow discussions (cf. example: https://stackoverflow.com/questions/3451111/unzipping-files-in-python)


#[1] Cheng, J., Huang, W., Cao, S., Yang, R., Zhang, W., Wang, J., & Feng, H. (2017).
#    Brain Tumor Dataset. Figshare. https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
#    (Dataset used in this pipeline.)
# -------------------------------------------------------


import os
import zipfile
import shutil

DOWNLOADED_ZIP_PATH = r"C:\Users\uysal\Downloads\1512427.zip"
PROJECT_FOLDER = r"C:\Users\uysal\Desktop\bt_final"


def unpack_dataset(zip_path: str, project_dir: str):
    """
    Unzips the main dataset zip file and extracts all zips files into the 'data_raw' folder.
    The files that are extracted: cvind.mat, README 2023.txt, and 4 Train Tumour dataset files.
    Args:
        zip_path (str): The path to the main downloaded .zip archive.
        project_dir (str): The path to the root project folder.
    """
    # Target directory for the raw data. This is the folder where all nested zips will be extracted.
    data_raw_folder = os.path.join(project_dir, "data_raw")
    
    # Create the project and data_raw folders if they don't already exist.
    os.makedirs(data_raw_folder, exist_ok=True)
    
    # Extract the main zip file
    print(f"Extracting main zip file: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(project_dir)

    # Find all the zip files 
    print("Extracting zip files into 'data_raw'...")
    for item in os.listdir(project_dir):
        if item.endswith(".zip"):
            nested_zip_path = os.path.join(project_dir, item)
            print(f"   - Unzipping {item}...")
            # Extract each nested zip into the data_raw folder - 4 Train Tumour dataset files.
            with zipfile.ZipFile(nested_zip_path, 'r') as nested_zf:
                nested_zf.extractall(data_raw_folder)
            # Remove the nested zip file after its contents are extracted
            os.remove(nested_zip_path)
            
    print("Moving 'cvind.mat' to the out o the folder")
    # It is moved from the data folder to the project root for easy access.
    cvind_source = os.path.join(data_raw_folder, "cvind.mat")
    cvind_dest = os.path.join(project_dir, "cvind.mat")
    
    # Check if the file exists before trying to move it.
    if os.path.exists(cvind_source):
        shutil.move(cvind_source, cvind_dest)
        
    print("Dataset is ready in'data_raw' folder.")

# Run the script when executed directly
if __name__ == "__main__":
    unpack_dataset(DOWNLOADED_ZIP_PATH, PROJECT_FOLDER)





# References:
# [1] Python Software Foundation. "Python 3 Standard Library documentation." https://docs.python.org/3/library/
    """
    Imported modules (os, zipfile, shutil): How to import and use these modules.
    os.makedirs() and os.makedirs(..., exist_ok=True): Creating directories if they do not already exist.
    os.path.join(): Joining path elements for cross-platform path building.
    os.listdir(): Listing files in a directory.
    os.remove(): Removing files by path.
    shutil.move(): Moving (renaming) files between directories.
    os.path.exists(): Checking if a file exists.
    zipfile.ZipFile: Opening, reading from, and extracting content from ZIP archives.  """
# [2] Python Software Foundation. "Modules: Executing modules as scripts." https://docs.python.org/3/tutorial/modules.html#executing-modules-as-scripts
# [3] Python Software Foundation, "str.endswith". https://docs.python.org/3/library/stdtypes.html#str.endswith
    """ 
        .endswith('.zip'): Check for file extensions in filenames when iterating.
        Source: Python Standard Library, str.endswith
    """
# [4] Python Software Foundation. "os.path.exists, shutil.move". https://docs.python.org/3/library/
# [5] Stack Overflow, "Unzipping files in python", https://stackoverflow.com/questions/3451111/unzipping-files-in-python