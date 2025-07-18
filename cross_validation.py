# cross_validation.py
# Handles the creation of patient-level data splits for 5-fold cross-validation.

# --- Source and Pattern Citations ---
#
# 1. Patient-Wise Cross-Validation Strategy:
#    - Varoquaux, G., & Cheplygina, V. (2022). "Machine learning for neuroimaging:
#      Potentials and pitfalls." NeuroImage. (Justifies patient-wise splitting to
#      prevent data leakage).
#    - Bishop, C. M. (2006). "Pattern Recognition and Machine Learning." Springer.
#      (Justifies the general use of 5-fold cross-validation for robust performance
#      estimation).
#
# 2. Dataset and Implementation Logic:
#    - Cheng, J., et al. (2017). "Brain Tumor Dataset." Figshare.
#      (The origin of the dataset and the cvind.mat file).
#    - The specific logic for reading cvind.mat, sorting filenames, and using
#      boolean arrays is derived directly from the original project file
#     .
#
# 3. Standard Library and Community Patterns:
#    - Reading .mat files (v7.3+): h5py Documentation
#      URL: https://docs.h5py.org/en/stable/high/dataset.html#reading-writing-data
#    - Listing and filtering files in a directory: A common pattern, for which
#      Stack Overflow provides representative examples.
#      URL: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
#    - Sorting strings with numbers inside them: A common pattern, with examples
#      found on Stack Overflow.
#      URL: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
# ------------------------------------------------------------------------------------

import os
import h5py
import numpy as np
from typing import List, Tuple


def generate_patient_splits(data_dir: str, cvind_path: str) -> List[Tuple]:
    """
    Generates 5-fold cross-validation splits based on patient IDs.

    This function uses the provided `cvind.mat` file to ensure that all images
    from a single patient belong to the same fold. As justified by Varoquaux &
    Cheplygina (2022), this patient-wise splitting prevents data leakage and
    provides a more realistic assessment of model generalization.

    The 5-fold strategy provides a robust performance estimate, as justified
    by Bishop (2006).

    Args:
        data_dir (str): The path to the 'data_raw' directory with .mat files.
        cvind_path (str): The path to the 'cvind.mat' file containing patient fold assignments.

    Returns:
        A list of 5 tuples, where each tuple contains the train, validation,
        and test files and labels for one fold.
    """

    # 1. Load the cross-validation indices from the 'cvind.mat' file.
    with h5py.File(cvind_path, "r") as f:
        fold_assignments = f["cvind"][()].flatten()

    # 2. Get a numerically sorted list of all .mat image filenames.
    def get_filenumber(filename: str) -> int:
        """Helper to extract the integer from a filename like '123.mat' for sorting."""
        return int(os.path.splitext(filename)[0])

    # Create a list of all files in the directory that end with ".mat".
    mat_files_only = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".mat"):
            mat_files_only.append(filename)

    # Sort that list numerically to align with the fold assignments.
    all_mat_files = sorted(mat_files_only, key=get_filenumber)  
    all_mat_files = np.array(all_mat_files)

    # 3. Extract the ground truth label for every image file.
    labels = []
    for filename in all_mat_files:
        filepath = os.path.join(data_dir, filename)
        with h5py.File(filepath, "r") as f:
            labels.append(int(f["cjdata"]["label"][0][0]))
    labels = np.array(labels)

    # 4. Generate the specific train, validation, and test sets for each of the 5 folds.
    all_splits_data = []
    for fold_idx in range(1, 6):

        # Define the role of each fold number based on the project's cyclic strategy.
        test_fold_num = fold_idx
        validation_fold_num = (fold_idx % 5) + 1

        # Determine which fold numbers belong to the training set for this iteration.
        train_fold_nums = []
        for i in range(1, 6):
            if i != test_fold_num and i != validation_fold_num:
                train_fold_nums.append(i)

        # Create boolean arrays to identify which samples belong to each set.
        test_indices = (fold_assignments == test_fold_num)
        validation_indices = (fold_assignments == validation_fold_num)
        training_indices = np.isin(fold_assignments, train_fold_nums)

        # Use the boolean arrays to select the files and labels for each set.
        current_split = (
            all_mat_files[training_indices].tolist(), labels[training_indices].tolist(),
            all_mat_files[validation_indices].tolist(), labels[validation_indices].tolist(),
            all_mat_files[test_indices].tolist(), labels[test_indices].tolist()
        )
        all_splits_data.append(current_split)

    return all_splits_data