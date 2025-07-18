# data_loader.py
# Defines the custom PyTorch Dataset for loading and transforming the brain tumor MRI scans.


# --- Source and Pattern Citations ---
#
# [1] Standard PyTorch Custom Dataset conventions: subclassing, __init__, __len__, __getitem__.
#     https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#     https://pytorch.org/docs/stable/data.html
#
# [2] Rasa, S., et al. (2024). Deep learning based brain tumor classification using MRI images.
#     Biomedical Signal Processing and Control.
#     (Augmentation pipeline and parameter choices for brain tumor MRI.)
#
# [3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
#     CVPR 2016. https://arxiv.org/abs/1512.03385
#     (ResNet model input requirements: 224x224 and 3 channels.)
#
# [4] torchvision.transforms and normalization usage:
#     https://pytorch.org/vision/stable/transforms.html
#     https://pytorch.org/hub/pytorch_vision_resnet/
#
# [5] h5py documentation for reading MATLAB .mat (v7.3+):
#     https://docs.h5py.org/en/stable/
#     https://docs.h5py.org/en/stable/high/dataset.html#reading-writing-data
#
# [6] NumPy docs for normalization/conversion:
#     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html
#     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html
#
# [7] PyTorch docs: CrossEntropyLoss expects zero-indexed labels.
#     https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
#
# [8] Standard Python library (os, typing).
#     https://docs.python.org/3/library/
#
# [9] torchvision ImageNet normalization stats:
#     https://pytorch.org/vision/stable/models.html
#
# [10] Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning.
#      Journal of Big Data, 6, Article 60. https://doi.org/10.1186/s40537-019-0197-0
#      (Standard reference for normalization and augmentation improving convergence and generalization.)
# ------------------------------------------------------------


import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple

class BrainTumourDataset(Dataset): 
    """
    A custom PyTorch Dataset to handle the brain tumor MRI data.
    This class loads the .mat files, extracts the images and its label, and applys transformations.
    """
    def __init__(self, data_folder: str, filenames: List[str], labels: List[int], is_train: bool = True): 
        """
        Initializes the dataset.
        Args:
            data_folder (str): Path to the directory containing the .mat files.
            filenames (list): A list of the specific filenames to be loaded by this dataset instance.
            labels (list): A list of corresponding integer labels for the filenames.
            is_train (bool): A flag to indicate if this is the training set. If True, data augmentation transformations will be applied.
        """
        self.data_folder = data_folder 
        self.filenames = filenames 
        self.labels = labels 
        self.is_train = is_train 

        
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3), # Required: ResNet expects 3-channel RGB input (He et al., 2016)
                transforms.Resize((224, 224)), # Param from Rasa et al., 2024 
                transforms.RandomHorizontalFlip(p=0.5), # Param from Rasa et al., 2024 
                transforms.RandomVerticalFlip(p=0.5), # Param from Rasa et al., 2024 
                transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=20), # Param from Rasa et al., 2024 
                transforms.ColorJitter(brightness=0.05, contrast=0.1), # Param from Rasa et al., 2024 
                transforms.ToTensor(), # Converts PIL image to a PyTorch Tensor.
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization (PyTorch docs)
            ])
        
        # For validation and test sets, perform the necessary transformations without any random data augmentation.
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),  # Required: ResNet expects 3-channel RGB input (He et al., 2016)
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization (PyTorch docs)
            ])

    def __len__(self) -> int: #Returns the total number of samples in the dataset. 
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]: 
        """
        Fetches a single data sample (image and label) at the given index.
        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            A tuple containing the transformed image tensor and its corresponding 0-indexed integer label.
        """
        # Construct the full path to the .mat file.
        filepath = os.path.join(self.data_folder, self.filenames[index]) #

        # Load the .mat file using h5py, which is recommended for modern .mat files (v7.3+).
        with h5py.File(filepath, "r") as f: #
            # The image data is stored within the 'cjdata' struct.
            # [()] extracts the data from the HDF5 dataset into a NumPy array.
            image = f["cjdata"]["image"][()] #

        # --- Pre-processing ---
        image = image.astype(np.float32) #  Convert image data type to float32 for transformations.
        image /= image.max() #   Rescaling for convergence; standard DL practice (Shorten & Khoshgoftaar, 2019)
        image_tensor = self.transform(image)
        
        # The dataset labels are 1-indexed (1, 2, 3). PyTorch's CrossEntropyLoss
        # expects 0-indexed labels (0, 1, 2), so we subtract 1.
        # This logic is identical to the original implementation.

        label = torch.tensor(self.labels[index] - 1, dtype=torch.long)
        
        return image_tensor, label
    









    # data_loader.py
# Defines the custom PyTorch Dataset for loading and transforming the brain tumor MRI scans.
# ------------------------------------------------------------
# --- Source and Pattern Citations ---
# [1] Standard PyTorch Custom Dataset conventions: ... (see above)
# [2] Rasa, S., et al. (2024) ...
# [3] He, K., Zhang, X., Ren, S., & Sun, J. (2016) ...
# [4] torchvision.transforms usage: ...
# [5] h5py documentation: ...
# [6] NumPy documentation: ...
# [7] PyTorch CrossEntropyLoss: ...
# [8] Standard Python stdlib: ...
# [9] torchvision ImageNet normalization: ...
# [10] Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning.
#      Journal of Big Data, 6, Article 60. https://doi.org/10.1186/s40537-019-0197-0
# ------------------------------------------------------------