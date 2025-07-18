Technical Summary of the Brain Tumor Classification Project
This project focuses on the classification of brain tumors (Meningioma, Glioma, Pituitary) from T1-weighted contrast-enhanced MRI images using deep learning, specifically leveraging transfer learning with pre-trained Convolutional Neural Networks (CNNs). The core objective is to conduct a comparative study of different training strategies, optimizers, and learning rates, while ensuring robust and reproducible results.

1. Data Acquisition and Preprocessing (unzip.py, data.py)

Dataset: The project utilizes the publicly available Figshare Brain Tumor Dataset, comprising 3064 T1-weighted contrast-enhanced MRI slices from 233 patients, stored in .mat file format. Each image is associated with a tumor type label (1-indexed: 1, 2, 3).
Initial Preparation (unzip.py): A utility script handles the extraction of nested .zip archives and organizes the raw .mat files into a data_raw directory, along with the cvind.mat file containing cross-validation indices.
Data Loading (data.py): A custom BrainTumourDataset class (inheriting from torch.utils.data.Dataset) is implemented to load individual .mat files. It extracts the 2D image array and its corresponding 1-indexed label, converting the label to a 0-indexed PyTorch tensor. Images are pre-scaled to [0, 1] range and converted to PIL Image format.
Image Preprocessing: All images are resized to 224x224 pixels, converted to 3-channel grayscale (by duplicating the single channel) to match the input requirements of ImageNet pre-trained models, and then normalized using standard ImageNet mean and standard deviation.
Data Augmentation (data.py): For training data, a comprehensive set of on-the-fly augmentations is applied, directly inspired by Rasa et al. (2023/2024). This includes RandomHorizontalFlip, RandomVerticalFlip, RandomAffine transformations (covering ±7° rotation, ±5% translation, ±10% scaling/zoom, and ±20% shear), and ColorJitter for ±5% brightness and ±10% contrast variations.
2. Model Architecture (model.py)

Backbone: The project employs pre-trained ResNet-18 and ResNet-50 architectures (from torchvision.models, pre-trained on ImageNet-1K). These models serve as powerful feature extractors.
Classifier Head: The original 1000-class classification head of the ResNet models is replaced with a custom "Standard Head." This head consists of a nn.Sequential module comprising a nn.Linear layer with 512 units, followed by a nn.ReLU activation, a nn.Dropout layer (with p=0.5), and a final nn.Linear layer mapping to the 3 tumor classes. This design balances model capacity with the risk of overfitting, a common practice in transfer learning for medical imaging.
3. Cross-Validation Strategy (splits.py)

Patient-wise 5-Fold Cross-Validation: The splits.py module implements a robust patient-level 5-fold cross-validation strategy, utilizing the provided cvind.mat file. This ensures that images from the same patient are never present in both the training/validation and test sets of a given fold, preventing data leakage and providing a more realistic assessment of model generalization.
Cyclic Split: For each fold, one fold of patient data serves as the test set, the subsequent fold (in a cyclic manner) serves as the validation set, and the remaining three folds constitute the training set.
4. Training and Evaluation Protocol (train.py)

Training Strategies: Two primary training strategies are compared:
Fine-tune (Two-Stage Transfer Learning):
Stage 1 (Head Training): The pre-trained ResNet backbone layers are frozen (requires_grad=False), and only the newly added classifier head is trained for a few epochs (head_epochs).
Stage 2 (Full Fine-tuning): All layers of the entire network (backbone and head) are unfrozen (requires_grad=True), and the model is further trained with a significantly reduced learning rate (initial LR / 10.0) for additional epochs (max_epochs).
Baseline: The entire network (backbone and new head) is trained end-to-end from the beginning, without an initial head-only training stage.
Optimization: Experiments are conducted using five different optimizers: Adam, AdamW, SGD (with Nesterov momentum), RMSprop, and Adadelta. For Adam, AdamW, SGD, and RMSprop, a grid search over five learning rates (1e-2 to 1e-6) is performed. Adadelta is used with its default learning rate (1.0).
Loss Function: nn.CrossEntropyLoss is used for multi-class classification.
Early Stopping: Training for both stages (or baseline) incorporates early stopping with a defined patience, monitoring validation loss to prevent overfitting and optimize training time.
Evaluation Metrics: For each fold, the model's performance on the held-out test set is comprehensively evaluated using:
test_loss and test_accuracy.
Per-class precision, recall, F1-score, and AUC (Area Under the Receiver Operating Characteristic curve).
Macro-averaged and Weighted-averaged precision, recall, and F1-score to provide aggregate performance views.
Result Aggregation: After all 5 folds are completed for a given experiment configuration, the mean and standard deviation of all metrics are calculated across the folds, providing a robust estimate of the model's performance and its variability.
5. Experiment Orchestration and Output Management (train.py)

Experiment Naming: Each unique combination of model, strategy, optimizer, and learning rate is assigned a unique experiment_name.
Structured Output: All results for a single experiment are consolidated into a dedicated folder (experiments/EXPERIMENT_NAME/).
models/: Stores the best model checkpoint (.pth file) for each individual fold.
outputs/: Contains the aggregated results.csv (with per-fold data, mean, and standard deviation rows), a normalized confusion_matrix.png, and a roc_curve.png plot.
details.json: A comprehensive JSON file storing all experiment metadata, including command-line arguments, system information, key hyperparameters, and dynamic string representations of the model head and data augmentation pipeline, ensuring full reproducibility.
6. Explainability (Planned - gradcam.py)

A separate gradcam.py script is prepared to generate Grad-CAM visualizations for selected images and models. This post-analysis step aims to provide interpretability by highlighting regions of the input image that are most influential in the model's predictions.
