# model.py
# Creates the deep learning model for classifying tumors.


# --- Source and Pattern Citations ---
#
# 1. Pre-trained ResNet Architecture: The use of ResNet models is based on the original
#    paper that introduced the architecture.
#    Reference: He, K., et al. (2016). "Deep Residual Learning for Image Recognition."
#
# 2. ImageNet-1K Pre-training Dataset: The models are loaded with weights pre-trained on
#    the ImageNet-1K dataset.
#    Reference: Russakovsky, O., et al. (2015). "ImageNet Large Scale Visual
#    Recognition Challenge."
#
# 3. Transfer Learning Implementation Pattern: The strategy of loading a pre-trained
#    model, freezing the base layers (`param.requires_grad = False`), and replacing
#    the final fully-connected layer (`fc`) is a standard PyTorch pattern for fine-tuning.
#    Reference: PyTorch Official Tutorials, "Transfer Learning for Computer Vision"
#    URL: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#
# 4. PyTorch `torchvision.models` and `torch.nn` API: The code uses the official
#    PyTorch library to load models and define layers for the custom classifier head.
#    Reference: PyTorch Official Documentation
#    URL: https://pytorch.org/vision/stable/models.html

# ------------------------------------------------------------------------------------

from torch import nn
from torchvision import models

def create_brain_tumour_model(model_name: str = 'resnet50', pretrained: bool = True) -> nn.Module:
    """
    Creates a pre-trained model (ResNet18 or ResNet50) and adapts it for 3-class classification.

    This function implements the transfer learning strategy described in the project's
    technical summary. It freezes the convolutional
    base and replaces the final classifier with a custom head.

    Args:
        model_name (str): The name of the model architecture ('resnet18' or 'resnet50').
        pretrained (bool): If True, loads a model pre-trained on ImageNet.

    Returns:
        A PyTorch model ready for the initial stage of training.
    """
    model = None
    # Load the specified ResNet model using the torchvision library.
    if model_name.lower() == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        num_features = model.fc.in_features
    elif model_name.lower() == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
    else:
        raise ValueError(f"Model '{model_name}' not supported. Choose 'resnet18' or 'resnet50'.")

    # Freeze the pre-trained layers for the initial stage of training.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the original fully-connected layer (`fc`) with a new custom classifier head,
    # as specified in the project's README.md file.
    #
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 3))

    return model