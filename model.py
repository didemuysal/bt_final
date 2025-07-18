# model.py
# Creates the deep learning model for classifying tumors.

from torch import nn
from torchvision import models

def create_brain_tumour_model(model_name: str = 'resnet50', pretrained: bool = True):
    """
    Creates a pre-trained model (ResNet18 or ResNet50) and adapts it for 3-class classification.

    Args:
        model_name (str): The name of the model architecture ('resnet18' or 'resnet50').
        pretrained (bool): If True, loads a model pre-trained on ImageNet.
    """
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

    # Freeze the pre-trained layers for initial head training
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer with a robust, standard classifier head.
    # This architecture is a common and defensible choice for transfer learning.
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 3))

    return model