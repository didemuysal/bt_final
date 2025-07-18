import argparse
import torch
import cv2
import numpy as np
import os
import h5py
import sys

# --- Imports from your project ---
from model import create_brain_tumour_model
from data import BrainTumourDataset 

# --- Imports from the grad-cam library ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

CLASS_INDEX_TO_NAME = {0: 'meningioma', 1: 'glioma', 2: 'pituitary'}

def get_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for the Brain Tumour Classification model.")
    
    # --- FIX: Replaced --model_path with arguments that match the new folder structure ---
    parser.add_argument('--experiment_name', type=str, required=True, 
                        help='The base name of the experiment, e.g., "resnet50_finetune_adam_lr-0.0001".')
    parser.add_argument('--fold', type=int, required=True,
                        help="The fold number of the model to use (e.g., 1).")
    parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet18', 'resnet50'],
                        help="The model architecture used during training.")
    
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to the input image (.mat file).')
    parser.add_argument('--target_class', type=int, default=None,
                        help='Optional: specify class index (0-2). If None, the top predicted class is used.')
    return parser.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- FIX: Build the correct model path from the arguments ---
    model_folder = f"fold_{args.fold}_{args.experiment_name}"
    model_path = os.path.join(model_folder, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please ensure the --experiment_name and --fold arguments are correct.")
        sys.exit(1)

    # 1. Load the Model
    print(f"Loading model: {model_path}")
    model = create_brain_tumour_model(model_name=args.model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    # 2. Select the Target Layer
    target_layers = [model.layer4]

    # 3. Prepare the Input Image
    dummy_dataset = BrainTumourDataset(data_folder="", filenames=[], labels=[], is_train=False)
    image_transform = dummy_dataset.transform

    with h5py.File(args.image_path, "r") as f:
        image_data = f["cjdata"]["image"][()]
        true_label_index = int(f["cjdata"]["label"][0][0]) - 1
    
    rgb_img_for_vis = cv2.cvtColor(np.uint8(255 * image_data / image_data.max()), cv2.COLOR_GRAY2RGB)
    rgb_img_for_vis = np.float32(cv2.resize(rgb_img_for_vis, (224, 224))) / 255
    input_tensor = image_transform(image_data).unsqueeze(0).to(device)

    # 4. Instantiate and Run Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(args.target_class)] if args.target_class is not None else None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # 5. Visualize and Save the Result
    visualization = show_cam_on_image(rgb_img_for_vis, grayscale_cam, use_rgb=True)
    
    output = torch.softmax(model(input_tensor), dim=1)
    predicted_class_name = CLASS_INDEX_TO_NAME.get(output.argmax().item())
    true_class_name = CLASS_INDEX_TO_NAME.get(true_label_index)
    
    # --- FIX: Create an output directory based on the model's parent folder ---
    output_dir = os.path.join("gradcam_outputs", model_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    img_basename = os.path.splitext(os.path.basename(args.image_path))[0]
    output_filename = f"gradcam_img-{img_basename}_true-{true_class_name}_pred-{predicted_class_name}.png"
    output_path = os.path.join(output_dir, output_filename)

    cv2.imwrite(output_path, visualization)
    print(f"✅ Grad-CAM visualization saved to: {output_path}")

if __name__ == '__main__':
    main()