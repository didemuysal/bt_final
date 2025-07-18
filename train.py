# train.py

# Final, corrected version for dissertation experiments.
# Fixes the tqdm AttributeError and includes all features.

import argparse
import copy
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import sys # Added for system info in experiment details
import json # Added for saving experiment details as JSON

from sklearn.metrics import (auc, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)
from sklearn.preprocessing import label_binarize
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from data import BrainTumourDataset # Ensure data.py is updated as per recommendations
from model import create_brain_tumour_model # Ensure model.py is updated as per recommendations
from splits import get_patient_level_splits

# --- Helper Function ---
def run_one_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        scores = torch.softmax(outputs, dim=1)
        preds = torch.argmax(scores, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores.cpu().detach().numpy())
        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(loader.iterable.dataset)
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_scores)

def get_optimizer(model_params, optimizer_name, lr):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=0.01)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr, alpha=0.99, eps=1e-8)
    elif optimizer_name == 'adadelta':
        # Adadelta's LR is typically fixed at 1.0 and not tuned.
        # The 'lr' argument passed here will be ignored if it's not 1.0.
        return optim.Adadelta(model_params, rho=0.9, eps=1e-6)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

# --- Main Training Loop ---
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("="*50)
    print(f"Starting Experiment: {time.ctime()}")
    print(f"Running with arguments: {args}")
    print(f"Using device: {DEVICE}")
    print("="*50)

    lr_str = f"{args.lr}" if args.optimizer != 'adadelta' else "default"
    experiment_name = f"{args.model}_{args.strategy}_{args.optimizer}_lr-{lr_str}"

    # Define the base folder for this specific experiment
    base_experiment_folder = os.path.join("experiments", experiment_name) # All experiments go into a 'experiments' root folder
    
    # Define subdirectories within the base folder
    model_dir = os.path.join(base_experiment_folder, "models")
    report_dir = os.path.join(base_experiment_folder, "outputs")
    
    # Create all necessary directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(base_experiment_folder, exist_ok=True) # Ensure base folder exists for details.json

    print(f"All experiment outputs will be saved to: {base_experiment_folder}")
    print(f"Model checkpoints will be saved to: {model_dir}")
    print(f"Reports (CSV, plots) will be saved to: {report_dir}")

    results_log, n_classes = [], 3
    all_splits = get_patient_level_splits(args.data_folder, args.cvind_path)
    summed_cm = np.zeros((n_classes, n_classes), dtype=int)
    mean_fpr, tprs, aucs = np.linspace(0, 1, 100), [], []

    # Instantiate a dummy dataset to get transform string
    # This is safe as it doesn't load actual data, just sets up transforms
    dummy_train_dataset = BrainTumourDataset(args.data_folder, [], [], is_train=True)
    data_augmentation_string = str(dummy_train_dataset.transform)

    for i, (train_files, train_labels, val_files, val_labels, test_files, test_labels) in enumerate(all_splits):
        fold_num = i + 1
        print(f"\n--- FOLD {fold_num}/5 ---")
        
        train_dataset = BrainTumourDataset(args.data_folder, train_files, train_labels, is_train=True)
        val_dataset = BrainTumourDataset(args.data_folder, val_files, val_labels, is_train=False)
        test_dataset = BrainTumourDataset(args.data_folder, test_files, test_labels, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        model = create_brain_tumour_model(model_name=args.model).to(DEVICE)
        model_head_string = str(model.fc) # Get string representation of the model's head

        criterion = nn.CrossEntropyLoss()

        if args.strategy == 'finetune':
            print(f"\nStage 1: Training head for {args.head_epochs} epochs...")
            for param in model.parameters(): param.requires_grad = False
            for param in model.fc.parameters(): param.requires_grad = True
            head_optimizer = get_optimizer(model.fc.parameters(), args.optimizer, args.lr)

            for epoch in range(args.head_epochs):
                pbar = tqdm(train_loader, desc=f"Fold {fold_num} Head Epoch {epoch+1}/{args.head_epochs}")
                run_one_epoch(model, pbar, criterion, head_optimizer, device=DEVICE)
            
            fine_tune_lr = args.lr / 10.0 # LR decay for Stage 2
            print(f"\nStage 2: Fine-tuning the full network with LR={fine_tune_lr}...")
        else: # baseline strategy
            fine_tune_lr = args.lr # Use the same LR for baseline
            print(f"\nStage 2: Training full network with LR={fine_tune_lr}...")
        
        for param in model.parameters(): param.requires_grad = True # Unfreeze all for Stage 2 or baseline
        optimizer = get_optimizer(model.parameters(), args.optimizer, fine_tune_lr) # Re-initialize optimizer

        best_val_loss, epochs_without_improvement = float('inf'), 0
        best_model_state = None

        for epoch in range(args.max_epochs):
            train_pbar = tqdm(train_loader, desc=f"Fold {fold_num} FT Epoch {epoch+1}/{args.max_epochs} (Train)")
            run_one_epoch(model, train_pbar, criterion, optimizer, device=DEVICE)
            
            val_pbar = tqdm(val_loader, desc=f"Fold {fold_num} FT Epoch {epoch+1}/{args.max_epochs} (Val)", leave=False)
            val_loss, _, _, _ = run_one_epoch(model, val_pbar, criterion, device=DEVICE)
            
            print(f"  -> Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss, best_model_state = val_loss, copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"  -> No improvement in validation loss for {epochs_without_improvement} epoch(s).")
                if epochs_without_improvement >= args.patience:
                    print(f"  -> EARLY STOPPING triggered after {args.patience} epochs without improvement.")
                    break
        
        # Model save path now uses 'model_dir'
        model_save_name = f"Fold_{fold_num}_best_model.pth" # Simplified name within the folder
        save_path = os.path.join(model_dir, model_save_name)
        torch.save(best_model_state, save_path)
        print(f"  -> Best model for fold {fold_num} saved to {save_path}")

        print("\nTesting the best model...")
        model.load_state_dict(best_model_state)
        test_pbar = tqdm(test_loader, desc="Testing")
        
        test_loss, test_labels, test_preds, test_scores = run_one_epoch(model, test_pbar, criterion, device=DEVICE)
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average=None, labels=list(range(n_classes)), zero_division=0)
        acc = (test_preds == test_labels).mean()
        summed_cm += confusion_matrix(test_labels, test_preds, labels=list(range(n_classes)))
        test_labels_binarized = label_binarize(test_labels, classes=list(range(n_classes)))

        fold_results = {'fold': fold_num, 'test_loss': test_loss, 'test_accuracy': acc}
        for class_idx, name in enumerate(['meningioma', 'glioma', 'pituitary']):
            fpr, tpr, _ = roc_curve(test_labels_binarized[:, class_idx], test_scores[:, class_idx])
            roc_auc = auc(fpr, tpr)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(roc_auc)
            for metric, value in zip(['precision', 'recall', 'f1', 'auc'], [precision[class_idx], recall[class_idx], f1[class_idx], roc_auc]):
                fold_results[f"{name}_{metric}"] = value

        # Calculate Macro Averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='macro', labels=list(range(n_classes)), zero_division=0
        )
        fold_results['macro_precision'] = macro_precision
        fold_results['macro_recall'] = macro_recall
        fold_results['macro_f1'] = macro_f1

        # Calculate Weighted Averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='weighted', labels=list(range(n_classes)), zero_division=0
        )
        fold_results['weighted_precision'] = weighted_precision
        fold_results['weighted_recall'] = weighted_recall
        fold_results['weighted_f1'] = weighted_f1

        results_log.append(fold_results)
        print(f"✅ Fold {fold_num} Test Accuracy: {acc:.3%}")

    # After all folds are complete, calculate mean and std
    df_folds = pd.DataFrame(results_log)

    mean_row = df_folds.drop(columns=['fold']).mean().to_dict()
    mean_row['fold'] = 'Mean'
    results_log.append(mean_row)

    std_row = df_folds.drop(columns=['fold']).std().to_dict()
    std_row['fold'] = 'Std Dev'
    results_log.append(std_row)

    df = pd.DataFrame(results_log)

    print("\n--- Cross-Validation Complete ---")
    print("Mean Performance Metrics Across 5 Folds:")
    print(df.loc[df['fold'] == 'Mean'].drop(columns=['fold']).transpose())
    print("\nStandard Deviation of Performance Metrics Across 5 Folds:")
    print(df.loc[df['fold'] == 'Std Dev'].drop(columns=['fold']).transpose())

    # Report paths now use 'report_dir'
    results_csv_path = os.path.join(report_dir, "results.csv") # Simplified name within 'outputs'
    cm_plot_path = os.path.join(report_dir, "confusion_matrix.png") # Simplified name
    roc_plot_path = os.path.join(report_dir, "roc_curve.png") # Simplified name

    cm_normalized = summed_cm.astype('float') / summed_cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Meningioma', 'Glioma', 'Pituitary'], yticklabels=['Meningioma', 'Glioma', 'Pituitary'])
    plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    plt.title(f'Normalized Confusion Matrix\n({experiment_name.replace("_", " ").title()})')
    plt.savefig(cm_plot_path)
    print(f"\nSaved confusion matrix to {cm_plot_path}")

    plt.figure(figsize=(8, 6))
    tprs_per_class = np.array(tprs).reshape(-1, n_classes, len(mean_fpr))
    aucs_per_class = np.array(aucs).reshape(-1, n_classes)
    for i, (color, name) in enumerate(zip(['aqua', 'darkorange', 'cornflowerblue'], ['Meningioma', 'Glioma', 'Pituitary'])):
        mean_tpr = np.mean(tprs_per_class[:, i, :], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs_per_class[:, i])
        std_auc = np.std(aucs_per_class[:, i])
        plt.plot(mean_fpr, mean_tpr, color=color, lw=2, label=f'ROC {name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'Mean ROC Curves\n({experiment_name.replace("_", " ").title()})'); plt.legend(loc="lower right")
    plt.savefig(roc_plot_path)
    print(f"Saved ROC curve to {roc_plot_path}")

    df.to_csv(results_csv_path, index=False)
    print(f"Saved detailed results to {results_csv_path}")

    # Save Experiment Details to JSON in the base_experiment_folder
    experiment_details = {
        "run_date": time.ctime(),
        "device": DEVICE,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "command_line_arguments": vars(args),
        "key_hyperparameters": {
            "batch_size": args.batch_size,
            "head_epochs": args.head_epochs,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "initial_learning_rate": args.lr,
            "lr_decay_factor_stage2": 10.0,
            "dataloader_num_workers": args.num_workers,
        },
        "model_architecture_head": model_head_string, # Dynamic string
        "data_augmentation_training": data_augmentation_string, # Dynamic string
        "final_summary_metrics_mean": df.loc[df['fold'] == 'Mean'].drop(columns=['fold']).iloc[0].to_dict(),
        "final_summary_metrics_std": df.loc[df['fold'] == 'Std Dev'].drop(columns=['fold']).iloc[0].to_dict(),
        "results_csv_path": os.path.relpath(results_csv_path, base_experiment_folder), # Store relative path
        "confusion_matrix_plot_path": os.path.relpath(cm_plot_path, base_experiment_folder), # Store relative path
        "roc_curve_plot_path": os.path.relpath(roc_plot_path, base_experiment_folder) # Store relative path
    }

    details_file_path = os.path.join(base_experiment_folder, "details.json") # Save directly in base folder
    with open(details_file_path, 'w') as f:
        json.dump(experiment_details, f, indent=4)
    print(f"Experiment details saved to: {details_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Brain Tumour Classification Model')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet50'])
    parser.add_argument('--strategy', type=str, default='finetune', choices=['finetune', 'baseline'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adadelta'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_folder', type=str, default='data_raw')
    parser.add_argument('--cvind_path', type=str, default='cvind.mat')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--head_epochs', type=int, default=3)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader') # Added num_workers argument
    args = parser.parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)