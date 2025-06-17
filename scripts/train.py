#!/usr/bin/env python3
"""
WiFi IDS Training Script

This script orchestrates the training of WiFi intrusion detection models using PyTorch Lightning.
It loads configuration, prepares data, creates models, and manages the training process.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import WiFiIDSDataset, create_data_loaders, create_array_data_loaders
from src.models.classifier import create_model, TreeBasedModel
from src.training.trainer import WiFiIDSLightningModule, create_trainer

# XGBoost and RandomForest imports removed - handled by TreeBasedModel class
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import json



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_reproducibility(seed: int = 42) -> None:
    """Set up reproducibility for training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optimize for Tensor Cores on RTX GPUs (2x speedup)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # or 'high' for even more speed
        logger.info("🚀 Tensor Cores optimization enabled for faster training")


def evaluate_and_save_results(
    model, X_test: np.ndarray, y_test: np.ndarray, 
    class_names: list, model_dir: str, model_name: str,
    training_time: float = None, val_accuracy: float = None
) -> dict:
    """
    Comprehensive evaluation and visualization of model results.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        model_dir: Directory to save results
        model_name: Name of the model
        training_time: Training time in seconds
        val_accuracy: Validation accuracy
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info("Performing comprehensive evaluation...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities if available
    try:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        elif hasattr(model, '_Booster'):  # XGBoost case
            import xgboost as xgb
            dtest = xgb.DMatrix(X_test)
            y_pred_proba = model._Booster.predict(dtest)
        else:
            y_pred_proba = None
    except:
        y_pred_proba = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC if probabilities available
    roc_auc_macro = None
    roc_auc_weighted = None
    if y_pred_proba is not None:
        try:
            roc_auc_macro = roc_auc_score(y_test, y_pred_proba, average='macro', multi_class='ovr')
            roc_auc_weighted = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr')
        except:
            pass
    
    # Compile results
    results = {
        'model_name': model_name,
        'training_time': training_time,
        'test_accuracy': accuracy,
        'validation_accuracy': val_accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'per_class_metrics': {
            class_names[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            } for i in range(len(class_names))
        },
        'classification_report': class_report,
        'confusion_matrix': cm.tolist()
    }
    
    # Save metrics to JSON
    metrics_path = os.path.join(model_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save detailed classification report
    report_path = os.path.join(model_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Classification Report for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(classification_report(y_test, y_pred, target_names=class_names))
        f.write(f"\n\nOverall Metrics:\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n" if val_accuracy else "")
        f.write(f"Training Time: {training_time:.2f} seconds\n" if training_time else "")
        f.write(f"Macro F1-Score: {f1_macro:.4f}\n")
        f.write(f"Weighted F1-Score: {f1_weighted:.4f}\n")
        if roc_auc_macro:
            f.write(f"ROC AUC (Macro): {roc_auc_macro:.4f}\n")
    
    # Create visualizations
    plt.style.use('default')
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Normalized Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-class Performance Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision
    axes[0].bar(range(len(class_names)), precision_per_class)
    axes[0].set_title('Precision per Class')
    axes[0].set_xlabel('Classes')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    
    # Recall
    axes[1].bar(range(len(class_names)), recall_per_class)
    axes[1].set_title('Recall per Class')
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    
    # F1-Score
    axes[2].bar(range(len(class_names)), f1_per_class)
    axes[2].set_title('F1-Score per Class')
    axes[2].set_xlabel('Classes')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_ylim([0, 1])
    
    plt.suptitle(f'Per-Class Performance Metrics - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "per_class_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. ROC Curves (if probabilities available)
    if y_pred_proba is not None:
        try:
            plt.figure(figsize=(12, 8))
            
            # Convert labels to binary format for ROC curve
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
            
            # Plot ROC curve for each class
            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {model_name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "roc_curves.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate ROC curves: {e}")
    
    # 5. Model Summary Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall metrics comparison
    metrics_names = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
    metrics_values = [accuracy, precision_macro, recall_macro, f1_macro]
    
    axes[0, 0].bar(metrics_names, metrics_values)
    axes[0, 0].set_title('Overall Performance Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(metrics_values):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Class distribution in test set
    unique, counts = np.unique(y_test, return_counts=True)
    class_counts = [counts[unique == i][0] if i in unique else 0 for i in range(len(class_names))]
    
    axes[0, 1].bar(range(len(class_names)), class_counts)
    axes[0, 1].set_title('Test Set Class Distribution')
    axes[0, 1].set_xlabel('Classes')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks(range(len(class_names)))
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    
    # Training info (if available)
    if training_time and val_accuracy:
        info_text = f"Model: {model_name}\n"
        info_text += f"Training Time: {training_time:.2f}s\n"
        info_text += f"Test Accuracy: {accuracy:.4f}\n"
        info_text += f"Validation Accuracy: {val_accuracy:.4f}\n"
        info_text += f"Total Test Samples: {len(y_test):,}\n"
        info_text += f"Number of Classes: {len(class_names)}"
        
        axes[1, 0].text(0.1, 0.5, info_text, transform=axes[1, 0].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Model Information')
    
    # Top 5 and Bottom 5 performing classes
    f1_with_names = list(zip(class_names, f1_per_class))
    f1_sorted = sorted(f1_with_names, key=lambda x: x[1], reverse=True)
    
    top_5 = f1_sorted[:5]
    bottom_5 = f1_sorted[-5:]
    
    top_names, top_scores = zip(*top_5)
    bottom_names, bottom_scores = zip(*bottom_5)
    
    y_pos = np.arange(5)
    axes[1, 1].barh(y_pos, top_scores, alpha=0.7, color='green', label='Top 5')
    axes[1, 1].barh(y_pos - 0.4, bottom_scores, alpha=0.7, color='red', label='Bottom 5')
    axes[1, 1].set_yticks(y_pos - 0.2)
    axes[1, 1].set_yticklabels([f"{top_names[i]}\nvs\n{bottom_names[i]}" for i in range(5)])
    axes[1, 1].set_xlabel('F1-Score')
    axes[1, 1].set_title('Best vs Worst Performing Classes')
    axes[1, 1].legend()
    axes[1, 1].set_xlim([0, 1])
    
    plt.suptitle(f'Model Performance Summary - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "model_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation complete. Results saved to {model_dir}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1-Score: {f1_macro:.4f}")
    logger.info(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    return results


def load_and_prepare_data(config: Dict[str, Any]) -> tuple:
    """
    Load and prepare WiFi IDS data for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (features, labels, feature_names, class_names, scaler, label_encoder)
    """
    logger.info("Loading and preparing data...")
    
    # Load data from multiple CSV files
    data_path = config['data']['data_path']
    data_frames = []
    
    # Iterate through attack folders
    for attack_type in config['data']['attack_types']:
        folder_path = os.path.join(data_path, attack_type)
        if os.path.exists(folder_path):
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            for csv_file in csv_files:
                file_path = os.path.join(folder_path, csv_file)
                df = pd.read_csv(file_path)
                
                # Add attack type label if not present
                if 'Label' not in df.columns:
                    df['Label'] = attack_type
                    
                data_frames.append(df)
                logger.info(f"Loaded {len(df)} samples from {attack_type}/{csv_file}")
    
    if not data_frames:
        raise ValueError(f"No data found in {data_path}")
        
    # Combine all data
    data = pd.concat(data_frames, ignore_index=True)
    logger.info(f"Total samples loaded: {len(data)}")
    
    # 🚀 SPEED OPTIMIZATION: Sample data if enabled
    if config['data'].get('enable_sampling', True):
        sample_size = config['data'].get('sample_size', 1000000)
        if len(data) > sample_size:
            logger.info(f"Sampling {sample_size:,} samples from {len(data):,} total samples")
            data = data.sample(n=sample_size, random_state=config['data']['random_seed'])
            logger.info(f"Sampled dataset size: {len(data):,}")
    else:
        logger.info("Using full dataset (sampling disabled)")
    
    # Get feature columns (exclude Label)
    feature_columns = [col for col in data.columns if col != 'Label']
    
    # Ensure we have the expected number of features
    if len(feature_columns) != config['data']['num_features']:
        logger.warning(f"Expected {config['data']['num_features']} features, got {len(feature_columns)}")
    
    # Prepare features and labels
    X = data[feature_columns].values
    y = data['Label'].values
    
    # Handle missing values
    if np.isnan(X).any():
        logger.warning("Found missing values, filling with median")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    # The processed data already contains numeric labels from preprocessing
    # We need to map them back to class names and then use consistent encoding
    
    # Mapping from preprocessing script numeric labels to class names
    LABEL_TO_NAME = {
        0: "(Re)Assoc", 1: "Botnet", 2: "Deauth", 3: "Disas", 4: "Evil_Twin", 
        5: "Kr00K", 6: "Krack", 7: "Malware", 8: "Normal", 9: "RogueAP", 
        10: "SQL_Injection", 11: "SSDP", 12: "SSH", 13: "Website_spoofing"
    }
    
    # Convert numeric labels to class names first
    y_names = []
    for label in y:
        if isinstance(label, str):
            if label.isdigit():
                label_num = int(label)
            elif label.startswith(('1.', '2.', '3.')):
                # Handle folder names if they appear
                if label.startswith('1.'):
                    label_num = 2  # Deauth
                elif label.startswith('2.'):
                    label_num = 3  # Disas
                elif label.startswith('3.'):
                    label_num = 0  # (Re)Assoc
                elif label.startswith('4.'):
                    label_num = 9  # RogueAP
                elif label.startswith('5.'):
                    label_num = 6  # Krack
                elif label.startswith('6.'):
                    label_num = 5  # Kr00K
                elif label.startswith('7.'):
                    label_num = 12  # SSH
                elif label.startswith('8.'):
                    label_num = 1  # Botnet
                elif label.startswith('9.'):
                    label_num = 7  # Malware
                elif label.startswith('10.'):
                    label_num = 10  # SQL_Injection
                elif label.startswith('11.'):
                    label_num = 11  # SSDP
                elif label.startswith('12.'):
                    label_num = 4  # Evil_Twin
                elif label.startswith('13.'):
                    label_num = 13  # Website_spoofing
                else:
                    label_num = 8  # Normal
            else:
                # Already a class name
                continue
        else:
            label_num = int(label)
        
        if label_num in LABEL_TO_NAME:
            y_names.append(LABEL_TO_NAME[label_num])
        else:
            logger.warning(f"Unknown label: {label}, mapping to Normal")
            y_names.append("Normal")
    
    # Create label encoder with alphabetical ordering for consistency
    KNOWN_LABELS = [
        "(Re)Assoc", "Botnet", "Deauth", "Disas", "Evil_Twin", "Kr00K", 
        "Krack", "Malware", "Normal", "RogueAP", "SQL_Injection", 
        "SSDP", "SSH", "Website_spoofing"
    ]
    
    label_encoder = LabelEncoder()
    label_encoder.fit(KNOWN_LABELS)
    
    # Encode the class names
    y_encoded = label_encoder.transform(y_names)
    
    class_names = list(label_encoder.classes_)
    logger.info(f"Classes found: {class_names}")
    logger.info("Label mapping:")
    for i, label in enumerate(class_names):
        logger.info(f"  {label}: {i}")
    
    # Log the original numeric to final mapping
    logger.info("Original numeric label -> Final encoded mapping:")
    for orig_num, class_name in LABEL_TO_NAME.items():
        final_encoded = label_encoder.transform([class_name])[0]
        logger.info(f"  {orig_num} ({class_name}) -> {final_encoded}")
    
    # Create scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, feature_columns, class_names, scaler, label_encoder


def split_data(X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Label vector
        config: Configuration dictionary
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Splitting data...")
    
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    random_seed = config['data']['random_seed']
    
    # First split: train + val vs test
    test_size = test_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    # Second split: train vs val from remaining data
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_seed, stratify=y_temp
    )
    
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_class_weights(y_train: np.ndarray, config: Dict[str, Any]) -> torch.Tensor:
    """Compute class weights for handling imbalanced data."""
    if config['training'].get('use_class_weights', False):
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        return torch.FloatTensor(weights)
    return None


def create_datasets_and_loaders(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    feature_names: list, scaler: StandardScaler, label_encoder: LabelEncoder,
    config: Dict[str, Any]
) -> tuple:
    """Create PyTorch datasets and data loaders efficiently from preprocessed arrays."""
    logger.info("Creating datasets and data loaders from preprocessed arrays...")
    
    # Extract DataLoader parameters
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    pin_memory = config['hardware']['pin_memory']
    
    # Create data loaders directly from arrays (no file I/O, no double preprocessing)
    train_loader, val_loader, test_loader = create_array_data_loaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Get class names from label encoder
    class_names = list(label_encoder.classes_)
    
    logger.info(f"Created efficient data loaders:")
    logger.info(f"  Train: {len(train_loader.dataset)} samples")
    logger.info(f"  Validation: {len(val_loader.dataset)} samples") 
    logger.info(f"  Test: {len(test_loader.dataset)} samples")
    logger.info(f"  Batch size: {batch_size}, Workers: {num_workers}")
    
    return train_loader, val_loader, test_loader, class_names


def train_neural_network(
    config: Dict[str, Any],
    train_loader, val_loader, test_loader,
    class_names: list, class_weights: torch.Tensor
) -> None:
    """Train neural network models using existing infrastructure."""
    logger.info("Training neural network...")
    
    # Create model using existing function
    model = create_model(
        model_type=config['model']['architecture'],
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['num_classes'],
        config=config['model']
    )
    logger.info(f"Created {config['model']['architecture']} model")
    
    # Create unique model directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(
        config['paths']['base_dir'],
        config['model']['architecture'],
        timestamp
    )
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Created model directory: {model_dir}")
    
    # Create Lightning module using existing class
    lightning_module = WiFiIDSLightningModule(
        model=model,
        num_classes=config['model']['num_classes'],
        class_names=class_names,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        optimizer=config['training']['optimizer'],
        scheduler=config['training']['scheduler'],
        class_weights=class_weights,
        loss_function=config['training']['loss_function'],
        l1_lambda=config['training']['l1_lambda'],
        l2_lambda=config['training']['l2_lambda']
    )
    
    # Create trainer using existing function
    trainer = create_trainer(
        config=config['logging'],
        checkpoint_dir=model_dir,  # Use the unique model directory
        log_dir=os.path.join(config['paths']['log_dir'], config['model']['architecture'], timestamp)
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(lightning_module, train_loader, val_loader)
    
    # Test model
    logger.info("Running final evaluation...")
    test_results = trainer.test(lightning_module, test_loader)
    
    # Save final model and metadata
    final_model_path = os.path.join(model_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    
    # Save config file in the model directory for reference
    import yaml
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Final model saved to {model_dir}")
    logger.info("Comprehensive evaluation results will be saved automatically by the trainer")


def train_tree_model(
    config: Dict[str, Any],
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray
) -> None:
    """Train tree-based models."""
    logger.info(f"Training tree model: {config['model']['architecture']}")
    
    # Create unique model directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(
        config['paths']['base_dir'],
        config['model']['architecture'],
        timestamp
    )
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Created model directory: {model_dir}")
    
    # Create tree model
    model = TreeBasedModel(config)
    
    # Train model
    import time
    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    training_time = time.time() - start_time
    
    # Evaluate
    val_results = model.evaluate(X_val, y_val)
    test_results = model.evaluate(X_test, y_test)
    
    # Save model
    model_path = os.path.join(model_dir, "final_model.pkl")
    model.save(model_path)
    
    # Save config file in the model directory for reference
    import yaml
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance is not None:
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(feature_importance))],
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(model_dir, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importances - {config["model"]["architecture"]}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Get class names for evaluation
    class_names = [
        "(Re)Assoc", "Botnet", "Deauth", "Disas", "Evil_Twin", "Kr00K", 
        "Krack", "Malware", "Normal", "RogueAP", "SQL_Injection", 
        "SSDP", "SSH", "Website_spoofing"
    ]
    
    # Comprehensive evaluation and visualization
    evaluation_results = evaluate_and_save_results(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        model_dir=model_dir,
        model_name=config['model']['architecture'],
        training_time=training_time,
        val_accuracy=val_results['accuracy']
    )
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Model and comprehensive evaluation results saved to {model_dir}")


# Duplicate functions removed - TreeBasedModel class handles both Random Forest and XGBoost training





def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train WiFi IDS model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['feedforward', 'lstm', 'cnn', 'random_forest', 'xgboost'],
        help="Model architecture (overrides config)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Override model if specified
    if args.model:
        config.model.architecture = args.model
        logger.info(f"Model architecture overridden to: {args.model}")
    
    # Setup directories
    os.makedirs(config.paths.base_dir, exist_ok=True)
    os.makedirs(config.paths.log_dir, exist_ok=True)
    os.makedirs(config.paths.results_dir, exist_ok=True)
    
    # Setup reproducibility
    setup_reproducibility(config.reproducibility.seed)
    
    # Load and prepare data
    X, y, feature_names, class_names, scaler, label_encoder = load_and_prepare_data(config)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train, config)
    
    # Train based on model type
    if config.model.architecture in ['feedforward', 'lstm', 'cnn']:
        # Neural network training
        train_loader, val_loader, test_loader, _ = create_datasets_and_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test,
            feature_names, scaler, label_encoder, config
        )
        train_neural_network(config, train_loader, val_loader, test_loader, class_names, class_weights)
        
    elif config.model.architecture in ['random_forest', 'xgboost']:
        # Tree model training (handles everything including evaluation)
        train_tree_model(config, X_train, X_val, X_test, y_train, y_val, y_test)
        
    else:
        raise ValueError(f"Unknown model architecture: {config.model.architecture}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 