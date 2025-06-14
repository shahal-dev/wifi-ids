#!/usr/bin/env python3
"""
WiFi IDS Evaluation Script

This script evaluates a trained WiFi IDS model on test data and generates detailed reports.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from omegaconf import OmegaConf

from training.trainer import WiFiIDSLightningModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str) -> WiFiIDSLightningModule:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    model = WiFiIDSLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def evaluate_model(model: WiFiIDSLightningModule, test_loader, class_names: list) -> dict:
    """Evaluate model and generate comprehensive metrics."""
    logger.info("Evaluating model...")
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC (multi-class)
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    roc_auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str = None):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: list, save_path: str = None):
    """Plot ROC curves for each class."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(15, 10))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    plt.show()


def generate_report(results: dict, class_names: list, output_dir: str):
    """Generate detailed evaluation report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save classification report
    report_df = pd.DataFrame(results['classification_report']).transpose()
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    logger.info(f"Classification report saved to {report_path}")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(results['confusion_matrix'], 
                        index=class_names, columns=class_names)
    cm_path = os.path.join(output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Plot confusion matrix
    cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_plot_path)
    
    # Plot ROC curves
    roc_plot_path = os.path.join(output_dir, 'roc_curves.png')
    plot_roc_curves(results['y_true'], results['y_prob'], class_names, roc_plot_path)
    
    # Summary metrics
    summary = {
        'Overall Accuracy': results['classification_report']['accuracy'],
        'Macro Average F1': results['classification_report']['macro avg']['f1-score'],
        'Weighted Average F1': results['classification_report']['weighted avg']['f1-score'],
        'ROC AUC (Macro)': results['roc_auc']
    }
    
    summary_path = os.path.join(output_dir, 'summary_metrics.txt')
    with open(summary_path, 'w') as f:
        f.write("WiFi IDS Model Evaluation Summary\n")
        f.write("=" * 40 + "\n\n")
        for metric, value in summary.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    logger.info(f"Summary metrics saved to {summary_path}")
    
    return summary


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate WiFi IDS model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Here you would load your test data
    # For now, we'll assume you have a test_loader available
    # In practice, you'd recreate this from your saved data splits
    
    logger.info("Note: You need to implement test data loading based on your saved splits")
    logger.info("The evaluation framework is ready for use once test data is available")
    
    # Example of how to use once test_loader is available:
    # results = evaluate_model(model, test_loader, config['data']['attack_types'])
    # summary = generate_report(results, config['data']['attack_types'], args.output_dir)
    # print("\nEvaluation Summary:")
    # for metric, value in summary.items():
    #     print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main() 