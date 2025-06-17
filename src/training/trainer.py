import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torchmetrics
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class WiFiIDSLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for WiFi IDS classification.
    
    Handles training, validation, testing, and logging for WiFi intrusion detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        class_names: List[str],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adam",
        scheduler: str = "reduce_on_plateau",
        class_weights: Optional[torch.Tensor] = None,
        loss_function: str = "cross_entropy",
        l1_lambda: float = 0.0,
        l2_lambda: float = 1e-4
    ):
        """
        Initialize the Lightning module.
        
        Args:
            model: PyTorch model to train
            num_classes: Number of output classes
            class_names: List of class names
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            optimizer: Optimizer name ('adam', 'sgd', 'adamw')
            scheduler: Scheduler name ('reduce_on_plateau', 'cosine', 'step')
            class_weights: Weights for class balancing
            loss_function: Loss function name ('cross_entropy', 'focal_loss')
            l1_lambda: L1 regularization weight
            l2_lambda: L2 regularization weight
        """
        super().__init__()
        
        self.model = model
        self.num_classes = num_classes
        self.class_names = class_names
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        # Loss function
        if loss_function == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_function == "focal_loss":
            self.criterion = self._focal_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
            
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        
        # Store predictions for detailed analysis
        self.test_predictions = []
        self.test_targets = []
        self.test_probabilities = []
        
        # Training info for evaluation
        self.training_start_time = None
        self.training_end_time = None
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Compute focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
        
    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute L1 and L2 regularization losses."""
        l1_loss = 0.0
        l2_loss = 0.0
        
        for param in self.model.parameters():
            if param.requires_grad:
                l1_loss += torch.norm(param, 1)
                l2_loss += torch.norm(param, 2)
                
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        logits = self(x)
        
        # Compute loss
        loss = self.criterion(logits, y)
        
        # Add regularization
        reg_loss = self._compute_regularization_loss()
        total_loss = loss + reg_loss
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.train_f1(preds, y)
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        
        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_ce_loss', loss, prog_bar=True)
        self.log('train_reg_loss', reg_loss)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)
        self.log('train_f1', self.train_f1)
        self.log('train_precision', self.train_precision)
        self.log('train_recall', self.train_recall)
        
        return total_loss
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        logits = self(x)
        
        # Compute loss
        loss = self.criterion(logits, y)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.val_f1(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        self.log('val_precision', self.val_precision)
        self.log('val_recall', self.val_recall)
        
        return loss
        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        x, y = batch
        logits = self(x)
        
        # Compute loss
        loss = self.criterion(logits, y)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        self.test_accuracy(preds, y)
        self.test_f1(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        
        # Store predictions for detailed analysis
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        self.test_probabilities.extend(probs.cpu().numpy())
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy)
        self.log('test_f1', self.test_f1)
        self.log('test_precision', self.test_precision)
        self.log('test_recall', self.test_recall)
        
        return loss
        
    def on_train_start(self) -> None:
        """Called when training starts."""
        import time
        self.training_start_time = time.time()
    
    def on_train_end(self) -> None:
        """Called when training ends."""
        import time
        self.training_end_time = time.time()
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch - comprehensive evaluation."""
        if not (self.test_predictions and self.test_targets):
            return
            
        # Convert to numpy arrays
        y_pred = np.array(self.test_predictions)
        y_true = np.array(self.test_targets)
        y_probs = np.array(self.test_probabilities) if self.test_probabilities else None
        
        # Generate detailed classification report
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        # Log per-class metrics
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    self.log(f'test_{class_name}_{metric_name}', value)
        
        # Get model directory from trainer
        model_dir = None
        if hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback:
            model_dir = os.path.dirname(self.trainer.checkpoint_callback.dirpath)
        elif hasattr(self.trainer, 'default_root_dir'):
            model_dir = self.trainer.default_root_dir
        
        if model_dir:
            # Calculate training time
            training_time = None
            if self.training_start_time and self.training_end_time:
                training_time = self.training_end_time - self.training_start_time
            
            # Get validation accuracy
            val_accuracy = None
            if hasattr(self, 'trainer') and hasattr(self.trainer, 'logged_metrics'):
                for key, value in self.trainer.logged_metrics.items():
                    if 'val_acc' in key.lower():
                        val_accuracy = float(value)
                        break
            
            # Save comprehensive evaluation
            self.save_comprehensive_evaluation(
                y_true, y_pred, y_probs, model_dir, training_time, val_accuracy
            )
        
        # Generate confusion matrix (existing functionality)
        cm = confusion_matrix(y_true, y_pred)
        self.log_confusion_matrix(cm)
    
    def save_comprehensive_evaluation(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: Optional[np.ndarray],
        model_dir: str, training_time: Optional[float] = None, val_accuracy: Optional[float] = None
    ) -> None:
        """Save comprehensive evaluation results with visualizations."""
        logger.info("Saving comprehensive evaluation results...")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC if probabilities available
        roc_auc_macro = None
        roc_auc_weighted = None
        if y_probs is not None:
            try:
                roc_auc_macro = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr')
                roc_auc_weighted = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
            except:
                pass
        
        # Compile results
        results = {
            'model_name': self.__class__.__name__,
            'model_architecture': getattr(self.model, '__class__', {}).get('__name__', 'Unknown'),
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
                self.class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i])
                } for i in range(len(self.class_names))
            },
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'hyperparameters': dict(self.hparams)
        }
        
        # Save metrics to JSON
        metrics_path = os.path.join(model_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed classification report
        report_path = os.path.join(model_dir, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Classification Report for {results['model_architecture']}\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(y_true, y_pred, target_names=self.class_names))
            f.write(f"\n\nOverall Metrics:\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Validation Accuracy: {val_accuracy:.4f}\n" if val_accuracy else "")
            f.write(f"Training Time: {training_time:.2f} seconds\n" if training_time else "")
            f.write(f"Macro F1-Score: {f1_macro:.4f}\n")
            f.write(f"Weighted F1-Score: {f1_weighted:.4f}\n")
            if roc_auc_macro:
                f.write(f"ROC AUC (Macro): {roc_auc_macro:.4f}\n")
        
        # Create visualizations
        self.create_evaluation_plots(y_true, y_pred, y_probs, cm, model_dir, results)
        
        logger.info(f"Comprehensive evaluation saved to {model_dir}")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1-Score: {f1_macro:.4f}")
        logger.info(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    def create_evaluation_plots(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: Optional[np.ndarray],
        cm: np.ndarray, model_dir: str, results: dict
    ) -> None:
        """Create comprehensive evaluation plots."""
        plt.style.use('default')
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {results["model_architecture"]}')
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
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Normalized Confusion Matrix - {results["model_architecture"]}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Per-class Performance Bar Chart
        precision_per_class = [results['per_class_metrics'][cls]['precision'] for cls in self.class_names]
        recall_per_class = [results['per_class_metrics'][cls]['recall'] for cls in self.class_names]
        f1_per_class = [results['per_class_metrics'][cls]['f1_score'] for cls in self.class_names]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Precision
        axes[0].bar(range(len(self.class_names)), precision_per_class)
        axes[0].set_title('Precision per Class')
        axes[0].set_xlabel('Classes')
        axes[0].set_ylabel('Precision')
        axes[0].set_xticks(range(len(self.class_names)))
        axes[0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0].set_ylim([0, 1])
        
        # Recall
        axes[1].bar(range(len(self.class_names)), recall_per_class)
        axes[1].set_title('Recall per Class')
        axes[1].set_xlabel('Classes')
        axes[1].set_ylabel('Recall')
        axes[1].set_xticks(range(len(self.class_names)))
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1].set_ylim([0, 1])
        
        # F1-Score
        axes[2].bar(range(len(self.class_names)), f1_per_class)
        axes[2].set_title('F1-Score per Class')
        axes[2].set_xlabel('Classes')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_xticks(range(len(self.class_names)))
        axes[2].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[2].set_ylim([0, 1])
        
        plt.suptitle(f'Per-Class Performance Metrics - {results["model_architecture"]}')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "per_class_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. ROC Curves (if probabilities available)
        if y_probs is not None:
            try:
                plt.figure(figsize=(12, 8))
                
                # Convert labels to binary format for ROC curve
                y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
                
                # Plot ROC curve for each class
                for i in range(len(self.class_names)):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, linewidth=2, 
                            label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')
                
                plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves - {results["model_architecture"]}')
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
        metrics_values = [results['test_accuracy'], results['precision_macro'], 
                         results['recall_macro'], results['f1_macro']]
        
        axes[0, 0].bar(metrics_names, metrics_values)
        axes[0, 0].set_title('Overall Performance Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim([0, 1])
        for i, v in enumerate(metrics_values):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Class distribution in test set
        unique, counts = np.unique(y_true, return_counts=True)
        class_counts = [counts[unique == i][0] if i in unique else 0 for i in range(len(self.class_names))]
        
        axes[0, 1].bar(range(len(self.class_names)), class_counts)
        axes[0, 1].set_title('Test Set Class Distribution')
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(range(len(self.class_names)))
        axes[0, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # Training info
        info_text = f"Model: {results['model_architecture']}\n"
        if results['training_time']:
            info_text += f"Training Time: {results['training_time']:.2f}s\n"
        info_text += f"Test Accuracy: {results['test_accuracy']:.4f}\n"
        if results['validation_accuracy']:
            info_text += f"Validation Accuracy: {results['validation_accuracy']:.4f}\n"
        info_text += f"Total Test Samples: {len(y_true):,}\n"
        info_text += f"Number of Classes: {len(self.class_names)}"
        
        axes[1, 0].text(0.1, 0.5, info_text, transform=axes[1, 0].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Model Information')
        
        # Top 5 and Bottom 5 performing classes
        f1_with_names = list(zip(self.class_names, f1_per_class))
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
        
        plt.suptitle(f'Model Performance Summary - {results["model_architecture"]}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "model_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()
            
    def log_confusion_matrix(self, cm: np.ndarray) -> None:
        """Log confusion matrix."""
        # Convert to DataFrame for better visualization
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        
        # Log as text
        logger.info(f"Confusion Matrix:\n{cm_df}")
        
        # If using wandb, log as plot
        if isinstance(self.logger, WandbLogger):
            import wandb
            self.logger.experiment.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=self.test_targets,
                    preds=self.test_predictions,
                    class_names=self.class_names
                )
            })
            
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Optimizer
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
            
        # Scheduler
        if self.scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }
        elif self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        elif self.scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        else:
            return optimizer


def create_trainer(
    config: Dict[str, Any],
    checkpoint_dir: str = "models/checkpoints",
    log_dir: str = "logs"
) -> pl.Trainer:
    """
    Create a PyTorch Lightning trainer with callbacks and loggers.
    
    Args:
        config: Training configuration
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs
        
    Returns:
        Configured PyTorch Lightning trainer
    """
    # Callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_f1:.4f}',
        monitor=config.get('monitor_metric', 'val_f1'),
        mode=config.get('mode', 'max'),
        save_top_k=config.get('save_top_k', 3),
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=config.get('monitor_metric', 'val_f1'),
        patience=config.get('patience', 10),
        mode=config.get('mode', 'max'),
        min_delta=config.get('min_delta', 0.001),
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Loggers
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="wifi_ids",
        version=None
    )
    loggers.append(tb_logger)
    
    # WandB logger (if configured and enabled)
    if config.get('use_wandb', False) and config.get('wandb_project'):
        wandb_logger = WandbLogger(
            project=config['wandb_project'],
            name=config.get('experiment_name', 'wifi_ids_experiment')
        )
        loggers.append(wandb_logger)
    
    # Create trainer with explicit GPU settings
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 100),
        accelerator=config.get('accelerator', 'gpu'),  # Explicitly set accelerator
        devices=config.get('devices', 1),  # Explicitly set number of devices
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=50,
        deterministic=config.get('deterministic', True),
        precision=config.get('precision', 32),
        gradient_clip_val=config.get('gradient_clip_val', 0.0),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer 