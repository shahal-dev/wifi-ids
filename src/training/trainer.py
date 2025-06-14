import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torchmetrics
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
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
        self.test_accuracy(preds, y)
        self.test_f1(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        
        # Store predictions for detailed analysis
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy)
        self.log('test_f1', self.test_f1)
        self.log('test_precision', self.test_precision)
        self.log('test_recall', self.test_recall)
        
        return loss
        
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        # Generate detailed classification report
        if self.test_predictions and self.test_targets:
            report = classification_report(
                self.test_targets,
                self.test_predictions,
                target_names=self.class_names,
                output_dict=True
            )
            
            # Log per-class metrics
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        self.log(f'test_{class_name}_{metric_name}', value)
                        
            # Generate confusion matrix
            cm = confusion_matrix(self.test_targets, self.test_predictions)
            self.log_confusion_matrix(cm)
            
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