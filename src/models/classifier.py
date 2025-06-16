"""
WiFi Intrusion Detection System (IDS) Model Architectures

This module contains the neural network architectures for WiFi intrusion detection,
including feedforward, LSTM, CNN, and TabNet models built with PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Configure logging
logger = logging.getLogger(__name__)

# TabNet imports
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    logger.warning("pytorch_tabnet not available. Install with: pip install pytorch-tabnet")


class FeedForwardClassifier(nn.Module):
    """
    Feedforward Neural Network for WiFi intrusion detection.
    
    A multi-layer perceptron with configurable hidden layers, dropout,
    and batch normalization for robust classification of network traffic patterns.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super(FeedForwardClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Extract hyperparameters from config
        hidden_sizes = config.get('hidden_sizes', [256, 128, 64])
        dropout_rate = config.get('dropout_rate', 0.3)
        use_batch_norm = config.get('use_batch_norm', True)
        activation = config.get('activation', 'relu')
        
        # Build the network layers
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation function
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sequential WiFi traffic analysis.
    
    Uses LSTM layers to capture temporal dependencies in network traffic,
    suitable for analyzing packet sequences and time-series patterns.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super(LSTMClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Extract hyperparameters
        hidden_size = config.get('hidden_size', 128)
        num_layers = config.get('num_layers', 2)
        dropout_rate = config.get('dropout_rate', 0.3)
        bidirectional = config.get('bidirectional', True)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate the size after LSTM
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM and classifier."""
        batch_size = x.size(0)
        
        # If input is 2D, add sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output from LSTM
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            last_output = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_output = hidden[-1]
        
        # Classification
        output = self.classifier(last_output)
        return output


class CNNClassifier(nn.Module):
    """
    1D CNN classifier for WiFi traffic pattern recognition.
    
    Uses 1D convolutional layers to extract local patterns and features
    from network traffic data, effective for detecting attack signatures.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super(CNNClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Extract hyperparameters
        num_filters = config.get('num_filters', [64, 128, 256])
        kernel_sizes = config.get('kernel_sizes', [3, 5, 7])
        dropout_rate = config.get('dropout_rate', 0.3)
        pool_size = config.get('pool_size', 2)
        
        # Ensure we have matching number of filters and kernel sizes
        if len(num_filters) != len(kernel_sizes):
            kernel_sizes = kernel_sizes * len(num_filters)
            kernel_sizes = kernel_sizes[:len(num_filters)]
        
        # Build convolutional layers
        conv_layers = []
        in_channels = 1
        
        for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            conv_layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout_rate)
            ])
            in_channels = filters
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the size after convolutions for the linear layer
        # This is approximate - we'll compute it dynamically
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_filters[-1], num_filters[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(num_filters[-1] // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN and classifier."""
        batch_size = x.size(0)
        
        # Reshape for 1D convolution: (batch_size, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(batch_size, -1)
        
        # Classification
        output = self.classifier(x)
        return output


class WiFiIDSTabNet(nn.Module):
    """
    TabNet classifier for WiFi intrusion detection.
    
    Uses TabNet architecture for interpretable deep learning on tabular data,
    providing feature selection and high performance on structured datasets.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super(TabNetClassifier, self).__init__()
        
        if not TABNET_AVAILABLE:
            raise ImportError("pytorch_tabnet is required for TabNet model. Install with: pip install pytorch-tabnet")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Extract TabNet hyperparameters from config
        tabnet_config = config.get('tabnet', {})
        
        # Create TabNet model
        self.tabnet_model = TabNetClassifier(
            n_d=tabnet_config.get('n_d', 8),
            n_a=tabnet_config.get('n_a', 8),
            n_steps=tabnet_config.get('n_steps', 3),
            gamma=tabnet_config.get('gamma', 1.3),
            n_independent=tabnet_config.get('n_independent', 2),
            n_shared=tabnet_config.get('n_shared', 2),
            lambda_sparse=tabnet_config.get('lambda_sparse', 1e-3),
            momentum=tabnet_config.get('momentum', 0.3),
            mask_type=tabnet_config.get('mask_type', 'entmax'),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=tabnet_config.get('optimizer_params', {'lr': 0.02}),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params=tabnet_config.get('scheduler_params', {'mode': 'min', 'patience': 10}),
            verbose=0
        )
        
        self.feature_importances_ = None
        self.is_fitted = False
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train the TabNet model."""
        logger.info("Training TabNet model...")
        
        tabnet_config = self.config.get('tabnet', {})
        
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        eval_name = ['val'] if eval_set else None
        
        self.tabnet_model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=['accuracy'],
            max_epochs=tabnet_config.get('max_epochs', 100),
            patience=tabnet_config.get('patience', 20),
            batch_size=tabnet_config.get('batch_size', 1024),
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        # Store feature importances
        self.feature_importances_ = self.tabnet_model.feature_importances_
        self.is_fitted = True
        logger.info("TabNet training completed")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.tabnet_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.tabnet_model.predict_proba(X)
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Get feature importances from the trained model."""
        return self.feature_importances_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (for compatibility with PyTorch Lightning)."""
        # Convert to numpy for TabNet
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # Get predictions
        proba = self.predict_proba(x_np)
        
        # Convert back to tensor
        return torch.tensor(proba, device=x.device if isinstance(x, torch.Tensor) else 'cpu')


def create_model(model_type: str, input_dim: int, output_dim: int, config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create model instances.
    
    Args:
        model_type: Type of model ('feedforward', 'lstm', 'cnn')
        input_dim: Input feature dimension
        output_dim: Number of output classes
        config: Model configuration dictionary
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_map = {
        'feedforward': FeedForwardClassifier,
        'lstm': LSTMClassifier,
        'cnn': CNNClassifier,
        'tabnet': WiFiIDSTabNet
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: {list(model_map.keys())}")
    
    model_class = model_map[model_type]
    return model_class(input_dim, output_dim, config)


class TreeBasedModel:
    """
    Tree-based model wrapper for Random Forest and XGBoost classifiers.
    
    Provides a unified interface for tree-based models with automatic hyperparameter
    tuning, feature importance analysis, and model persistence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tree-based model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.model_type = config['model']['architecture']
        self.model = None
        self.feature_importance_ = None
        self.is_fitted = False
        
        # Initialize the appropriate model
        if self.model_type == 'random_forest':
            self._init_random_forest()
        elif self.model_type == 'xgboost':
            self._init_xgboost()
        elif self.model_type == 'tabnet':
            self._init_tabnet()
        else:
            raise ValueError(f"Unsupported tree model type: {self.model_type}")
    
    def _init_random_forest(self):
        """Initialize Random Forest classifier."""
        model_config = self.config['model']
        
        self.model = RandomForestClassifier(
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', None),
            min_samples_split=model_config.get('min_samples_split', 2),
            min_samples_leaf=model_config.get('min_samples_leaf', 1),
            max_features='sqrt',
            bootstrap=True,
            random_state=self.config.get('reproducibility', {}).get('seed', 42),
            n_jobs=-1,
            verbose=0
        )
    
    def _init_xgboost(self):
        """Initialize XGBoost classifier."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is required for XGBoost model. Install with: pip install xgboost")
        
        model_config = self.config['model']
        
        self.model = xgb.XGBClassifier(
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', 6),
            learning_rate=model_config.get('learning_rate', 0.1),
            subsample=model_config.get('subsample', 0.8),
            colsample_bytree=model_config.get('colsample_bytree', 0.8),
            random_state=self.config.get('reproducibility', {}).get('seed', 42),
            n_jobs=-1,
            verbosity=0,  # Use verbosity instead of verbose for newer XGBoost
            enable_categorical=False
        )
    
    def _init_tabnet(self):
        """Initialize TabNet classifier."""
        if not TABNET_AVAILABLE:
            raise ImportError("pytorch_tabnet is required for TabNet model. Install with: pip install pytorch-tabnet")
        
        tabnet_config = self.config.get('tabnet', {})
        
        self.model = TabNetClassifier(
            n_d=tabnet_config.get('n_d', 8),
            n_a=tabnet_config.get('n_a', 8),
            n_steps=tabnet_config.get('n_steps', 3),
            gamma=tabnet_config.get('gamma', 1.3),
            n_independent=tabnet_config.get('n_independent', 2),
            n_shared=tabnet_config.get('n_shared', 2),
            lambda_sparse=tabnet_config.get('lambda_sparse', 1e-3),
            momentum=tabnet_config.get('momentum', 0.3),
            mask_type=tabnet_config.get('mask_type', 'entmax'),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=tabnet_config.get('optimizer_params', {'lr': 0.02}),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params=tabnet_config.get('scheduler_params', {'mode': 'min', 'patience': 10}),
            verbose=0
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the tree-based model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info(f"Training {self.model_type} model...")
        
        # For XGBoost, we can use early stopping with validation data
        if self.model_type == 'xgboost' and X_val is not None and y_val is not None:
            try:
                # Try new XGBoost API (v1.7+)
                from xgboost.callback import EarlyStopping
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[EarlyStopping(rounds=10, save_best=True)],
                    verbose=False
                )
            except (ImportError, TypeError):
                # Fallback to older API
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
        # For TabNet, use validation data if available
        elif self.model_type == 'tabnet':
            tabnet_config = self.config.get('tabnet', {})
            eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
            eval_name = ['val'] if eval_set else None
            
            self.model.fit(
                X_train=X_train,
                y_train=y_train,
                eval_set=eval_set,
                eval_name=eval_name,
                eval_metric=['accuracy'],
                max_epochs=tabnet_config.get('max_epochs', 100),
                patience=tabnet_config.get('patience', 20),
                batch_size=tabnet_config.get('batch_size', 1024),
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        
        self.is_fitted = True
        logger.info(f"{self.model_type} training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array or None if not available
        """
        return self.feature_importance_
    
    def save(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'config': self.config,
            'feature_importance': self.feature_importance_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TreeBasedModel':
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded TreeBasedModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.model_type = model_data['model_type']
        instance.feature_importance_ = model_data['feature_importance']
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return instance 