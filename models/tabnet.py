import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class WiFiIDSTabNet(nn.Module):
    """
    TabNet model for WiFi IDS classification.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = None
        self.feature_importances = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train the TabNet model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("Initializing TabNet model...")
        
        # Create TabNet model
        self.model = TabNetClassifier(
            n_d=self.config['tabnet']['n_d'],
            n_a=self.config['tabnet']['n_a'],
            n_steps=self.config['tabnet']['n_steps'],
            gamma=self.config['tabnet']['gamma'],
            n_independent=self.config['tabnet']['n_independent'],
            n_shared=self.config['tabnet']['n_shared'],
            lambda_sparse=self.config['tabnet']['lambda_sparse'],
            momentum=self.config['tabnet']['momentum'],
            mask_type=self.config['tabnet']['mask_type'],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=self.config['tabnet']['optimizer_params'],
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params=self.config['tabnet']['scheduler_params'],
            verbose=10
        )
        
        # Train the model
        logger.info("Training TabNet model...")
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['val'],
            eval_metric=['accuracy'],
            max_epochs=self.config['tabnet']['max_epochs'],
            patience=self.config['tabnet']['patience'],
            batch_size=self.config['tabnet']['batch_size'],
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        # Get feature importances
        self.feature_importances = self.model.feature_importances_
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the trained model.
        
        Returns:
            Feature importances array
        """
        if self.feature_importances is None:
            raise ValueError("Model has not been trained yet")
        return self.feature_importances 