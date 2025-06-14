"""
Utility functions for WiFi IDS project.
"""

import os
import json
import pickle
import logging
from typing import Any, Dict, List, Union
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


def save_model_artifacts(
    model_path: str,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    feature_names: List[str],
    class_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Save model artifacts for deployment.
    
    Args:
        model_path: Path where model checkpoint is saved
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        feature_names: List of feature names
        class_names: List of class names
        config: Model configuration
    """
    artifacts_dir = os.path.dirname(model_path)
    
    # Save scaler
    scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save label encoder
    encoder_path = os.path.join(artifacts_dir, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Label encoder saved to {encoder_path}")
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'class_names': class_names,
        'num_features': len(feature_names),
        'num_classes': len(class_names),
        'model_config': config
    }
    metadata_path = os.path.join(artifacts_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")


def load_model_artifacts(artifacts_dir: str) -> Dict[str, Any]:
    """
    Load model artifacts for deployment.
    
    Args:
        artifacts_dir: Directory containing model artifacts
        
    Returns:
        Dictionary containing loaded artifacts
    """
    artifacts = {}
    
    # Load scaler
    scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            artifacts['scaler'] = pickle.load(f)
        logger.info(f"Scaler loaded from {scaler_path}")
    
    # Load label encoder
    encoder_path = os.path.join(artifacts_dir, 'label_encoder.pkl')
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            artifacts['label_encoder'] = pickle.load(f)
        logger.info(f"Label encoder loaded from {encoder_path}")
    
    # Load metadata
    metadata_path = os.path.join(artifacts_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            artifacts['metadata'] = json.load(f)
        logger.info(f"Metadata loaded from {metadata_path}")
    
    return artifacts


def preprocess_single_sample(
    sample: Union[Dict[str, float], pd.Series, np.ndarray],
    scaler: StandardScaler,
    feature_names: List[str]
) -> torch.Tensor:
    """
    Preprocess a single sample for inference.
    
    Args:
        sample: Input sample (dict, pandas Series, or numpy array)
        scaler: Fitted StandardScaler
        feature_names: List of expected feature names
        
    Returns:
        Preprocessed tensor ready for model input
    """
    if isinstance(sample, dict):
        # Convert dict to array in correct order
        sample_array = np.array([sample[name] for name in feature_names])
    elif isinstance(sample, pd.Series):
        # Convert Series to array in correct order
        sample_array = sample[feature_names].values
    elif isinstance(sample, np.ndarray):
        # Assume array is already in correct order
        sample_array = sample
    else:
        raise ValueError(f"Unsupported sample type: {type(sample)}")
    
    # Reshape for scaling (scaler expects 2D input)
    sample_2d = sample_array.reshape(1, -1)
    
    # Scale the sample
    sample_scaled = scaler.transform(sample_2d)
    
    # Convert to tensor
    return torch.FloatTensor(sample_scaled)


def postprocess_prediction(
    logits: torch.Tensor,
    label_encoder: LabelEncoder,
    return_probabilities: bool = True
) -> Dict[str, Any]:
    """
    Postprocess model predictions.
    
    Args:
        logits: Raw model output
        label_encoder: Fitted LabelEncoder
        return_probabilities: Whether to return class probabilities
        
    Returns:
        Dictionary containing prediction results
    """
    # Convert to probabilities
    probabilities = torch.softmax(logits, dim=-1)
    
    # Get predicted class
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    # Get confidence score
    confidence = probabilities.max().item()
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'predicted_class_idx': predicted_class_idx
    }
    
    if return_probabilities:
        class_names = label_encoder.classes_
        class_probs = dict(zip(class_names, probabilities.cpu().numpy()))
        result['class_probabilities'] = class_probs
    
    return result


def calculate_feature_importance(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    feature_names: List[str],
    method: str = 'gradient'
) -> pd.DataFrame:
    """
    Calculate feature importance using gradient-based methods.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for computing importance
        feature_names: List of feature names
        method: Method for computing importance ('gradient' or 'permutation')
        
    Returns:
        DataFrame with feature importance scores
    """
    model.eval()
    
    if method == 'gradient':
        # Gradient-based feature importance
        total_gradients = torch.zeros(len(feature_names))
        num_samples = 0
        
        for batch in data_loader:
            x, y = batch
            x.requires_grad_(True)
            
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            
            # Accumulate gradients
            batch_gradients = torch.abs(x.grad).mean(dim=0)
            total_gradients += batch_gradients
            num_samples += x.size(0)
            
            x.grad.zero_()
        
        # Average gradients
        importance_scores = total_gradients / len(data_loader)
        
    else:
        raise NotImplementedError(f"Method '{method}' not implemented")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores.cpu().numpy()
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )


def print_model_summary(model: torch.nn.Module, input_shape: tuple) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model Architecture Summary")
    print("=" * 50)
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {input_shape}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("\nLayer details:")
    print("-" * 50)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            print(f"{name:30} | {str(module):50} | {num_params:>8,} params")


class EarlyStopping:
    """
    Early stopping utility class.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif self._is_better(val_score):
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta 