import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class WiFiIDSDataset(Dataset):
    """
    PyTorch Dataset for WiFi Intrusion Detection System data.
    
    This dataset handles loading and preprocessing of WiFi network traffic data
    for multi-class attack classification.
    """
    
    def __init__(
        self,
        data_path: str,
        features: List[str],
        target_column: str = "Label",
        normalize: bool = True,
        scaler: Optional[StandardScaler] = None,
        label_encoder: Optional[LabelEncoder] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize the WiFi IDS Dataset.
        
        Args:
            data_path: Path to the CSV file or directory containing data
            features: List of feature column names to use
            target_column: Name of the target/label column
            normalize: Whether to normalize the features
            scaler: Pre-fitted StandardScaler (for validation/test sets)
            label_encoder: Pre-fitted LabelEncoder (for validation/test sets)
            transform: Optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.features = features
        self.target_column = target_column
        self.normalize = normalize
        self.transform = transform
        
        # Load and preprocess data
        self.data = self._load_data()
        self.X, self.y = self._prepare_features_and_targets()
        
        # Fit or use provided scalers/encoders
        if scaler is None and normalize:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif scaler is not None and normalize:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)
        else:
            self.scaler = None
            
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)
        else:
            self.label_encoder = label_encoder
            self.y = self.label_encoder.transform(self.y)
            
        # Convert to tensors
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)
        
        logger.info(f"Dataset loaded: {len(self)} samples, {self.X.shape[1]} features")
        
    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV file or directory."""
        if os.path.isdir(self.data_path):
            # Load from multiple CSV files in directory
            data_frames = []
            for file in os.listdir(self.data_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(self.data_path, file)
                    df = pd.read_csv(file_path)
                    data_frames.append(df)
            if not data_frames:
                raise ValueError(f"No CSV files found in directory: {self.data_path}")
            data = pd.concat(data_frames, ignore_index=True)
        else:
            # Load from single CSV file
            data = pd.read_csv(self.data_path)
            
        logger.info(f"Loaded data with shape: {data.shape}")
        return data
        
    def _prepare_features_and_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector."""
        # Select features
        if not all(col in self.data.columns for col in self.features):
            missing = [col for col in self.features if col not in self.data.columns]
            raise ValueError(f"Missing feature columns: {missing}")
            
        X = self.data[self.features].values
        
        # Handle missing values
        if np.isnan(X).any():
            logger.warning("Found missing values in features, filling with median")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            
        # Get targets
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
            
        y = self.data[self.target_column].values
        
        return X, y
        
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.X[idx]
        target = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, target
        
    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        return list(self.label_encoder.classes_)
        
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        return self.features


def create_data_loaders(
    train_dataset: WiFiIDSDataset,
    val_dataset: WiFiIDSDataset,
    test_dataset: WiFiIDSDataset,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader 