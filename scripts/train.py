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
from typing import Dict, Any, List

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

from data.dataset import WiFiIDSDataset, create_data_loaders
from models.classifier import create_model, TreeBasedModel
from training.trainer import WiFiIDSLightningModule, create_trainer

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# TabNet is now integrated into src.models.classifier

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
    
    # Create label encoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    class_names = list(label_encoder.classes_)
    logger.info(f"Classes found: {class_names}")
    
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
    """Create PyTorch datasets and data loaders."""
    logger.info("Creating datasets and data loaders...")
    
    # Convert to temporary CSV format for dataset loading
    # (In a real scenario, you might want to modify the dataset class to accept arrays directly)
    temp_dir = Path("temp_data")
    temp_dir.mkdir(exist_ok=True)
    
    # Save train data
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['Label'] = label_encoder.inverse_transform(y_train)
    train_path = temp_dir / "train.csv"
    train_df.to_csv(train_path, index=False)
    
    # Save validation data  
    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['Label'] = label_encoder.inverse_transform(y_val)
    val_path = temp_dir / "val.csv"
    val_df.to_csv(val_path, index=False)
    
    # Save test data
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['Label'] = label_encoder.inverse_transform(y_test)
    test_path = temp_dir / "test.csv"
    test_df.to_csv(test_path, index=False)
    
    # Create datasets (no additional preprocessing since data is already prepared)
    train_dataset = WiFiIDSDataset(
        data_path=str(train_path),
        features=feature_names,
        normalize=False,  # Already normalized
        scaler=scaler,
        label_encoder=label_encoder
    )
    
    val_dataset = WiFiIDSDataset(
        data_path=str(val_path),
        features=feature_names,
        normalize=False,
        scaler=scaler,
        label_encoder=label_encoder
    )
    
    test_dataset = WiFiIDSDataset(
        data_path=str(test_path),
        features=feature_names,
        normalize=False,
        scaler=scaler,
        label_encoder=label_encoder
    )
    
    # 🚀 SPEED OPTIMIZED DataLoader parameters
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    pin_memory = config['hardware']['pin_memory']
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    return train_loader, val_loader, test_loader, train_dataset.get_class_names()


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
    trainer.test(lightning_module, test_loader)
    
    # Save final model and metadata
    final_model_path = os.path.join(model_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    
    # Save config file in the model directory for reference
    import yaml
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Final model and config saved to {model_dir}")


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
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Model and metadata saved to {model_dir}")


def train_xgboost_model(X_train, y_train, X_val, y_val, config: Dict[str, Any]) -> XGBClassifier:
    """
    Train XGBoost model with progress monitoring.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model...")
    
    # Create DMatrix for training and validation
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set up parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': config['data']['num_classes'],
        'eval_metric': 'mlogloss',
        'tree_method': config['xgboost']['tree_method'],
        'gpu_id': config['xgboost']['gpu_id'],
        'predictor': config['xgboost']['predictor'],
        'max_depth': config['xgboost']['max_depth'],
        'learning_rate': config['xgboost']['learning_rate'],
        'n_estimators': config['xgboost']['n_estimators'],
        'subsample': config['xgboost']['subsample'],
        'colsample_bytree': config['xgboost']['colsample_bytree'],
        'min_child_weight': config['xgboost']['min_child_weight'],
        'gamma': config['xgboost']['gamma'],
        'reg_alpha': config['xgboost']['reg_alpha'],
        'reg_lambda': config['xgboost']['reg_lambda'],
        'random_state': config['data']['random_seed']
    }
    
    # Log GPU usage
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        logger.warning("GPU not available, falling back to CPU")
    
    # Set up evaluation list
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    # Train with progress monitoring
    num_round = config['xgboost']['n_estimators']
    model = xgb.train(
        params,
        dtrain,
        num_round,
        evallist,
        verbose_eval=10,  # Print progress every 10 iterations
        early_stopping_rounds=50
    )
    
    # Convert to scikit-learn API for consistency
    xgb_model = XGBClassifier(**params)
    xgb_model._Booster = model
    
    return xgb_model


def train_random_forest_model(X_train, y_train, X_val, y_val, config: Dict[str, Any]) -> RandomForestClassifier:
    """
    Train Random Forest model with progress monitoring.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary
        
    Returns:
        Trained Random Forest model
    """
    logger.info("Training Random Forest model...")
    
    # Create model with warm start for progress monitoring
    model = RandomForestClassifier(
        n_estimators=config['model']['n_estimators'],
        max_depth=config['model']['max_depth'],
        min_samples_split=config['model']['min_samples_split'],
        min_samples_leaf=config['model']['min_samples_leaf'],
        max_features=config['model']['max_features'],
        bootstrap=config['model']['bootstrap'],
        random_state=config['data']['random_seed'],
        n_jobs=-1,  # Use all available cores
        warm_start=True,  # Enable warm start for progress monitoring
        verbose=1  # Enable progress output
    )
    
    # Train in batches to monitor progress
    batch_size = 100  # Number of trees to add in each batch
    n_batches = (config['model']['n_estimators'] + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        current_estimators = min((i + 1) * batch_size, config['model']['n_estimators'])
        model.n_estimators = current_estimators
        model.fit(X_train, y_train)
        
        # Calculate and log metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        logger.info(f"Progress: {current_estimators}/{config['model']['n_estimators']} trees "
                   f"(Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f})")
    
    return model


def train_tabnet_model(X_train, y_train, X_val, y_val, config: Dict[str, Any], feature_names: List[str]) -> TreeBasedModel:
    """
    Train TabNet model with progress monitoring.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary
        feature_names: List of feature names
        
    Returns:
        Trained TabNet model
    """
    logger.info("Training TabNet model...")
    
    # Create and train model using TreeBasedModel wrapper
    model = TreeBasedModel(config)
    model.fit(X_train, y_train, X_val, y_val)
    
    # Log feature importances
    feature_importances = model.get_feature_importance()
    if feature_importances is not None:
        logger.info("Top 10 most important features:")
        for i, (feature, importance) in enumerate(sorted(zip(feature_names, feature_importances), 
                                                       key=lambda x: x[1], reverse=True)[:10]):
            logger.info(f"{i+1}. {feature}: {importance:.4f}")
    
    return model


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
        choices=['feedforward', 'lstm', 'cnn', 'random_forest', 'xgboost', 'tabnet'],
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
        
    elif config.model.architecture in ['random_forest', 'xgboost', 'tabnet']:
        # Tree model training
        train_tree_model(config, X_train, X_val, X_test, y_train, y_val, y_test)
        
        # XGBoost training
        if config.model.architecture == 'xgboost':
            model = train_xgboost_model(X_train, y_train, X_val, y_val, config)
        
        # Random Forest training
        elif config.model.architecture == 'random_forest':
            model = train_random_forest_model(X_train, y_train, X_val, y_val, config)
        
        # TabNet training
        elif config.model.architecture == 'tabnet':
            model = train_tabnet_model(X_train, y_train, X_val, y_val, config, feature_names)
        
    else:
        raise ValueError(f"Unknown model architecture: {config.model.architecture}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 