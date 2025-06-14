# WiFi Intrusion Detection System (IDS) with PyTorch

A professional machine learning framework for detecting WiFi network attacks using deep learning. This project provides a complete pipeline for training, evaluating, and deploying neural network models to classify 13 different types of WiFi attacks.

## 🚀 Features

- **Multiple Model Architectures**: FeedForward, LSTM, and CNN classifiers
- **13 Attack Types**: Deauth, SQL_Injection, SSDP, Evil_Twin, Website_spoofing, Disas, (Re)Assoc, Rogue_AP, Krack, Kr00k, SSH, Botnet, Malware
- **30 Network Features**: Frame, Radio/Physical Layer, and WLAN/802.11 features
- **Production-Ready Training**: PyTorch Lightning with automatic checkpointing, early stopping, and logging
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, per-class metrics
- **Professional Structure**: Industry-standard ML project organization

## 📁 Project Structure

```
wifi_ids_pytorch/
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Raw data
│   ├── processed/              # Processed data
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── data/
│   │   └── dataset.py          # PyTorch Dataset classes
│   ├── models/
│   │   └── classifier.py       # Neural network architectures
│   ├── training/
│   │   └── trainer.py          # PyTorch Lightning training module
│   └── utils/
│       └── helpers.py          # Utility functions
├── scripts/
│   ├── train.py               # Main training script
│   └── evaluate.py            # Model evaluation script
├── models/
│   ├── checkpoints/           # Training checkpoints
│   └── final/                 # Final trained models
├── logs/                      # Training logs
├── results/                   # Evaluation results
├── notebooks/                 # Jupyter notebooks
└── requirements.txt           # Dependencies
```

## 🛠️ Installation

1. **Clone or navigate to the project directory**:
```bash
cd wifi_ids_pytorch
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Update data path** (if needed):
Edit `configs/config.yaml` to point to your data directory:
```yaml
data:
  raw_data_path: "../new_processed_balanced_common_features"
```

## 🎯 Quick Start

### 1. Training a Model

Train a FeedForward classifier:
```bash
python scripts/train.py --config configs/config.yaml
```

Resume from checkpoint:
```bash
python scripts/train.py --config configs/config.yaml --checkpoint models/checkpoints/last.ckpt
```

### 2. Monitor Training

Launch TensorBoard to monitor training progress:
```bash
tensorboard --logdir logs/
```

### 3. Evaluate Model

Evaluate a trained model:
```bash
python scripts/evaluate.py --checkpoint models/checkpoints/best.ckpt --config configs/config.yaml
```

## 📊 Model Architectures

### FeedForward Classifier (Default)
- Deep fully-connected network
- Configurable hidden layers: [512, 256, 128, 64]
- Dropout and batch normalization
- Best for tabular network features

### LSTM Classifier
- Bidirectional LSTM layers
- Captures temporal patterns in network traffic
- Ideal for sequential attack detection

### CNN Classifier
- 1D convolutional layers
- Detects local patterns in network features
- Good for pattern recognition tasks

## ⚙️ Configuration

The `configs/config.yaml` file contains all hyperparameters:

```yaml
# Model Configuration
model:
  architecture: "feedforward"  # feedforward, lstm, cnn
  hidden_dims: [512, 256, 128, 64]
  dropout_rate: 0.3

# Training Configuration
training:
  batch_size: 256
  learning_rate: 0.001
  max_epochs: 100
  optimizer: "adam"
```

## 📈 Data Format

The system expects data in the following format:
- **Directory structure**: One folder per attack type
- **File format**: CSV files with 30 feature columns + 1 Label column
- **Features**: Frame, Radio/Physical Layer, and WLAN/802.11 features
- **Labels**: Attack type names (e.g., "Deauth", "Evil_Twin", etc.)

Example directory structure:
```
data/
├── 1.Deauth/
│   └── deauth_data.csv
├── 2.Disas/
│   └── disas_data.csv
└── ...
```

## 🎯 Model Performance

The system provides comprehensive evaluation metrics:
- **Overall Accuracy**: Multi-class classification accuracy
- **Per-Class Metrics**: Precision, recall, F1-score for each attack type
- **Confusion Matrix**: Visual representation of classification results
- **ROC Curves**: Per-class ROC analysis
- **Feature Importance**: Gradient-based feature ranking

## 🔧 Advanced Usage

### Custom Model Architecture

Create a new model in `src/models/classifier.py`:
```python
class CustomClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Your custom architecture
        
    def forward(self, x):
        # Forward pass
        return output
```

### Custom Training Loop

Extend the Lightning module in `src/training/trainer.py`:
```python
class CustomWiFiIDS(WiFiIDSLightningModule):
    def training_step(self, batch, batch_idx):
        # Custom training logic
        return loss
```

### Experiment Tracking

The system supports both TensorBoard and Weights & Biases:
```bash
# TensorBoard
tensorboard --logdir logs/

# Weights & Biases (configure in config.yaml)
wandb login
```

## 🚀 Deployment

After training, the system saves:
- **Model checkpoint**: `models/checkpoints/best.ckpt`
- **Preprocessors**: Scaler and label encoder
- **Metadata**: Feature names and model configuration

For deployment, load the trained model:
```python
from src.training.trainer import WiFiIDSLightningModule
model = WiFiIDSLightningModule.load_from_checkpoint("models/checkpoints/best.ckpt")
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.13+
- PyTorch Lightning 1.8+
- scikit-learn 1.3+
- pandas, numpy, matplotlib, seaborn
- Optional: CUDA for GPU acceleration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Run tests: `pytest tests/`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch_size in config.yaml
2. **CUDA Error**: Set device to "cpu" in config.yaml
3. **Data Loading Error**: Check data path and CSV format

### Getting Help

- Check the example notebook: `notebooks/wifi_ids_training_example.ipynb`
- Review configuration: `configs/config.yaml`
- Examine logs: `logs/` directory

## 📚 Citation

If you use this framework in your research, please cite:
```bibtex
@software{wifi_ids_pytorch,
  title={WiFi Intrusion Detection System with PyTorch},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/wifi_ids_pytorch}
}
```

---

**Happy Training! 🎯** 