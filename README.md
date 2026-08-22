# Adaptive Gated DenseNet121 for Paddy Disease Classification: Performance Gains and Limits of Sparsity-Based Pruning
## Overview

This research project investigates the effectiveness of an adaptive layer-gated mechanism for creating sparse neural networks in the context of agricultural disease classification. The study compares a standard DenseNet121 baseline against a novel Adaptive DenseNet architecture that learns to selectively activate layers through learnable gate parameters.

The research demonstrates that adaptive gating can maintain competitive classification performance while enabling network pruning and potential computational efficiency gains.

## Research Objective

To develop and evaluate a layer-gated Adaptive Sparse DenseNet121 model for paddy disease classification, comparing its performance against a standard DenseNet121 baseline in terms of:
- Classification accuracy and F1-score
- Model parameter efficiency
- Pruning capabilities through gate-based sparsity

## Dataset

**Paddy Disease Classification Dataset** (Kaggle)

- **Classes**: 10 disease categories
  - bacterial_leaf_blight
  - bacterial_leaf_streak
  - bacterial_panicle_blight
  - blast
  - brown_spot
  - dead_heart
  - downy_mildew
  - hispa
  - normal
  - tungro

- **Dataset Split**: 
  - Training: 8,325 images
  - Validation: 2,082 images (20% stratified split)
  
- **Image Specifications**: 224×224 RGB images
- **Data Augmentation**: Random horizontal flip, random rotation (15°)

## Model Architectures

### 1. Baseline DenseNet121
- Standard DenseNet121 architecture pretrained on ImageNet
- Modified classifier head for 10-class classification
- **Parameters**: 6,964,106
- **Model Size**: 26.89 MB

### 2. Adaptive DenseNet121 (Layer-Gated)
- Novel architecture with learnable scalar gates on each DenseLayer
- **Key Components**:
  - `ScalarGate`: Learnable sigmoid-gated parameter per layer
  - `GatedDenseLayer`: Wraps standard DenseLayer with gate mechanism
  - `GatedDenseBlock`: Replaces standard DenseBlock with gated layers
  - `ZeroDenseLayer`: Placeholder for pruned layers
  - `AdaptiveDenseNet121`: Full model with adaptive sparse blocks

- **Parameters**: 6,964,164 (including 58 learnable gates)
- **Model Size**: 26.89 MB
- **Sparsity Loss**: L1 regularization on gate values (λ = 5e-4)

## Training Configuration

### Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4
- **Batch Size**: 32
- **Epochs**: 30 (with early stopping, patience=6)
- **Sparsity Lambda**: 5e-4 (adaptive model only)
- **Gate Initialization**: 2.0 (adaptive model only)

### Training Setup
- **Framework**: PyTorch 2.10.0
- **Hardware**: Tesla T4 GPU (Kaggle)
- **Mixed Precision**: AMP (Automatic Mixed Precision)
- **Data Normalization**: ImageNet mean/std

## Results

### Baseline DenseNet121
- **Best Epoch**: 16
- **Validation Accuracy**: 96.78%
- **Validation F1-Score**: 0.9664
- **Validation Precision**: 97.35%
- **Validation Recall**: 96.03%
- **Parameters**: 6,964,106

### Adaptive DenseNet121
- **Best Epoch**: 16
- **Validation Accuracy**: 97.21%
- **Validation F1-Score**: 0.9689
- **Validation Precision**: 97.16%
- **Validation Recall**: 96.65%
- **Parameters**: 6,964,164 (58 gates)

### Performance Comparison
| Metric | Baseline | Adaptive | Improvement |
|--------|----------|----------|-------------|
| Accuracy | 96.78% | 97.21% | +0.43% |
| F1-Score | 0.9664 | 0.9689 | +0.0025 |
| Precision | 97.35% | 97.16% | -0.19% |
| Recall | 96.03% | 96.65% | +0.62% |

### Pruning Analysis
The adaptive model enables structured pruning through gate thresholds:

| Threshold | Parameters | Size (MB) | Accuracy | F1-Score |
|-----------|------------|-----------|----------|----------|
| 0.823 | 6,964,164 | 26.89 | 97.21% | 0.9689 |
| 0.825 | 6,839,683 | 26.41 | 96.88% | 0.9639 |
| 0.827 | 5,111,024 | 19.73 | 17.20% | 0.0505 |
| 0.829 | 1,149,389 | 4.43 | 16.95% | 0.0290 |

**Key Finding**: The model maintains near-original performance with minimal pruning (threshold=0.825, reducing parameters by ~1.8%), but aggressive pruning significantly degrades performance, indicating the gates learn meaningful layer importance.

## Project Structure

```
DenseNet Research/
├── Adaptive/
│   ├── adaptive-densenet-layer-paddy-d-c.ipynb  # Adaptive model training notebook
│   └── pruning_params.txt                        # Pruning experiment results
├── Baseline/
│   ├── densenet121-baseline-paddy-d-c.ipynb     # Baseline model training notebook
│   └── epochs.txt                               # Training epoch logs
├── Adaptive_Gated_DenseNet121_Paddy_Disease_Final_Report_Adaptive.docx  # Research report
└── README.md                                     # This file
```

## Requirements

### Python Packages
```bash
torch>=2.10.0
torchvision
pandas
matplotlib
seaborn
pillow
```

### Hardware Requirements
- GPU with CUDA support (recommended: Tesla T4 or equivalent)
- Minimum 8GB GPU memory
- System RAM: 16GB+ recommended

## Usage

### Running the Baseline Model
1. Open `Baseline/densenet121-baseline-paddy-d-c.ipynb`
2. Ensure the paddy disease dataset is available at the specified path
3. Run all cells to train the baseline DenseNet121 model
4. Results will be saved to the output directory

### Running the Adaptive Model
1. Open `Adaptive/adaptive-densenet-layer-paddy-d-c.ipynb`
2. Ensure the paddy disease dataset is available at the specified path
3. Run all cells to train the adaptive gated DenseNet121 model
4. Results, metrics, and gate values will be saved to the output directory

### Dataset Path Configuration
Both notebooks expect the dataset at:
```python
DATA_DIR = Path('/kaggle/input/competitions/paddy-disease-classification')
```

Modify this path to match your local dataset location.

## Key Features

### Adaptive Model Innovations
1. **Learnable Layer Gates**: Each DenseLayer has a learnable scalar gate that controls its contribution
2. **Sparsity Regularization**: L1 penalty on gate values encourages automatic pruning
3. **Structured Pruning**: Gates enable removal of entire layers while maintaining network structure
4. **Gate Dynamics Tracking**: Monitors gate values throughout training to understand layer importance

### Evaluation Metrics
- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1-Score (macro-averaged)
- Confusion Matrix
- Training/Validation loss curves

## Outputs

### Baseline Model Outputs
- `best_model.pt`: Best model checkpoint
- `metrics.csv`: Training/validation metrics per epoch
- `confusion_matrix.png`: Confusion matrix visualization
- `validation_metrics.png`: Accuracy and F1-score curves

### Adaptive Model Outputs
- `best_model.pt`: Best model checkpoint
- `metrics.csv`: Training/validation metrics per epoch
- `gate_values.csv`: Gate value evolution across epochs
- `confusion_matrix.png`: Confusion matrix visualization
- `validation_metrics.png`: Accuracy and F1-score curves
- `gate_dynamics.png`: Gate value dynamics over training
- `adaptive_densenet_layer_torchscript.pt`: TorchScript export for deployment

## Research Findings

1. **Comparable Performance**: The adaptive model achieves slightly better F1-score (0.9689 vs 0.9664) than the baseline
2. **Gate Learning**: Gates successfully learn layer importance, with values converging to different levels
3. **Pruning Potential**: Limited pruning is possible without significant performance degradation
4. **Interpretability**: Gate values provide insights into which layers are most important for the task

## Future Work

- Explore different gate initialization strategies
- Investigate layer-wise gate initialization
- Experiment with different sparsity regularization strengths
- Test on other agricultural disease classification tasks
- Implement gradual pruning schedules
- Analyze computational efficiency gains on edge devices

## Citation

```
Adaptive Gated DenseNet121 for Paddy Disease Classification Research Project
```

## License

This research project is for academic purposes. Please ensure compliance with the original dataset license terms.

## Contact

For questions about this research, please refer to the project documentation or contact the research team.

## Acknowledgments

- Kaggle for the Paddy Disease Classification dataset
- PyTorch team for the deep learning framework

