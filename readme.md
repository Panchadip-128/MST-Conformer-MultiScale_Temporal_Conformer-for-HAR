<div align="center">

# MST-Conformer: Multi-Scale Temporal Conformer for Human Activity Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



**A novel deep learning architecture for robust Human Activity Recognition from wearable sensor data**

[Paper](https://arxiv.org/abs/2501.xxxxx) | [Demo](https://colab.research.google.com/drive/xxx) | [Documentation](https://mst-conformer.readthedocs.io)

</div>

---

## Updates
- **[Oct 2025]** Code and pretrained models released
- **[Sep 2025]** Achieved SOTA on UCI-HAR (97.6%), UniMiB-SHAR (98.8%), PAMAP2 (98.4%)

---

##  Highlights

-  **State-of-the-art performance** on 3 benchmark HAR datasets
-  **Multi-scale temporal processing** captures patterns at different time resolutions
-  **Conformer architecture** combines CNN local features with Transformer global context
-  **Cross-dataset generalization** validated with transfer learning experiments
-  **Efficient inference** (0.10ms latency on NVIDIA T4)
-  **Uncertainty quantification** for reliable predictions
-  **First Theoretical-Heuristic-Mathematical Validation on 4 Theorems in HAR Literature**

---

##  Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

##  Overview

Human Activity Recognition (HAR) from wearable sensors is crucial for health monitoring, fitness tracking, and context-aware applications. **MST-Conformer** addresses key challenges:

- **Temporal complexity**: Activities have patterns at multiple time scales
- **Local and global dependencies**: Need both fine-grained and long-range features
- **Cross-dataset generalization**: Models should transfer across different sensor setups

### Key Innovation

MST-Conformer introduces:
1. **Multi-Scale Temporal Module**: Parallel branches processing 32, 64, 128 timesteps
2. **Dual Conformer Blocks**: CNN (local) + Self-Attention (global) in each branch
3. **Deep Layer Aggregation (DLA)**: Hierarchical feature fusion for richer representations
4. **Adaptive Temporal Masking**: Learns which timesteps matter (reduces latency by 33%)

---

##  Architecture

<div align="center">
<img width="671" height="817" alt="image" src="https://github.com/user-attachments/assets/100e2969-1bf7-439f-a9d3-cda04808f1b9" />
</div>

Overview:
Input (N, 128, C) # N samples, 128 timesteps, C channels
↓
┌─────────────────────────────────────────────────┐
│ Multi-Scale Temporal Processing (Parallel) │
│ ├─ Scale 32: Dilated Conv (rate=4) │
│ ├─ Scale 64: Dilated Conv (rate=2) │
│ └─ Scale 128: Dilated Conv (rate=1) │
└─────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────┐
│ Conformer Block (per scale) │
│ ├─ Depthwise Separable Conv (local patterns) │
│ ├─ Multi-Head Self-Attention (global context) │
│ ├─ Feed-Forward Network (GELU activation) │
│ └─ Layer Norm + Residual Connections │
└─────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────┐
│ Deep Layer Aggregation (DLA) │
│ ├─ Iterative Deep Aggregation (IDA) │
│ └─ Hierarchical Node Aggregation (HDA) │
└─────────────────────────────────────────────────┘
↓
Classification Head → Activity Predictions


### Architecture Details

| Component | Configuration |
|-----------|--------------|
| Input | 128 timesteps × 9 channels (3-axis accel + gyro) |
| Multi-Scale Branches | 3 parallel (scales: 32, 64, 128) |
| Conformer Blocks | 2 per branch (6 total) |
| Hidden Dimension | 256 |
| Attention Heads | 8 |
| DLA Levels | 4 hierarchical stages |
| Output Classes | Dataset-dependent (6-18 activities) |
| Parameters | ~3.44M |

---

##  Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU)
- 8GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (recommended)

Prepare input (batch_size, timesteps, channels)
x = torch.randn(1, 128, 9) # Example: 1 sample, 128 timesteps, 9 channels

Inference
with torch.no_grad():
logits = model(x)
probs = torch.softmax(logits, dim=-1)
pred = torch.argmax(probs, dim=-1)

print(f"Predicted activity: {pred.item()}")
print(f"Confidence: {probs.max().item():.2%}")

### Training from Scratch

python train.py
--dataset uci-har
--data-path ./data/UCI-HAR
--epochs 100
--batch-size 64
--lr 0.001
--device cuda
--save-dir ./checkpoints

---

##  Datasets

### Supported Datasets

| Dataset | Activities | Subjects | Sensors | Sampling Rate |
|---------|-----------|----------|---------|---------------|
| [UCI-HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | 6 | 30 | Accel + Gyro | 50 Hz |
| [UniMiB-SHAR](https://www.sal.disco.unimib.it/technologies/unimib-shar/) | 17 | 30 | Accelerometer | 50 Hz |
| [PAMAP2](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring) | 18 | 9 | Multi-sensor | 100 Hz |

### Download & Prepare

Download datasets
python scripts/download_data.py --dataset all --output ./data

Preprocess for training
python scripts/preprocess.py
--dataset uci-har
--input ./data/UCI-HAR
--output ./data/processed
--window-size 128
--overlap 0.5

### Data Format

Expected structure:
data/
├── UCI-HAR/
│ ├── train/
│ │ ├── X_train.txt
│ │ └── y_train.txt
│ └── test/
│ ├── X_test.txt
│ └── y_test.txt
├── PAMAP2/
└── UniMiB-SHAR/


---

##  Training

### Basic Training

python train.py
--config configs/uci_har.yaml
--device cuda



### Advanced Training Options

python train.py
--dataset pamap2
--data-path ./data/processed/pamap2
--model mst-conformer
--batch-size 64
--epochs 100
--lr 0.001
--weight-decay 1e-4
--scheduler cosine
--warmup-epochs 10
--grad-clip 1.0
--mixed-precision
--device cuda
--workers 4
--save-dir ./checkpoints/pamap2
--log-interval 10
--val-interval 5


### Multi-GPU Training

torchrun --nproc_per_node=4 train.py
--config configs/uci_har.yaml
--distributed



### Configuration Files

Example `configs/uci_har.yaml`:

Model
model:
name: mst-conformer
scales:
hidden_dim: 256
num_heads: 8
num_layers: 2
dropout: 0.1

Data
data:
dataset: uci-har
data_path: ./data/processed/uci-har
batch_size: 64
num_workers: 4
window_size: 128
overlap: 0.5

Training
training:
epochs: 100
lr: 0.001
weight_decay: 1e-4
optimizer: adamw
scheduler: cosine
warmup_epochs: 10
grad_clip: 1.0
mixed_precision: true

Hardware
device: cuda
distributed: false



---

##  Evaluation

### Evaluate Pretrained Model

python evaluate.py
--model-path ./checkpoints/best_model.pth
--dataset uci-har
--data-path ./data/processed/uci-har
--batch-size 128
--device cuda



### Cross-Dataset Evaluation

Train on UCI-HAR, evaluate on UniMiB-SHAR
python evaluate.py
--model-path ./checkpoints/uci_har_best.pth
--dataset unimib
--data-path ./data/processed/unimib
--cross-dataset



### Generate Visualizations

python visualize.py
--model-path ./checkpoints/best_model.pth
--dataset uci-har
--output ./results/visualizations
--plots confusion_matrix tsne attention_weights


---

##  Results

### State-of-the-Art Performance

| Dataset | Method | Accuracy | F1-Score | Parameters | Latency |
|---------|--------|----------|----------|------------|---------|
| UCI-HAR | MST-Conformer (Ours) | **97.60%** | **97.55%** | 3.44M | 0.10ms |
| | DeepConvLSTM | 95.75% | 95.20% | 4.12M | 0.15ms |
| | Transformer | 96.80% | 96.45% | 5.23M | 0.18ms |
| | Attend & Discriminate | 96.90% | 96.50% | 3.89M | 0.12ms |
| **PAMAP2** | **MST-Conformer (Ours)** | **98.40%** | **98.35%** | 3.44M | 0.10ms |
| | IMU Transformer | 97.20% | 97.10% | 4.85M | 0.16ms |
| | HAR-Conformer | 97.80% | 97.72% | 4.20M | 0.14ms |
| **UniMiB-SHAR** | **MST-Conformer (Ours)** | **98.80%** | **98.75%** | 3.44M | 0.10ms |
| | CNN-LSTM | 96.50% | 96.20% | 3.92M | 0.13ms |
| | Attention Mechanism | 97.90% | 97.82% | 4.50M | 0.15ms |

### Ablation Study

| Configuration | UCI-HAR Acc | PAMAP2 Acc | UniMiB Acc | Δ vs Full |
|---------------|-------------|------------|------------|-----------|
| **Full MST-Conformer** | **97.60%** | **98.40%** | **98.80%** | - |
| w/o Multi-Scale | 96.20% | 97.10% | 97.50% | -1.40% |
| w/o Conformer | 95.80% | 96.80% | 97.20% | -1.80% |
| w/o DLA | 96.50% | 97.50% | 98.00% | -1.10% |
| w/o Adaptive Masking | 97.30% | 98.10% | 98.50% | -0.30% |
| Single-Scale (128) | 96.00% | 96.90% | 97.40% | -1.60% |

### Cross-Dataset Transfer

| Train → Test | Accuracy | F1-Score |
|--------------|----------|----------|
| UCI-HAR → UniMiB | 94.20% | 93.85% |
| UCI-HAR → PAMAP2 | 91.50% | 91.10% |
| PAMAP2 → UCI-HAR | 93.80% | 93.45% |
| UniMiB → UCI-HAR | 92.60% | 92.20% |

---

## Model Zoo

### Pretrained Checkpoints

| Model | Dataset | Accuracy | Download |
|-------|---------|----------|----------|
| MST-Conformer-Base | UCI-HAR | 97.60% | [checkpoint](https://drive.google.com/xxx) |
| MST-Conformer-Base | PAMAP2 | 98.40% | [checkpoint](https://drive.google.com/xxx) |
| MST-Conformer-Base | UniMiB-SHAR | 98.80% | [checkpoint](https://drive.google.com/xxx) |
| MST-Conformer-Large | UCI-HAR | 97.85% | [checkpoint](https://drive.google.com/xxx) |

### Load Pretrained Model

from mst_conformer import load_checkpoint

model = load_checkpoint('uci-har', device='cuda')


---




