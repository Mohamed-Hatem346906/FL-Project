# Federated Vision Transformer (ViT) Experiments on CIFAR-100

This project explores **federated learning**, **sparse training**, and **masking strategies** for Vision Transformers (ViT) using the CIFAR-100 dataset. The codebase enables reproducible experiments on centralized, IID and non-IID federated learning, and analyzes the effect of sparse updates and mask overlap between clients.

## Features

- **Centralized and Federated Training**: Compare single-machine and federated (FedAvg) training paradigms.
- **IID and Non-IID Partitioning**: Flexible data partitioning for realistic federated setups.
- **Vision Transformer Backbone**: Uses state-of-the-art ViT-Small with DINO self-supervised pretraining (via `timm`).
- **Sparse Training and Masking**: Implements Fisher Information, magnitude, random, and hybrid strategies for masking parameters during optimization.
- **Hyperparameter Grid Search**: Automated sweeps for learning rate, batch size, and more.
- **Visualization Utilities**: Training curves, confusion matrices, and heatmaps for results and mask overlaps.
- **Reproducibility**: Seeds, checkpoints, and logs for all experiments.

## Directory Structure

```
.
├── requirements.txt
├── README.md
├── main.ipynb / main.py  # (your main notebook or script)
├── checkpoints/
│   └── (https://drive.google.com/drive/folders/1gM9x-O6LWJUdQDD_9tP6UH3WtuJLITV7?usp=sharing)         # Model checkpoints
├── logs/
│   └──  # Training/validation logs
├── figures/
│   └── ...               # Plots and result images
└── (other utility files)
```

## Installation

This project is designed for Google Colab or any Python 3.8+ environment with GPU support.

1. **Clone the repo** or upload the code files to your Colab or server.
2. **Install dependencies** (see below).

