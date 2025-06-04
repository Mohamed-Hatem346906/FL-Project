# FL-Project
federated learning and sparse training techniques using Vision Transformer (ViT) models on the CIFAR-100 dataset
# Federated Learning Experiments with CIFAR-100

This repository contains a comprehensive Jupyter Notebook (`FL-Source Code.ipynb`) that implements a series of federated learning experiments using the CIFAR-100 dataset. It explores centralized training, Federated Averaging (FedAvg), sparse training strategies, and mask overlap analysis to evaluate performance under different training paradigms.

## 📌 Key Features

- **Centralized Training**  
  Traditional training approach using the entire dataset on a single server.

- **Federated Averaging (FedAvg)**  
  Implements distributed training across multiple clients with periodic aggregation.

- **Sparse Training Experiments**  
  Investigates sparsity strategies (e.g., least-sensitive, most-sensitive, hybrid) to optimize computation.

- **Mask Overlap Analysis**  
  Evaluates the similarity of sparsity masks used across different clients.

---

## 🧩 Dependencies

Ensure the following libraries are installed:

- Python 3.11  
- PyTorch (2.6.0+cu124)  
- torchvision (0.21.0+cu124)  
- `timm==1.0.15`  
- `numpy==2.0.2`  
- `scikit-learn`  
- `matplotlib`  
- `seaborn`

### Installation

```bash
pip install timm torch torchvision numpy scikit-learn matplotlib seaborn
```

---

## 💻 Running on Google Colab

Mount Google Drive to save checkpoints and logs:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🧱 Notebook Structure

### 1. Setup and Configuration
- Configures dataset parameters, model architecture, number of clients, etc.
- Sets paths for checkpoints and logging.

### 2. Data Loading and Processing
- Loads and normalizes CIFAR-100.
- Splits into training, validation, and test sets.
- Supports both IID and non-IID data partitioning.

### 3. Model Architecture
- Utilizes a Vision Transformer (ViT) model (`vit_small_patch16_224`) via `timm`.
- Custom `ViTSmallDINO` class used for modeling.

### 4. Training and Evaluation
- Implements functions for training and evaluation in both centralized and federated settings.
- Federated Averaging (FedAvg) for model aggregation.

### 5. Experiments
- Runs centralized training, FedAvg experiments, sparse training with different masking strategies, and mask overlap analysis.
- Includes result logging, plotting, and confusion matrix generation.

### 6. Utilities
- Helper functions for weight aggregation, mask handling, visualization, and metric logging.

---

## 🚀 Usage

Run the following functions within the notebook:

```python
run_centralized_training()
run_fedavg_experiments()
run_sparse_experiments()
analyze_mask_overlap()
```

---

## 📊 Results

- **Centralized Training**: Achieves high accuracy on CIFAR-100 with detailed training/validation curves.
- **Federated Averaging**: Explores class distribution and local update settings to evaluate model generalization.
- **Sparse Training**: Benchmarks various sparsity methods to optimize model complexity and communication costs.
- **Mask Overlap**: Measures how mask similarity affects training dynamics and performance.

---

## ⚙️ Notes

- Designed to run on GPU for efficient training. Ensure CUDA support if running locally.
- Checkpoints are saved to Google Drive (Colab) or a local directory.
- Hyperparameters and training modes can be adjusted in the `Config` class.
