<div align="center">

# 🧠 Brain Tumor MRI Classification

### Advanced Machine Learning Approaches for Brain Tumor Detection from MRI Scans

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red?style=for-the-badge&logo=keras)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)](https://jupyter.org/)

<p align="center">
  A comprehensive, two-approach machine learning pipeline for classifying brain tumors from MRI scans into four categories: <strong>Glioma</strong>, <strong>Meningioma</strong>, <strong>Pituitary Adenoma</strong>, and <strong>Normal</strong> — achieving up to <strong>99.67% accuracy</strong> using ensemble deep learning.
</p>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Clinical Motivation](#-clinical-motivation)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Models Implemented](#-models-implemented)
  - [Approach 1 — Baseline Comparison](#approach-1--baseline-comparison)
  - [Approach 2 — Advanced Ensemble](#approach-2--advanced-ensemble)
- [Performance Results](#-performance-results)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Results & Visualizations](#-results--visualizations)
- [Technical Details](#-technical-details)
- [Limitations & Future Work](#-limitations--future-work)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔭 Overview

Brain tumors are among the most life-threatening neurological conditions. Timely and accurate classification of tumor type from MRI scans is critical for guiding treatment decisions. Manual diagnosis by radiologists is time-consuming and subject to inter-observer variability.

This project implements a **complete ML/DL pipeline** covering two distinct experimental approaches:

| | Approach 1 | Approach 2 |
|---|---|---|
| **Goal** | Broad model comparison | Push accuracy to clinical-grade limits |
| **Models** | CNN, ResNet50, MobileNetV3, EfficientNetB0, SVM, XGBoost, CatBoost | EfficientNetB3 + TTA + CatBoost with embeddings |
| **Best Accuracy** | 91.21% | **99.67%** |
| **Key Techniques** | Transfer learning, voting ensembles, stacking | Advanced augmentation, Test-Time Augmentation, meta-learning |

Both approaches use the **PMRAM Bangladeshi Brain Cancer MRI Dataset**, a real-world clinical dataset of MRI scans from Bangladeshi patients.

---

## 🏥 Clinical Motivation

Brain tumors are classified broadly into:

| Tumor Type | Description | Typical Treatment |
|---|---|---|
| **Glioma** | Arises from glial cells; can be high- or low-grade | Surgery, radiation, chemotherapy |
| **Meningioma** | Originates from the meninges (brain lining); usually benign | Surgery or observation |
| **Pituitary Adenoma** | Benign tumor of the pituitary gland | Medication, surgery, or radiation |
| **Normal** | No tumor present; healthy brain scan | N/A |

Misclassification between these types can lead to severely incorrect treatment protocols. This project aims to assist radiologists with a reliable automated second-opinion system.

---

## 📂 Dataset

**Source:** [PMRAM Bangladeshi Brain Cancer MRI Dataset](https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset) on Kaggle

This dataset is notable because it is specifically collected from **Bangladeshi patients**, addressing representation bias in global medical AI datasets which are often predominantly from Western populations.

### Dataset Characteristics

- **Modality:** MRI (Magnetic Resonance Imaging)
- **Classes:** 4 (Glioma, Meningioma, Pituitary, Normal)
- **Format:** JPEG/PNG images
- **Origin:** PMRAM (clinical institution in Bangladesh)

### Class Distribution

```
Glioma      ████████████████ 
Meningioma  ██████████████
Pituitary   ████████████████████
Normal      ████████████████
```

> ⚠️ **Note:** The dataset is downloaded automatically within the notebooks via the Kaggle API. No manual download is required.

---

## 🏗️ Project Architecture

```
Input MRI Image
       │
       ▼
┌─────────────────────────────────────────────┐
│           DATA PREPROCESSING                │
│  • Resize to 224×224 (Approach 1)          │
│  • Resize to 300×300 (Approach 2, EffB3)   │
│  • Normalization ([0,1] or ImageNet stats)  │
│  • Train/Validation/Test Split              │
└────────────────────┬────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  ┌───────────────┐    ┌────────────────────┐
  │  APPROACH 1   │    │    APPROACH 2      │
  │  Baseline     │    │    Advanced        │
  │  Comparison   │    │    Ensemble        │
  └──────┬────────┘    └─────────┬──────────┘
         │                       │
         ▼                       ▼
  Multiple Independent    EfficientNetB3 +
  Models Trained          Test-Time Aug. +
  & Compared              CatBoost Embeddings
         │                       │
         ▼                       ▼
  Voting / Stacking /     Meta-Learning
  Blending Ensembles      Ensemble Fusion
         │                       │
         └──────────┬────────────┘
                    ▼
           Final Predictions
      (Glioma / Meningioma /
       Pituitary / Normal)
```

---

## 🤖 Models Implemented

### Approach 1 — Baseline Comparison

A broad survey of classical ML and deep learning models to establish baseline performance and understand each model's strengths.

#### Deep Learning Models

| Model | Architecture Notes |
|---|---|
| **Custom CNN** | Conv → BN → ReLU → Pool blocks; Global Average Pooling; Dense classifier; Dropout regularization |
| **ResNet50** | Pre-trained on ImageNet; fine-tuned top layers; residual connections handle vanishing gradients |
| **MobileNetV3** | Lightweight depthwise separable convolutions; suitable for resource-constrained inference |
| **EfficientNetB0** | Compound scaling of depth, width, resolution; strong baseline with fewer parameters |

#### Classical ML Models (on CNN-extracted features)

| Model | Notes |
|---|---|
| **SVM (tuned)** | RBF kernel; features extracted from CNN penultimate layer; hyperparameter-tuned with GridSearchCV |
| **XGBoost** | Gradient boosted trees on flattened CNN features; handles non-linear decision boundaries |
| **CatBoost** | Categorical-aware gradient boosting; robust to overfitting on tabular feature vectors |

#### Ensemble Methods

| Strategy | Description |
|---|---|
| **Hard Voting** | Majority vote across model predictions |
| **Soft Voting** | Average of predicted class probabilities |
| **Stacking (SVM+XGB → LR)** | Two-level: base models feed logistic regression meta-learner |
| **Blending (α-weighted)** | Weighted linear combination of probability outputs; α tuned on validation set |

---

### Approach 2 — Advanced Ensemble

A focused, state-of-the-art approach targeting maximum classification accuracy.

#### EfficientNetB3 + Advanced Augmentation

- **Architecture:** EfficientNetB3 (pre-trained on ImageNet), input size 300×300
- **Fine-tuning strategy:** Gradual unfreezing of top layers
- **Data Augmentation:**
  - Random horizontal/vertical flips
  - Random rotations (±15°)
  - Color jitter (brightness, contrast, saturation)
  - Random zoom and shear
  - Cutout / random erasing
  - MixUp augmentation

#### Test-Time Augmentation (TTA)

At inference time, each image is augmented N times and predictions are averaged, dramatically reducing prediction variance and improving robustness on ambiguous MRI scans.

```
Input Image
    │
    ├──[Aug 1]──► CNN ──► P1
    ├──[Aug 2]──► CNN ──► P2
    ├──[Aug 3]──► CNN ──► P3
    ├──[Aug 4]──► CNN ──► P4
    └──[Aug 5]──► CNN ──► P5
                          │
                     Mean(P1..P5)
                          │
                    Final Prediction
```

#### CatBoost with Embeddings

- Feature extraction from EfficientNetB3's penultimate layer (embedding vectors)
- CatBoost trained on these rich semantic embeddings
- Leverages gradient boosting's robustness to handle edge cases where CNN softmax is uncertain

#### Meta-Learning Ensemble

A learned fusion of CNN_TTA and CatBoost_embed predictions, trained on a held-out validation set to optimally weight each model's contribution.

---

## 📊 Performance Results

### Approach 1 — Top Models

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **CNN + SVM (tuned)** | **91.21%** | **91.12%** | **91.31%** | **91.20%** |
| Stacked Meta (SVM+XGB → LR) | 89.75% | 89.69% | 89.81% | 89.74% |
| Blend α=1.00 | 89.08% | 88.99% | 89.11% | 89.07% |
| Ensemble (Average) | 88.75% | 88.75% | 89.01% | 88.75% |
| EfficientNetB0 | ~87% | ~87% | ~87% | ~87% |
| ResNet50 | ~85% | ~85% | ~85% | ~85% |

### Approach 2 — Top Models

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **CatBoost_embed** | **99.67%** | **99.67%** | **99.67%** | **99.67%** |
| Blend (CNN_TTA + CatBoost) | **99.54%** | **99.54%** | **99.54%** | **99.54%** |
| CNN_TTA (EfficientNetB3) | **99.41%** | **99.40%** | **99.41%** | **99.40%** |

### Performance Progression

```
Approach 1 Best ──────────────────────────► Approach 2 Best
   91.21%                                      99.67%
     │                                            │
  CNN + SVM                              CatBoost_embed
```

> **Key Insight:** The jump from ~91% to ~99.67% comes from the combination of a stronger backbone (EfficientNetB3 vs B0), richer augmentation, Test-Time Augmentation, and using the CNN as a feature extractor for CatBoost rather than as a standalone classifier.

---

## 📁 Repository Structure

```
brain-tumor-mri-classification/
│
├── 📓 BrainTumor_aproach_1.ipynb         # Approach 1: Multi-model comparison
├── 📓 Braintumor_Aproach_2.ipynb         # Approach 2: Advanced ensemble
│
├── 📂 BrainMRI_Results/                  # Results from Approach 1
│   └── 2025-09-01_20-10-25/
│       ├── models/                       # Saved model files (*.keras, *.joblib, *.cbm)
│       ├── figures/                      # Plots: confusion matrices, ROC curves, etc.
│       ├── reports/                      # Classification reports (CSV/JSON)
│       └── numpy/                        # Saved prediction arrays (.npy)
│
├── 📂 BrainMRI_Results_Advanced/         # Results from Approach 2
│   ├── models/
│   ├── figures/
│   └── reports/
│
├── 📂 .github/
│   └── workflows/                        # GitHub Actions CI configuration
│
├── 📄 requirements.txt                   # Python dependencies
├── 📄 setup.py                           # Package setup (optional install)
├── 📄 CHANGELOG.md                       # Version history
├── 📄 CONTRIBUTING.md                    # Contribution guidelines
├── 📄 LICENSE                            # MIT License
└── 📄 README.md                          # This file
```

> **Note:** Trained model weight files (`*.keras`, `*.cbm`, `*.joblib`) are excluded from the repo due to size (>1 GB total). They are regenerated automatically when running the notebooks.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip
- A Kaggle account (for dataset download)
- GPU recommended (NVIDIA with CUDA for TensorFlow acceleration)

### 1. Clone the Repository

```bash
git clone https://github.com/n00rtahsin/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow keras
pip install scikit-learn xgboost catboost
pip install opencv-python-headless
pip install matplotlib seaborn plotly
pip install pandas numpy
pip install imagehash shap lime eli5
pip install kaggle
```

### 4. Set Up Kaggle API Credentials

To allow the notebooks to automatically download the dataset:

1. Go to [kaggle.com](https://www.kaggle.com) → Account → Create API Token
2. Download `kaggle.json`
3. Place it in the appropriate location:

```bash
# Linux/macOS:
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows:
# Place kaggle.json in C:\Users\<YourUser>\.kaggle\
```

### 5. Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

---

## 🚀 Usage

### Recommended Workflow

**Step 1 — Start with Approach 1** (`BrainTumor_aproach_1.ipynb`)

This notebook is self-contained and sequential. It:
- Downloads and preprocesses the dataset automatically
- Trains CNN, ResNet50, MobileNetV3, EfficientNetB0
- Extracts features and trains SVM, XGBoost, CatBoost on them
- Builds voting, stacking, and blending ensembles
- Evaluates all models and saves results to `BrainMRI_Results/`

**Step 2 — Run Approach 2** (`Braintumor_Aproach_2.ipynb`)

This notebook:
- Builds on insights from Approach 1
- Trains EfficientNetB3 with advanced augmentation
- Applies Test-Time Augmentation at inference
- Extracts embeddings and trains CatBoost on them
- Applies meta-learning ensemble fusion
- Saves final results to `BrainMRI_Results_Advanced/`

### Tips for Running

- If using a CPU-only machine, reduce batch sizes and epochs for faster iteration
- Results directories are timestamped automatically so reruns don't overwrite previous results
- All evaluation plots (confusion matrices, ROC curves, training curves) are saved to the `figures/` subdirectories

---

## 📈 Results & Visualizations

The following outputs are generated and saved automatically:

| Output | Description | Location |
|---|---|---|
| **Confusion Matrices** | Per-class TP/FP/FN breakdown for all models | `figures/` |
| **ROC Curves** | One-vs-rest AUC curves per class | `figures/` |
| **PR Curves** | Precision-Recall curves | `figures/` |
| **Training Curves** | Loss & accuracy over epochs | `figures/` |
| **Model Comparison Bar Chart** | Side-by-side accuracy/F1 of all models | `figures/` |
| **Feature Importance (SHAP)** | SHAP values for tree-based models | `figures/` |
| **Classification Reports** | Full per-class metrics | `reports/` |
| **Prediction Arrays** | Saved NumPy arrays of raw predictions | `numpy/` |

---

## 🔬 Technical Details

### Data Preprocessing Pipeline

```
Raw MRI Image
     │
     ▼
Resize (224×224 or 300×300)
     │
     ▼
Normalize pixel values (÷255 or ImageNet mean/std)
     │
     ▼
Data Augmentation (training set only)
     │
     ▼
Batch Generator → Model
```

### CNN Architecture (Custom)

```
Input (224, 224, 3)
    │
Conv2D(32) → BatchNorm → ReLU → MaxPool
    │
Conv2D(64) → BatchNorm → ReLU → MaxPool
    │
Conv2D(128) → BatchNorm → ReLU → MaxPool
    │
GlobalAveragePooling2D
    │
Dense(256) → ReLU → Dropout(0.5)
    │
Dense(4) → Softmax
```

### Transfer Learning Strategy

All pre-trained models (ResNet50, MobileNetV3, EfficientNetB0/B3) are loaded with ImageNet weights. The strategy follows two phases:

1. **Feature extraction:** Freeze all base model layers; train only the top classifier
2. **Fine-tuning:** Unfreeze the top N layers of the base model; train with a very low learning rate (1e-5)

### Ensemble Construction

**Stacking** uses cross-validated out-of-fold predictions to train the meta-learner (Logistic Regression), preventing data leakage.

**Blending** uses a held-out blend set (separate from the test set) to learn optimal α weights.

### Hyperparameter Tuning

- SVM: GridSearchCV over `{C: [0.1, 1, 10], kernel: [rbf, linear]}`
- XGBoost: Random search over learning rate, max depth, subsample
- CatBoost: Default parameters with early stopping on validation loss

---

## ⚠️ Limitations & Future Work

### Current Limitations

- **Single dataset:** Results are on one Bangladeshi clinical dataset; generalization to other populations is not validated
- **No external validation:** No independent held-out test set from a different institution
- **2D only:** MRI volumes are treated as 2D slices; 3D volumetric information is not exploited
- **No localization:** The model classifies but does not segment or locate the tumor
- **Class imbalance handling:** Could be further improved with advanced oversampling (SMOTE on embeddings)

### Planned Future Work

- [ ] **3D CNN / volumetric models** using full MRI scan stacks
- [ ] **Grad-CAM / Saliency maps** for visual explanation of model attention regions
- [ ] **Multi-dataset validation** across publicly available international MRI datasets (BraTS, etc.)
- [ ] **Tumor segmentation** pipeline using U-Net or Segment Anything Model (SAM)
- [ ] **Clinical web interface** for radiologist-facing real-time inference
- [ ] **Federated learning** to train across multiple hospitals without sharing patient data
- [ ] **Uncertainty quantification** with Monte Carlo Dropout or conformal prediction

---

## 🤝 Contributing

Contributions are warmly welcome! Please follow these steps:

1. **Fork** the repository
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and write tests if applicable
4. **Commit with a descriptive message:**
   ```bash
   git commit -m "feat: add Grad-CAM visualization for EfficientNetB3"
   ```
5. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** describing what your change does and why

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more detailed guidelines, including code style and commit message conventions.

---

## 📖 Citation

If you use this project in your research or build upon it, please cite:

```bibtex
@misc{brain_tumor_mri_classification_2025,
  title   = {Brain Tumor MRI Classification using Advanced Machine Learning Approaches},
  author  = {n00rtahsin},
  year    = {2025},
  url     = {https://github.com/n00rtahsin/brain-tumor-mri-classification},
  note    = {GitHub repository}
}
```

### Dataset Citation

```bibtex
@dataset{pmram_bangladeshi_brain_cancer_mri_2024,
  title  = {PMRAM Bangladeshi Brain Cancer MRI Dataset},
  author = {Orvile},
  year   = {2024},
  url    = {https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset}
}
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

You are free to use, modify, and distribute this project, including for commercial purposes, with attribution.

---

## 🙏 Acknowledgments

- **PMRAM** and the **Orvile** Kaggle team for providing and hosting the Bangladeshi Brain Cancer MRI Dataset — particularly valuable for addressing geographic bias in medical AI
- The **TensorFlow / Keras** team for exceptional deep learning frameworks
- The **scikit-learn**, **XGBoost**, and **CatBoost** communities for robust ML tooling
- The **Kaggle** community for dataset hosting and accessible data science infrastructure
- All researchers whose open-source code and papers on medical imaging AI informed this project

---

<div align="center">

**Made with ❤️ for better medical AI**

*If this project helped your research, please ⭐ star the repository!*

</div>
