# Brain Tumor MRI Classification

## Overview
This project implements advanced machine learning approaches for brain tumor classification using MRI images from the PMRAM Bangladeshi Brain Cancer MRI Dataset. The project includes two comprehensive approaches with multiple model architectures and ensemble methods.

## Dataset
The project uses the [PMRAM Bangladeshi Brain Cancer MRI Dataset](https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset) from Kaggle, which contains MRI images classified into four categories:
- **Glioma** - A type of brain tumor
- **Meningioma** - A tumor of the meninges
- **Pituitary** - Pituitary adenoma
- **Normal** - Healthy brain scans

## Project Structure

### Notebooks
- **`BrainTumor_aproach_1.ipynb`** - Initial approach with comprehensive model comparison
- **`Braintumor_Aproach_2.ipynb`** - Advanced approach with EfficientNetB3, data augmentation, and ensemble methods

### Results
- **`BrainMRI_Results/`** - Complete results from Approach 1
- **`BrainMRI_Results_Advanced/`** - Results from Approach 2

## Models Implemented

### Approach 1
- Convolutional Neural Networks (CNN)
- ResNet50
- MobileNetV3
- EfficientNetB0
- Support Vector Machine (SVM)
- XGBoost
- CatBoost
- Ensemble methods (Voting, Stacking, Blending)

### Approach 2 (Advanced)
- EfficientNetB3 with advanced data augmentation
- Test Time Augmentation (TTA)
- CatBoost with embeddings
- Meta-learning ensemble

## Performance Results

### Best Performing Models (Approach 1):
| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| CNN+SVM (tuned) | 91.21% | 91.12% | 91.31% | 91.20% |
| Stacked Meta (SVM+XGB→LR) | 89.75% | 89.69% | 89.81% | 89.74% |
| Blend α=1.00 | 89.08% | 88.99% | 89.11% | 89.07% |
| Ensemble (Average) | 88.75% | 88.75% | 89.01% | 88.75% |

### Best Performing Models (Approach 2 - Advanced):
| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| CatBoost_embed | **99.67%** | **99.67%** | **99.67%** | **99.67%** |
| Blend(CNN_TTA+Cat) | **99.54%** | **99.54%** | **99.54%** | **99.54%** |
| CNN_TTA | **99.41%** | **99.40%** | **99.41%** | **99.40%** |

> **Note**: Approach 2 shows significantly improved performance with advanced techniques including EfficientNetB3, sophisticated data augmentation, Test Time Augmentation (TTA), and CatBoost with embeddings.

## Installation

### Prerequisites
```bash
pip install tensorflow keras
pip install scikit-learn xgboost catboost
pip install opencv-python-headless
pip install matplotlib seaborn plotly
pip install pandas numpy
pip install imagehash shap lime eli5
pip install kaggle
```

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/n00rtahsin/brain-tumor-mri-classification.git
   cd brain-tumor-mri-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Set up your Kaggle API credentials
   - The notebooks include automated dataset download from Kaggle

### Note on Model Files
The trained model files (*.keras, *.cbm, *.joblib) are not included in this repository due to their large size (>1GB total). These models will be automatically generated when you run the notebooks. The results data (predictions, metrics, reports) are included to demonstrate the achieved performance.

## Usage

### Running the Notebooks
1. **Start with Approach 1**: Open `BrainTumor_aproach_1.ipynb`
   - This notebook provides a comprehensive comparison of multiple models
   - Includes data preprocessing, model training, and evaluation

2. **Advanced Approach**: Run `Braintumor_Aproach_2.ipynb`
   - Implements state-of-the-art techniques
   - Features advanced data augmentation and ensemble methods

### Key Features
- **Automated Data Pipeline**: Automatic dataset download and preprocessing
- **Multiple Model Architectures**: CNN, ResNet, EfficientNet, and more
- **Ensemble Methods**: Voting, stacking, and blending techniques
- **Comprehensive Evaluation**: Accuracy, F1-score, confusion matrices, ROC curves
- **Visualization**: Training curves, model performance comparisons
- **Model Persistence**: Trained models saved for future use

## Model Architecture Details

### CNN Architecture
- Custom convolutional layers with batch normalization
- Dropout for regularization
- Global average pooling
- Dense layers for classification

### Transfer Learning
- Pre-trained models: ResNet50, MobileNetV3, EfficientNetB0/B3
- Fine-tuning strategies
- Feature extraction approaches

### Ensemble Strategies
- **Voting Classifier**: Hard and soft voting
- **Stacking**: Meta-learner approaches
- **Blending**: Weighted combination of predictions

## Results Analysis

The project includes comprehensive result analysis:
- Confusion matrices for all models
- ROC curves and PR curves
- Feature importance analysis
- Model comparison visualizations
- Performance metrics summary

## File Structure
```
brain-tumor-mri-classification/
├── BrainTumor_aproach_1.ipynb
├── Braintumor_Aproach_2.ipynb
├── requirements.txt
├── README.md
├── LICENSE
├── BrainMRI_Results/
│   └── 2025-09-01_20-10-25/
│       ├── models/
│       ├── figures/
│       ├── reports/
│       └── numpy/
└── BrainMRI_Results_Advanced/
    ├── models/
    ├── figures/
    └── reports/
```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Citation
If you use this work in your research, please cite:

```bibtex
@misc{brain_tumor_mri_classification_2025,
  title={Brain Tumor MRI Classification using Advanced Machine Learning Approaches},
  author={Your Name},
  year={2025},
  howpublished={\\url{https://github.com/yourusername/brain-tumor-mri-classification}}
}
```

## Dataset Citation
```bibtex
@dataset{pmram_bangladeshi_brain_cancer_mri_dataset,
  title={PMRAM Bangladeshi Brain Cancer MRI Dataset},
  author={Orvile},
  year={2024},
  url={https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- PMRAM research team for providing the Bangladeshi Brain Cancer MRI Dataset
- Kaggle community for dataset hosting and platform
- TensorFlow and scikit-learn communities for excellent ML frameworks
