# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-09-02

### Added
- Initial release of Brain Tumor MRI Classification project
- Two comprehensive Jupyter notebooks with different approaches
- Multiple machine learning models including CNN, ResNet50, MobileNetV3, EfficientNetB0/B3
- Ensemble methods: voting, stacking, and blending
- Advanced data augmentation techniques
- Test Time Augmentation (TTA) implementation
- Comprehensive model evaluation and comparison
- Results visualization and analysis
- Support for PMRAM Bangladeshi Brain Cancer MRI Dataset
- Automated dataset download from Kaggle
- Model persistence and loading capabilities
- Detailed performance metrics and confusion matrices
- ROC curves and precision-recall curves
- Feature importance analysis

### Models Implemented
- **Approach 1**: CNN, ResNet50, MobileNetV3, EfficientNetB0, SVM, XGBoost, CatBoost
- **Approach 2**: EfficientNetB3 with advanced augmentation, CatBoost with embeddings

### Best Performance
- **Approach 2**: CatBoost_embed: **99.67% accuracy** (state-of-the-art performance)
- **Approach 2**: Blend(CNN_TTA+Cat): **99.54% accuracy**
- **Approach 2**: CNN_TTA: **99.41% accuracy**
- **Approach 1**: CNN+SVM (tuned): 91.21% accuracy
- **Approach 1**: Stacked Meta (SVM+XGB→LR): 89.75% accuracy
- Ensemble methods showing consistent high performance across both approaches

### Documentation
- Comprehensive README with installation and usage instructions
- MIT License
- Requirements specification
- Gitignore for proper version control
