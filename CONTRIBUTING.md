# Contributing to Brain Tumor MRI Classification

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Code of Conduct

Please be respectful and professional in all interactions. We want to maintain a welcoming environment for everyone.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the GitHub Issues
2. Provide a clear description of the problem
3. Include steps to reproduce the issue
4. Add relevant error messages or screenshots

### Suggesting Enhancements

1. Open an issue with the "enhancement" label
2. Describe the feature and its benefits
3. Provide examples of how it would be used

### Code Contributions

1. **Fork the repository**
   ```bash
   git clone https://github.com/n00rtahsin/brain-tumor-mri-classification.git
   cd brain-tumor-mri-classification
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   python setup.py
   ```

4. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed

5. **Test your changes**
   - Ensure notebooks run without errors
   - Test with different datasets if applicable
   - Validate model performance

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

7. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Guidelines

### Code Style
- Use meaningful variable and function names
- Follow PEP 8 for Python code
- Add docstrings for functions and classes
- Keep notebook cells focused and well-documented

### Notebook Guidelines
- Clear markdown explanations before code cells
- Include output for demonstration
- Clean up temporary variables
- Ensure reproducibility with random seeds

### Model Development
- Document model architecture decisions
- Include performance comparisons
- Save model checkpoints appropriately
- Add visualization of results

### Documentation
- Update README.md for significant changes
- Add inline comments for complex algorithms
- Update requirements.txt if adding dependencies
- Document new features in CHANGELOG.md

## Pull Request Process

1. **Ensure your PR**:
   - Has a clear title and description
   - References related issues
   - Includes tests if applicable
   - Updates documentation

2. **PR Review Process**:
   - Code review by maintainers
   - Automated testing via GitHub Actions
   - Discussion and feedback incorporation
   - Final approval and merge

3. **After Merge**:
   - Delete your feature branch
   - Update your local main branch

## Areas for Contribution

### High Priority
- **New Model Architectures**: Vision Transformers, ConvNeXt, etc.
- **Advanced Augmentation**: AutoAugment, RandAugment
- **Optimization**: Hyperparameter tuning, neural architecture search
- **Deployment**: Model serving, web interface

### Medium Priority
- **Evaluation Metrics**: Additional metrics, statistical tests
- **Visualization**: Interactive plots, model interpretability
- **Data Processing**: Advanced preprocessing techniques
- **Documentation**: Tutorials, examples

### Nice to Have
- **Multi-dataset Support**: Cross-dataset validation
- **Federated Learning**: Privacy-preserving training
- **Edge Deployment**: Mobile/embedded optimization
- **Real-time Processing**: Streaming inference

## Setting Up Development Environment

### Prerequisites
- Python 3.8+
- Git
- Jupyter Notebook/Lab
- CUDA-capable GPU (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/n00rtahsin/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py
```

### Testing Changes
```bash
# Test notebook execution
jupyter nbconvert --to notebook --execute BrainTumor_aproach_1.ipynb

# Validate code quality
flake8 setup.py
black --check setup.py
```

## Data and Model Guidelines

### Dataset Handling
- Never commit large dataset files
- Use `.gitignore` for data directories
- Document dataset requirements clearly
- Provide scripts for data download/preparation

### Model Files
- Save models in appropriate formats (.keras, .pkl, etc.)
- Use version control for model metadata
- Document model requirements and dependencies
- Provide model loading examples

### Results Reporting
- Include comprehensive evaluation metrics
- Provide comparison with baseline models
- Document computational requirements
- Share reproducible results

## Questions and Support

- **General Questions**: Open a GitHub Discussion
- **Bug Reports**: Create a GitHub Issue
- **Feature Requests**: Open an Issue with "enhancement" label
- **Security Issues**: Email maintainers directly

## Recognition

Contributors will be recognized in:
- README.md contributors section
- CHANGELOG.md for significant contributions
- GitHub contributors page

Thank you for contributing to advancing brain tumor classification research! 🧠✨
