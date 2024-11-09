# ML Tools

This directory contains implementations of various classical machine learning algorithms, each organized in a dedicated folder. Each subfolder includes code, documentation, and examples for using the algorithm in typical scenarios.

## Contents

- **ML TOOLS**
    Typical ML tools
  - SGD (Stochastic Gradient Descent)
    - Implementation of the SGD algorithm for optimizing linear models.
    - Includes examples and tests for tuning learning rate and convergence criteria.
  
  - LASSO (Least Absolute Shrinkage and Selection Operator)
    - Code for LASSO regression, used for feature selection and regularization.
    - Includes a demonstration of feature selection on a sample dataset.

  - PCA (Principal Component Analysis)
    - PCA implementation for dimensionality reduction.
    - Notebook with visualizations demonstrating variance explained by each principal component.
    - Facial recognition
      
- **NLP Grammar Checker**
   A tool for detecting grammatical errors and suggesting corrections in text. Includes a transformer-based model and rule-based approach with examples.

## Getting Started

To use any algorithm in this directory, navigate to the specific folder, review the README for usage details, and install any dependencies if required.

### Requirements

A general list of dependencies can be found below. Each subfolder may also have a `requirements.txt` file specific to that algorithm.
```bash
pip install -r requirements.txt
