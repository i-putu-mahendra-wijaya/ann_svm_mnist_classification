# MNIST Classification with ANN and SVM

A Jupyter notebook that classifies handwritten digits from the MNIST dataset using two approaches and compares them:

- **Artificial Neural Network (ANN)**: a simple fully connected network built with TensorFlow / Keras
- **Support Vector Machine (SVM)**: an RBF-kernel `SVC` from scikit-learn

## Contents

The notebook is `mnist_classification.ipynb`. It covers:

1. A short background on MNIST: its history, its importance, and why it is still relevant
2. **ANN section**
   - Loading MNIST from Keras and previewing random images
   - A network with two ReLU hidden layers (128 and 64 nodes) and a SoftMax output
   - Training with the Adam optimizer and sparse categorical cross-entropy loss
   - A training and validation loss curve
   - Random inference on test images, showing the prediction, the actual label, and the image
3. **SVM section**
   - An RBF-kernel `SVC` trained on a 10,000-image subset of the training data (`N_TRAIN`) to keep training time short
   - Accuracy and a per-class classification report on the full test set
   - Random inference on test images, in the same format as the ANN section

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management

Main dependencies: TensorFlow, scikit-learn, NumPy, pandas, seaborn, and Jupyter (see `pyproject.toml`).

## Getting Started

```bash
# Install dependencies into .venv
uv sync

# Launch Jupyter
uv run jupyter notebook mnist_classification.ipynb
```

The notebook can also be opened directly in PyCharm using the project's `.venv` interpreter.

Run the cells from top to bottom. The SVM section reuses the scaled data prepared in the ANN section. MNIST is downloaded automatically by Keras on first use.

## Notes

- The SVM trains on a subset by default. Raise `N_TRAIN` (up to 60,000) for a fairer comparison with the ANN, at the cost of a much longer training time.
- scikit-learn's `SVC` handles the 10 classes with a one-vs-one scheme, which trains 45 binary classifiers internally.
