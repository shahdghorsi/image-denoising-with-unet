
# Image Denoising Autoencoder

This project implements an image denoising autoencoder using the U-Net architecture. The model is designed to remove Gaussian noise from images and is evaluated on both CIFAR-10 and external datasets using two metrics PSNR and MSE

## Project Structure

- **`src/`**: Contains source code and scripts.
  - **`generate-noisy-dataset.py`**: Script to generate and save noisy CIFAR-10 dataset.
  - **`main.py`**: Entry point for the project; trains the model and saves it, can also be used to run the different testing files.
  - **`model.py`**: Defines the U-Net model architecture.
  - **`test_cifar10.py`**: Evaluates the model on the CIFAR-10 test dataset.
  - **`test_non_cifar10.py`**: Evaluates the model on an external image dataset.
  - **`train.py`**: Contains training logic for the U-Net model.
  - **`utils.py`**: Utility functions for data manipulation, noise addition, and metrics calculation.
- **`pyproject.toml`**: Dependency management file for Poetry.
- **`README.md`**: This documentation file.

## Getting Started

### 1. Setup Environment

Ensure that you have [Poetry](https://python-poetry.org/) installed. Poetry is used for managing dependencies.

1. **Install Poetry** (if not already installed):

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. **Install Dependencies**:

    Navigate to the project directory and run:

    ```bash
    poetry install
    ```

    This will install all necessary packages specified in `pyproject.toml`.
    Then to use the environment run
    ```bash
    poetry shell
    ```

### 2. Generate Noisy Dataset

To generate and save a noisy version of the CIFAR-10 dataset, run:

```bash
python src/generate-noisy-dataset.py
```

### 3. Train the Model
To train the U-Net model on the noisy CIFAR-10 dataset, run:

```bash
python src/train.py
```
This script will train the model and save the trained weights.

### 4. Evaluate the Model on CIFAR-10
To test the model on the CIFAR-10 dataset, run:

```bash
python src/test_cifar10.py
```
### 5. Evaluate the Model on External Images
To test the model on external images, update the src/test_non_cifar10.py script with your image path and run:
```bash
python src/test_non_cifar10.py
```
Or to do it all at once which might be handy at some scenarios run main.py and do not forget to uncomment the commands inside.
