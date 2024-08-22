import numpy as np
import os
from tensorflow.keras.datasets import cifar10
from utils import add_gaussian_noise

def generate_and_save_data(output_dir="data"):
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize images to [0, 1] range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add noise to the images
    x_train_noisy = add_gaussian_noise(x_train)
    x_test_noisy = add_gaussian_noise(x_test)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the datasets as a single .npz file
    np.savez(os.path.join(output_dir, 'noisy_cifar10.npz'), 
             x_train=x_train, x_train_noisy=x_train_noisy, 
             x_test=x_test, x_test_noisy=x_test_noisy, 
             y_train=y_train, y_test=y_test)

if __name__ == "__main__":
    generate_and_save_data()
