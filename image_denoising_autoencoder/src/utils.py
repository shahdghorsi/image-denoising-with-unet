import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
import os

def add_gaussian_noise(images, mean=0.0, std=0.1):
    """Add Gaussian noise to images"""
    noise = np.random.normal(loc=mean, scale=std, size=images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0.0, 1.0)
    return noisy_images

def save_image(image, path):
    """Save an image to a file"""
    plt.imsave(path, image)

def visualize_denoised_images(noisy_images, clean_images, denoised_images, num_images=10, output_dir="output_images"):
    """Visualize and save noisy, denoised, and clean images"""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(20, 6))
    
    for i in range(num_images):
        # Noisy images
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(noisy_images[i])
        plt.title("Noisy")
        plt.axis("off")
        save_image(noisy_images[i], os.path.join(output_dir, f"noisy_image_{i}.png"))

        # Denoised images
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(denoised_images[i])
        plt.title("Denoised")
        plt.axis("off")
        save_image(denoised_images[i], os.path.join(output_dir, f"denoised_image_{i}.png"))

        # Clean images
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(clean_images[i])
        plt.title("Clean")
        plt.axis("off")
        save_image(clean_images[i], os.path.join(output_dir, f"clean_image_{i}.png"))

    plt.show()

def calculate_metrics(clean_images, denoised_images):
    """Calculate PSNR and MSE metrics"""
    psnr_values = []
    mse_values = []
    for i in range(len(clean_images)):
        psnr = peak_signal_noise_ratio(clean_images[i], denoised_images[i])
        mse = mean_squared_error(clean_images[i], denoised_images[i])
        psnr_values.append(psnr)
        mse_values.append(mse)
    avg_psnr = np.mean(psnr_values)
    avg_mse = np.mean(mse_values)
    return avg_psnr, avg_mse
