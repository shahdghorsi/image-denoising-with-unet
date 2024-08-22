from tensorflow.keras.models import load_model
import numpy as np
from utils import visualize_denoised_images, calculate_metrics

# Load the preprocessed CIFAR-10 dataset
data = np.load('data/noisy_cifar10.npz')
x_test = data['x_test']
x_test_noisy = data['x_test_noisy']

# Load the trained model
model = load_model('unet_denoising_model.h5')

# Denoise the test images
num_images_to_visualize = 10
denoised_images = model.predict(x_test_noisy[:num_images_to_visualize])

# Visualize and save denoised images
visualize_denoised_images(x_test_noisy, x_test, denoised_images, num_images=num_images_to_visualize, output_dir="output_images")

# Calculate and print metrics
avg_psnr, avg_mse = calculate_metrics(x_test[:num_images_to_visualize], x_test_noisy[:num_images_to_visualize], denoised_images)
print(f'Average PSNR: {avg_psnr}')
print(f'Average MSE: {avg_mse}')
