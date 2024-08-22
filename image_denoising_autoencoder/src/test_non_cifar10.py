from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from utils import add_gaussian_noise, calculate_metrics

# Load the trained model
model = load_model('unet_denoising_model.h5')

# Load and preprocess an external image
img_path = 'test-images/test-img.jpg'  # Update with your image path
image = load_img(img_path, target_size=(32, 32))
image = img_to_array(image).astype('float32') / 255.0

# Add noise to the image
noisy_image = add_gaussian_noise(np.expand_dims(image, axis=0))

# Denoise the image using the model
denoised_image = model.predict(noisy_image)

# Remove batch dimension
noisy_image = np.squeeze(noisy_image, axis=0)
denoised_image = np.squeeze(denoised_image, axis=0)

# Calculate PSNR and MSE
psnr, mse = calculate_metrics([image], [denoised_image])
print(f'PSNR: {psnr:.2f}, MSE: {mse:.6f}')

# Visualize the original, noisy, and denoised images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_image)
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(denoised_image)
plt.title("Denoised Image")
plt.axis("off")

plt.show()


