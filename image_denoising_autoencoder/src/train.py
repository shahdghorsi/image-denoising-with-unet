import numpy as np
from model import unet_model


# Load the preprocessed CIFAR-10 dataset from the .npz file
data = np.load('data/noisy_cifar10.npz')
x_train = data['x_train']
x_train_noisy = data['x_train_noisy']

# Build and compile the U-Net model
model = unet_model((32, 32, 3))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train 
history = model.fit(x_train_noisy, x_train, epochs=1, batch_size=64, validation_split=0.1)

# Save
model.save('unet_denoising_model.h5')
