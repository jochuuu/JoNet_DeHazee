import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm  # Import tqdm for progress bar

# Function to load and preprocess an image from file path
def load_and_preprocess_image(image_path, target_height, target_width, is_hazed=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image using OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize image to target dimensions
    image = cv2.resize(image, (target_width, target_height))
    
    # Convert BGR to RGB (if using OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add haze if required
    if is_hazed:
        image = add_haze(image)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

# Function to add haze to an image
def add_haze(image):
    # Simulated haze effect (for demonstration purposes)
    haze_level = np.random.uniform(0.3, 0.7)  # Random haze intensity between 0.3 and 0.7
    haze = np.ones_like(image) * haze_level
    hazed_image = image * haze + (1 - haze)
    return hazed_image

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Custom callback to compute MSE after each epoch
class MSECallback(tf.keras.callbacks.Callback):
    def __init__(self, image_paths, y_data, model):
        super(MSECallback, self).__init__()
        self.mse_history = []  # Initialize an empty list to store MSE values
        self.x_data = self.load_images(image_paths)  # Load images and convert to float32
        #print(self.x_data)
        self.y_data = np.asarray(self.x_data)  # Convert y_data to float32 if needed
        #print(self.y_data)
        self.y_data = self.y_data.astype(np.float32)
        self.keras_model = model  # Store the Keras model instance

    def load_images(self, image_paths):
        images = []
        for path in image_paths:
            # Load and preprocess image
            image = load_and_preprocess_image(path, target_height, target_width, is_hazed=True)  # Adjust parameters as needed
            images.append(image[0])
        return np.asarray(images)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.keras_model.predict(self.x_data)
        mse = np.mean(np.square(self.y_data - y_pred))
        self.mse_history.append(mse)
        print(f'Epoch {epoch + 1} - loss: {logs["loss"]:.4f} - MSE: {mse:.4f}')

# Define the encoder layers with increased filters
inputs = Input(shape=(None, None, 3))  # Placeholder for image input shape
x = Conv2D(64, (3, 3), padding='same')(inputs)
x = ReLU()(x)
x = MaxPooling2D((2, 2), padding='same', strides=2)(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2), padding='same', strides=2)(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = ReLU()(x)
encoded = MaxPooling2D((2, 2), padding='same', strides=2)(x)

# Define the decoder layers
x = UpSampling2D((2, 2))(encoded)
x = Conv2D(16, (3, 3), padding='same')(x)
x = ReLU()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = ReLU()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = ReLU()(x)
decoded = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

# Create the model
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')  # Use Adam optimizer

# Load haze-free and hazed image paths
haze_free_dir = 'Test_Image/HaZeFre'
hazed_dir = 'Test_Image/HaZed'

haze_free_images = [os.path.join(haze_free_dir, img) for img in os.listdir(haze_free_dir)]
hazed_images = [os.path.join(hazed_dir, img) for img in os.listdir(hazed_dir)]

#print("Haze-free images with full paths:")
#print(haze_free_images)

#print("\nHazed images with full paths:")
#print(hazed_images)

#sys.exit(0) 
# Define total number of figures
total_figures = 500  # Adjust as needed

# Select images randomly based on total figures
num_haze_free = int(total_figures * 0.7)
num_hazed = total_figures - num_haze_free

selected_haze_free = np.random.choice(haze_free_images, num_haze_free, replace=False)
selected_hazed = np.random.choice(hazed_images, num_hazed, replace=False)

# Combine selected images
selected_images = list(selected_haze_free) + list(selected_hazed)
np.random.shuffle(selected_images)

# Train-test split
split_idx = int(len(selected_images) * 0.8)  # 80% train, 20% test
train_images = selected_images[:split_idx]
test_images = selected_images[split_idx:]

# Define target height and width for images
target_height = 256  # Adjust as necessary
target_width = 256   # Adjust as necessary

# Define functions for training and evaluation steps
mse_loss = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(image, model):
    with tf.GradientTape() as tape:
        reconstructed = model(image, training=True)
        loss = mse_loss(image, reconstructed)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test_step(image, model):
    reconstructed = model(image, training=False)
    loss = mse_loss(image, reconstructed)
    return loss

# Train the model with tqdm progress bar
batch_size = 1
num_epochs = 5
#print(train_images)
#print(test_images)
# Usage example
train_mse_callback = MSECallback(train_images, np.zeros_like(train_images), autoencoder)
test_mse_callback  = MSECallback(test_images, np.zeros_like(test_images), autoencoder)


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_losses = []

    # Train on train set
    for image_name in tqdm(train_images, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        if image_name in haze_free_images:
            is_hazed = False
        else:
            is_hazed = True
        
        image = load_and_preprocess_image(image_name, target_height, target_width, is_hazed=is_hazed)
        loss = train_step(image, autoencoder)
        epoch_losses.append(loss)
    
    train_loss = np.mean(epoch_losses)
    train_mse_callback.on_epoch_end(epoch, logs={'loss': train_loss})

    # Evaluate on test set
    test_losses = []
    for image_name in tqdm(test_images, desc=f"Testing Epoch {epoch + 1}/{num_epochs}"):
        if image_name in haze_free_images:
            is_hazed = False
        else:
            is_hazed = True

        image = load_and_preprocess_image(image_name, target_height, target_width, is_hazed=is_hazed)
        loss = test_step(image, autoencoder)
        test_losses.append(loss)
    
    test_loss = np.mean(test_losses)
    test_mse_callback.on_epoch_end(epoch, logs={'loss': test_loss})

# After training, demonstrate dehazing on a sample image
#sample_image_name = test_images[0]  # Change this to any image from test set
#if sample_image_name in haze_free_images:
#    is_hazed = False
#    image_path = os.path.join(haze_free_dir, sample_image_name)
#else:
#    is_hazed = True
#    image_path = os.path.join(hazed_dir, sample_image_name)

image_path   = 'Test_Image/Haze_158.jpg'
sample_image = load_and_preprocess_image(image_path, target_height, target_width, is_hazed=is_hazed)
denoised_sample = autoencoder.predict(sample_image)

# Calculate and print MSE value
mse_value = calculate_mse(sample_image, denoised_sample)
print(f'Mean Squared Error (MSE) between Hazed and Dehazed Image: {mse_value:.4f}')

# Display results: Original, Dehazed, Difference, MSE
plt.figure(figsize=(14, 6))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(sample_image[0])
plt.axis('off')
plt.title('Hazed Image')

# Dehazed Image
plt.subplot(1, 4, 2)
plt.imshow(denoised_sample[0])
plt.axis('off')
plt.title('Dehazed Image')

# Difference Image
plt.subplot(1, 4, 3)
diff_img = np.abs(sample_image[0] - denoised_sample[0])
plt.imshow(diff_img)
plt.axis('off')
plt.title('Difference Image')

# MSE value
plt.subplot(1, 4, 4)
plt.text(0.5, 0.5, f'MSE: {mse_value:.4f}', horizontalalignment='center', verticalalignment='center', fontsize=12)
plt.axis('off')
plt.title('MSE Value')

plt.tight_layout()
plt.show()

# Plot MSE history for train and test
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_mse_callback.mse_history, marker='o', linestyle='-', color='b', label='Train')
plt.plot(range(1, num_epochs + 1), test_mse_callback.mse_history, marker='o', linestyle='-', color='r', label='Test')
plt.title('Mean Squared Error (MSE) per Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()
