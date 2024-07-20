import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, Dense
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Layer, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.image import adjust_gamma
from JoNet_DeHazeLiberyCommon import c_JoNet_ImgLiBeryWrapper
import matplotlib.pyplot as plt
import cv2
import os
import random
from tqdm import tqdm

# Ensure eager execution is enabled
tf.config.run_functions_eagerly(True)

class BandpassFilterLayer(Layer):
    def __init__(self, initial_low_cutoff=5, initial_high_cutoff=50, **kwargs):
        super(BandpassFilterLayer, self).__init__(**kwargs)
        self.low_cutoff = tf.Variable(initial_value=initial_low_cutoff, trainable=True, dtype=tf.float32)
        self.high_cutoff = tf.Variable(initial_value=initial_high_cutoff, trainable=True, dtype=tf.float32)

    def build(self, input_shape):
        # No additional weights or parameters to initialize in this layer
        pass

    def call(self, inputs):
        # Ensure input is cast to complex for FFT
        inputs = tf.cast(inputs, tf.complex64)

        # Perform 2D FFT
        inputs_fft = tf.signal.fft2d(inputs)

        # Get shape of the input
        rows = tf.shape(inputs)[1]
        cols = tf.shape(inputs)[2]
        crow = rows // 2
        ccol = cols // 2

        # Create a meshgrid for frequency coordinates
        freq_row = tf.range(rows, dtype=tf.float32) - tf.cast(crow, tf.float32)
        freq_col = tf.range(cols, dtype=tf.float32) - tf.cast(ccol, tf.float32)
        freq_row, freq_col = tf.meshgrid(freq_row, freq_col, indexing='ij')
        distance = tf.sqrt(tf.square(freq_row) + tf.square(freq_col))

        # Create the bandpass mask
        mask = tf.logical_and(distance > self.low_cutoff, distance < self.high_cutoff)
        mask = tf.cast(mask, tf.complex64)

        # Expand mask to match the shape of inputs_fft
        mask = tf.expand_dims(mask, axis=-1)  # Shape: [256, 256, 1]
        mask = tf.tile(mask, [1, 1, tf.shape(inputs)[-1]])  # Shape: [256, 256, 3]

        # Apply the mask
        filtered_fft = inputs_fft * mask

        # Convert back to spatial domain
        filtered = tf.signal.ifft2d(filtered_fft)
        filtered = tf.math.real(filtered)  # Take the real part of the inverse FFT

        return filtered
    
class AtmosphericLightRemovalLayer(Layer):
    def __init__(self, initial_brightness=0.5, **kwargs):
        super(AtmosphericLightRemovalLayer, self).__init__(**kwargs)
        self.brightness = tf.Variable(initial_value=initial_brightness, trainable=True, dtype=tf.float32)

    def build(self, input_shape):
        # No additional weights or parameters to initialize in this layer
        pass

    def call(self, inputs):
        # Adjust brightness by adding the trainable brightness parameter
        adjusted_image = inputs - self.brightness
        adjusted_image = tf.clip_by_value(adjusted_image, 0.0, 1.0)  # Ensure values are in [0, 1]
        return adjusted_image

    def get_config(self):
        config = super(AtmosphericLightRemovalLayer, self).get_config()
        config.update({
            'initial_brightness': self.brightness.numpy()
        })
        return config

    def get_config(self):
        config = super(BandpassFilterLayer, self).get_config()
        config.update({
            'initial_low_cutoff': self.low_cutoff.numpy(),
            'initial_high_cutoff': self.high_cutoff.numpy()
        })
        return config

class SmokeRemovalLayer(Layer):
    def __init__(self, initial_gamma=1.2, **kwargs):
        super(SmokeRemovalLayer, self).__init__(**kwargs)
        self.gamma = tf.Variable(initial_value=initial_gamma, trainable=True, dtype=tf.float32)

    def build(self, input_shape):
        # No additional weights or parameters to initialize in this layer
        pass

    def call(self, inputs):
        # Ensure input values are in the range [0, 1]
        inputs = tf.clip_by_value(inputs, 0.0, 1.0)
        
        # Apply contrast enhancement with trainable gamma
        enhanced_image = adjust_gamma(inputs, gamma=self.gamma)
        
        return enhanced_image

    def get_config(self):
        config = super(SmokeRemovalLayer, self).get_config()
        config.update({
            'initial_gamma': self.gamma.numpy()
        })
        return config

class EdgeEnhancementLayer(Layer):
    def __init__(self, **kwargs):
        super(EdgeEnhancementLayer, self).__init__(**kwargs)
        # Initialize a trainable kernel for edge enhancement
        self.kernel = tf.Variable(initial_value=self.create_edge_enhance_kernel(), trainable=True, dtype=tf.float32)
    
    def create_edge_enhance_kernel(self):
        # Define a simple edge enhancement kernel
        kernel = tf.constant([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]], dtype=tf.float32)
        return kernel[..., tf.newaxis, tf.newaxis]

    def build(self, input_shape):
        # Create a trainable convolutional kernel
        self.kernel = self.add_weight(shape=(3, 3, 3, 1), initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        # Apply the kernel to the inputs using a 2D convolution
        # Add padding to keep the spatial dimensions the same
        enhanced_image = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        enhanced_image = tf.clip_by_value(enhanced_image, 0.0, 1.0)  # Ensure values are in [0, 1]
        return enhanced_image

    def get_config(self):
        config = super(EdgeEnhancementLayer, self).get_config()
        return config

class DehazingLayer(Layer):
    def __init__(self, initial_tmin=0.1, initial_omega=0.95, initial_radius=15, target_width=256, target_height=256, **kwargs):
        super(DehazingLayer, self).__init__(**kwargs)
        self.tmin = tf.Variable(initial_tmin, dtype=tf.float32)
        self.omega = tf.Variable(initial_omega, dtype=tf.float32)
        self.radius = tf.Variable(initial_radius, dtype=tf.int32)
        self.target_width = target_width
        self.target_height = target_height

    def call(self, inputs):
        dehazed_images = c_JoNet_ImgLiBeryWrapper.JoNet_LiBeryDehaze(
            inputs,
            tmin=self.tmin,
            omega=self.omega,
            radius=tf.cast(self.radius, tf.int32)
        )
        return dehazed_images

    def compute_output_shape(self, input_shape):
        return input_shape  # Modify this if the output shape is different from the input shape

    def get_config(self):
        config = super(DehazingLayer, self).get_config()
        config.update({
            'initial_tmin': self.tmin.numpy(),
            'initial_omega': self.omega.numpy(),
            'initial_radius': self.radius.numpy()
        })
        return config

# Example usage
# Ensure that eager execution is enabled globally
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Define the inputs and the model
inputs = tf.keras.Input(shape=(256, 256, 3))
dehazed_output = DehazingLayer()(inputs)
model = tf.keras.Model(inputs=inputs, outputs=dehazed_output)



class DehazingAutoencoder:
    def __init__(self, target_height=256, target_width=256, ImgLoadFlag=True, HaZeFree_list=None, HaZed_list=None):
        self.model = None
        self.ImgLoadFlag = ImgLoadFlag
        self.HaZeFree_list = HaZeFree_list
        self.HaZed_list = HaZed_list
        self.target_height = target_height
        self.target_width = target_width
        self.train_losses = []
        self.test_losses = []

    def build_model(self):
        input_shape = (self.target_height, self.target_width, 3)
        inputs = Input(shape=input_shape)

        # Apply bandpass filter
        Dehaze_inputs   = DehazingLayer()(inputs)

        filtered_inputs = BandpassFilterLayer(initial_low_cutoff=5, initial_high_cutoff=50)(Dehaze_inputs)

        # Apply atmospheric light removal
        light_removed_inputs = AtmosphericLightRemovalLayer(initial_brightness=0.5)(filtered_inputs)

        # Apply trainable smoke removal
        smoke_remove_inputs = SmokeRemovalLayer(initial_gamma=1.2)(light_removed_inputs)

        # Apply edge enhancement
        edge_enhanced_inputs = EdgeEnhancementLayer()(smoke_remove_inputs)

        # Frequency model layers as preprocessing
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(edge_enhanced_inputs)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
        freq_features = Flatten()(x)
        freq_features = Dense(2, activation='linear')(freq_features)

        # Encoder and decoder
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        self.model = Model(inputs, decoded)
        self.model.compile(optimizer=Adam(), loss='mse')
        print("Autoencoder model built and compiled.")

    def encoder(self, inputs):
        x = Conv2D(64, (3, 3), padding='same')(inputs)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same', strides=2)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same', strides=2)(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = ReLU()(x)
        encoded = MaxPooling2D((2, 2), padding='same', strides=2)(x)
        return encoded

    def decoder(self, encoded):
        x = UpSampling2D((2, 2))(encoded)
        x = Conv2DTranspose(16, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(32, (3, 3), padding='same')(x)
        x = ReLU()(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(64, (3, 3), padding='same')(x)
        x = ReLU()(x)
        decoded = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)
        return decoded

    def generate_random_image(self):
        random_image = np.random.rand(self.target_height, self.target_width, 3)
        random_image = random_image.astype(np.float32)
        random_image = np.expand_dims(random_image, axis=0)
        return random_image

    def load_and_preprocess_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_width, self.target_height))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def train_model(self, num_epochs, HazedFlag=False):
        epoch_losses = []
        image_list = self.HaZed_list if HazedFlag else self.HaZeFree_list

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            batch_losses = []
            for image_name in tqdm(image_list, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
                image = self.load_and_preprocess_image(image_name)
                loss = self.train_step(image)
                batch_losses.append(loss)
            train_loss = np.mean(batch_losses)
            self.train_losses.append(train_loss)
            print(f'Train loss: {train_loss:.4f}')

    def train_modelSampleGen(self, num_epochs):
        epoch_losses = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            batch_losses = []
            for _ in tqdm(range(10), desc=f"Training Epoch {epoch + 1}/{num_epochs}"):  # Generate 10 random samples per epoch
                image = self.generate_random_image()
                loss = self.train_step(image)
                batch_losses.append(loss)
            train_loss = np.mean(batch_losses)
            self.train_losses.append(train_loss)
            print(f'Train loss: {train_loss:.4f}')

    @tf.function
    def train_step(self, image):
        with tf.GradientTape() as tape:
            # Forward pass
            reconstructed = self.model(image, training=True)
            # Compute loss
            loss_fn = tf.keras.losses.MeanSquaredError()
            loss = loss_fn(image, reconstructed)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def test_model(self, num_epochs, test_images):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            mse_loss = tf.keras.losses.MeanSquaredError()
            batch_losses = []
            for image_name in tqdm(test_images, desc=f"Testing Epoch {epoch + 1}/{num_epochs}"): 
                image = self.load_and_preprocess_image(image_name)
                reconstructed = self.model(image, training=False)
                loss = mse_loss(image, reconstructed)
                batch_losses.append(loss)
            test_loss = np.mean(batch_losses)
            self.test_losses.append(test_loss)
            print(f'Test loss: {test_loss:.4f}')

    def De_Haze(self, image):
        return DehazingLayer()(image)

    def Band_Pass_filter(self, image):
        return BandpassFilterLayer()(image)

    def remove_smoke(self, image):
        return SmokeRemovalLayer()(image)

    def remove_atmospheric_light(self, image):
        return AtmosphericLightRemovalLayer()(image)

    def enhance_edges(self, image):
        return EdgeEnhancementLayer()(image)

    def plot_losses(self):
        plt.figure()
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.test_losses, label='Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss')
        plt.legend()
        plt.show()

    def plot_frequency_analysis(self, InputImage, OuputImage):

        image_fft  = tf.signal.fft2d(tf.cast(InputImage, tf.complex64))
        image_freq = tf.math.abs(tf.signal.fftshift(image_fft))

        # Apply bandpass filter
        filtered_image = self.model(InputImage, training=False)
        filtered_fft   = tf.signal.fft2d(tf.cast(filtered_image, tf.complex64))
        filtered_freq  = tf.math.abs(tf.signal.fftshift(filtered_fft))

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(tf.squeeze(image_freq.numpy(), axis=0), cmap='gray')
        plt.title('Original Frequency')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(tf.squeeze(filtered_freq.numpy(), axis=0), cmap='gray')
        plt.title('Filtered Frequency')
        plt.axis('off')

        plt.show()
    
    def plot_all_images(self, inputImage, OuputImage):


            plt.figure(figsize=(12, 6))

            plt.subplot(2, 1, 1)
            plt.title('Input Image')
            plt.imshow(tf.squeeze(inputImage, axis=0), cmap='gray')
            plt.axis('off')

            plt.subplot(2, 1, 2)
            plt.title('DeHaze Image')
            plt.imshow(tf.squeeze(OuputImage, axis=0), cmap='gray')
            plt.axis('off')

            plt.savefig(f'output_images/Sample_image_Process.png')
            plt.show()
            plt.close()
            
def load_image_filenames(hazefree_dir, hazed_dir):
    hazefree_files = [os.path.join(hazefree_dir, f) for f in os.listdir(hazefree_dir) if os.path.isfile(os.path.join(hazefree_dir, f))]
    hazed_files = [os.path.join(hazed_dir, f) for f in os.listdir(hazed_dir) if os.path.isfile(os.path.join(hazed_dir, f))]

    all_files = hazefree_files + hazed_files
    random.shuffle(all_files)
    split_idx = int(0.30 * len(all_files))

    hazed_list = all_files[:split_idx]
    hazefree_list = all_files[split_idx:]

    return hazefree_list, hazed_list

if __name__ == "__main__":
    ImgFlag     = True
    num_epochs  = 20
    numOfSample = 400
    
    if ImgFlag:
        hazefree_dir = 'Test_Image/HazeFre_v0.001'
        hazed_dir = 'Test_Image/HaZed_v0.001'
        HaZeFree_names, HaZed_names = load_image_filenames(hazefree_dir, hazed_dir)
        autoencoder = DehazingAutoencoder(ImgLoadFlag=ImgFlag, HaZeFree_list=HaZeFree_names[0:numOfSample], HaZed_list=HaZed_names[0:numOfSample])
        autoencoder.build_model()
        autoencoder.train_model(num_epochs=num_epochs, HazedFlag=False)
        autoencoder.train_model(num_epochs=num_epochs, HazedFlag=True)
        autoencoder.test_model(num_epochs=num_epochs, test_images=HaZed_names[0:3])
        
        sample_image_path = 'Test_Image/Haze_158.jpg'
        Sample_image      = autoencoder.load_and_preprocess_image(sample_image_path)
        FilterdImage      = autoencoder.model(Sample_image, training=False)

        autoencoder.plot_losses()
        autoencoder.plot_frequency_analysis(Sample_image, FilterdImage)
        autoencoder.plot_all_images(Sample_image, FilterdImage)
    else:
        autoencoder = DehazingAutoencoder(ImgLoadFlag=ImgFlag, target_height=256, target_width=256)
        autoencoder.build_model()
        autoencoder.train_modelSampleGen(num_epochs=num_epochs)
        autoencoder.train_modelSampleGen(num_epochs=num_epochs)