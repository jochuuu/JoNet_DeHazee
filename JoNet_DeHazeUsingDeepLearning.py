import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, MaxPooling2D, UpSampling2D, Layer, Conv2DTranspose
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

class CustomLossLayer(Layer):
    def __init__(self, **kwargs):
        super(CustomLossLayer, self).__init__(**kwargs)
        self.w = tf.Variable(1.0, trainable=True, name='w')
        self.G = tf.Variable(0.0, trainable=True, name='G')

    def call(self, inputs):
        reconstructed, original = inputs
        loss = tf.reduce_mean(tf.square(self.w * reconstructed + self.G - original))
        self.add_loss(loss)
        return reconstructed

class DehazingAutoencoder:
    def __init__(self, target_height=256, target_width=256, ImgLoadFlag=True, HaZeFree_list=None, HaZed_list=None):
        self.model = None
        self.ImgLoadFlag = ImgLoadFlag
        self.HaZeFree_list = HaZeFree_list
        self.HaZed_list = HaZed_list
        self.target_height = target_height
        self.target_width = target_width

    def build_model(self):
        input_shape = (self.target_height, self.target_width, 3)
        inputs = Input(shape=input_shape)
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        # custom_loss_layer = CustomLossLayer()([decoded, inputs])
        self.model = Model(inputs, decoded)
        self.model.compile(optimizer='adam')

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
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def train_model(self, num_epochs, HazedFlag=False):
        image_list = self.HaZeFree_list if not HazedFlag else self.HaZed_list
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_losses = []
            for image_name in tqdm(image_list, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
                image = self.load_and_preprocess_image(image_name)
                loss = self.train_step(image)
                epoch_losses.append(loss)
            # train_loss = np.mean(epoch_losses)
            # print(f'Train loss: {train_loss:.4f}')

    def train_modelSampleGen(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_losses = []
            for _ in tqdm(range(10), desc=f"Training Epoch {epoch + 1}/{num_epochs}"):  # Generate 10 random samples per epoch
                image = self.generate_random_image()
                loss = self.train_step(image)
                epoch_losses.append(loss)
            # train_loss = np.mean(epoch_losses)
            # print(f'Train loss: {train_loss:.4f}')

    @tf.function
    def train_step(self, image):
        with tf.GradientTape() as tape:
            reconstructed = self.model(image, training=True)
            loss = tf.reduce_sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def test_model(self, num_epochs, test_images):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            mse_loss = tf.keras.losses.MeanSquaredError()
            epoch_losses = []
            for image_name in tqdm(test_images, desc=f"Training Epoch {epoch + 1}/{num_epochs}"): 
                image = self.load_and_preprocess_image(image_name)
                reconstructed = self.model(image, training=False)
                loss = mse_loss(image, reconstructed)
                epoch_losses.append(loss)
            train_loss = np.mean(epoch_losses)
            print(f'Test loss: {train_loss:.4f}')

    def dehaze_image(self, image_path):
        image = self.load_and_preprocess_image(image_path)
        dehazed_image = self.model(image, training=False)
        dehazed_image = np.squeeze(dehazed_image.numpy(), axis=0)
        dehazed_image = (dehazed_image * 255).astype(np.uint8)
        return dehazed_image

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
        # autoencoder.test_model(num_epochs=num_epochs, test_images = HaZed_names[0:3])
        sample_image_path = 'Test_Image/Haze_158.jpg'
        dehazed_image = autoencoder.dehaze_image(sample_image_path)
        plt.imshow(dehazed_image)
        plt.title('Dehazed Image')
        plt.axis('off')
        plt.show()
    else:
        autoencoder = DehazingAutoencoder(ImgLoadFlag=ImgFlag, target_height=256, target_width=256)
        autoencoder.build_model()
        autoencoder.train_modelSampleGen(num_epochs=num_epochs)
        autoencoder.train_modelSampleGen(num_epochs=num_epochs)
