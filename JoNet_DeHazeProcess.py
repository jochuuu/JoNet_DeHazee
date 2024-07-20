import os
import cv2
import tensorflow as tf
import numpy as np
from JoNet_DeHazeLiberyCommon import c_JoNet_ImgLiBeryWrapper
import matplotlib.pyplot as plt

class DehazingLayer():
    def __init__(self, initial_tmin=0.1, initial_omega=0.95, initial_radius=15, target_width=256, target_height=256, **kwargs):
        self.tmin = tf.Variable(initial_tmin, dtype=tf.float32)
        self.omega = tf.Variable(initial_omega, dtype=tf.float32)
        self.radius = tf.Variable(initial_radius, dtype=tf.int32)
        self.target_width = target_width
        self.target_height = target_height

    def call(self, inputs):
        print("Joxy Debug: ", inputs.shape)
        dehazed_images = c_JoNet_ImgLiBeryWrapper.JoNet_LiBeryDehaze(
                                inputs, 
                                tmin=self.tmin.numpy(), 
                                omega=self.omega.numpy(), 
                                radius=int(self.radius.numpy())
            )
        
        return dehazed_images

    def load_and_preprocess_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_width, self.target_height))

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    @staticmethod
    def imshow(image, title):
        """
        Wrapper function to display an image using Matplotlib.
        
        Args:
        - image (np.ndarray): Image data in BGR format (OpenCV default).
        - title (str): Title for the image plot.
        """
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
    

def test_dehazing_layer(image_path):
    dehazing_layer = DehazingLayer()

    # Load and preprocess image
    preprocessed_image = dehazing_layer.load_and_preprocess_image(image_path)

    # Apply dehazing
    dehazed_image  = dehazing_layer.call(preprocessed_image)

    # Display original and dehazed images on the same figure
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    dehazing_layer.imshow(preprocessed_image, 'Original')
    
    # Dehazed image
    plt.subplot(1, 2, 2)
    dehazing_layer.imshow(dehazed_image, 'Dehazed')
    
    plt.tight_layout()
    plt.show()

# Test the function with an example image path
test_image_path = 'Test_Image/Haze_158.jpg'
test_dehazing_layer(test_image_path)
