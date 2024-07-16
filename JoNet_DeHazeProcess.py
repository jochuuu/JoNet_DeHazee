import cv2
import numpy as np
import matplotlib.pyplot as plt
from JoNet_DeHazeCommon import c_JoNet_ImgLiBeryWrapper, c_JoNet_ImgCusTomWrapper


def Myimshow(image, title):
    """
    Wrapper function to display an image using Matplotlib.
    
    Args:
    - image (np.ndarray): Image data in BGR format (OpenCV default).
    - title (str): Title for the image plot.
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')


def main(input_image_path, output_image_path):
    """
    Main function to perform dehazing on an image and display/save results.
    
    Args:
    - input_image_path (str): Path to the input image file.
    - output_image_path (str): Path to save the dehazed image.
    """
    # Read input image
    input_image = cv2.imread(input_image_path)
    
    if input_image is None:
        print(f"Error: Image not found at {input_image_path}")
        return
    # Perform dehazing
    opencv_dehazedImage = c_JoNet_ImgLiBeryWrapper.JoNet_LiBeryDehaze(input_image)
    custom_dehazedImage = c_JoNet_ImgCusTomWrapper.JoNet_CusTomgDehaze(input_image)
    
    # Display original and dehazed images on the same figure
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    Myimshow(input_image, 'Original')
    
    # Dehazed image
    plt.subplot(1, 2, 2)
    Myimshow(opencv_dehazedImage, 'Dehazed')
    
    plt.tight_layout()
    plt.show()

    # Original image
    plt.subplot(1, 2, 1)
    Myimshow(input_image, 'Original')
    
    # Dehazed image
    plt.subplot(1, 2, 2)
    Myimshow(custom_dehazedImage, 'Dehazed')
    
    plt.tight_layout()
    plt.show()
    
    
    # Save dehazed image
    cv2.imwrite(output_image_path, opencv_dehazedImage)
    print(f'Image saved as {output_image_path}')

if __name__ == '__main__':
    input_image_path = 'Test_Image/Haze_158.jpg'  # Replace with your input image path
    output_image_path = 'Test_Image/DeHazed_1.jpg'  # Replace with your desired output path
    
    # Call main function
    main(input_image_path, output_image_path)
