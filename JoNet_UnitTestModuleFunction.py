
import os
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from JoNet_DeHazeLiberyCommon import c_JoNet_ImgLiBeryWrapper
from JoNet_DeHazeUserDefineCommon import c_JoNet_ImgCusTomWrapper

class ImageProcessingTester:

    def __init__(self, height=100, width=100):
        self.height = height
        self.width = width
        self.ksize = (5, 5)
        self.shape = 'ELLIPSE'
        self.kernel = np.ones((3, 3), np.uint8)  # Example kernel for erosion
        self.test_results = []  # List to store test results
        self.image_files = []  # Placeholder for loaded images
        self.mmse_accumulator = {
            "JoNetUnitTestSplit": [],
            "JoNetUnitTestMin": [],
            "JoNetUnitTestErode": [],
            "JoNetUnitStructElement": [],
            "JoNetUnitDehaze":[]
        }  # Dictionary to accumulate MMSE results for each module

    def load_images_from_directory(self, directory):
        if not directory or not os.path.isdir(directory):
            raise ValueError(f"{directory} is not a valid directory")
        self.image_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if not self.image_files:
            raise ValueError(f"No image files found in directory {directory}")
    
    def load_random_image(self):
        if not self.image_files:
            raise ValueError("No images loaded. Use load_images_from_directory first.")
        
        random_image_path = random.choice(self.image_files)
        image = plt.imread(random_image_path)
        return image

    def generate_sample_image(self):
        # Generate a random sample image (BGR)
        return np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)

    def calculate_mmse(self, image1, image2):
        # Calculate Mean Squared Error (MSE) and Percentage Error
        mse = np.mean((image1 - image2) ** 2)
        mmse = mse / (self.height * self.width)
        max_possible_mse = 255 ** 2  # Assuming images are in the range [0, 255]
        percentage_error = (mse / max_possible_mse) * 100
        return mmse, percentage_error

    def test_module(self, module, sample_image, num_samples = 25):
        # Test a specific module
        test_function = getattr(self, f"test_{module}")
        
        print(f"Running {module} tests...")
        for _ in tqdm(range(num_samples), desc=f"{module} Progress", unit="iter"):
            test_function(sample_image)

    def test_JoNetUnitTestSplit(self, sample_image):
        # Test JoNetUnitTestSplit functionality
        opencv_split = c_JoNet_ImgLiBeryWrapper.JoNet_LiBerySplit(sample_image)
        custom_split = c_JoNet_ImgCusTomWrapper.JoNet_CusTomSplit(sample_image)
        
        mmse_split, perc_error_split = self.calculate_mmse(opencv_split[0], custom_split[0])
        tolerance = perc_error_split
        
        self.test_results.append({
            "function": "JoNetUnitTestSplit",
            "mmse": mmse_split,
            "perc_error": perc_error_split,
            "tolerance": tolerance
        })
        
        self.mmse_accumulator["JoNetUnitTestSplit"].append(mmse_split)
        
        assert perc_error_split <= tolerance, f"Percentage error for split functions exceeds {tolerance}%"

    def test_JoNetUnitTestMin(self, sample_image):
        # Test JoNetUnitTestMin functionality

        height, width, channels = sample_image.shape
        random_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

        opencv_min = c_JoNet_ImgLiBeryWrapper.JoNet_LiBeryMin(sample_image, random_image)
        custom_min = c_JoNet_ImgCusTomWrapper.JoNet_CusTomMin(sample_image, random_image)
        
        mmse_min, perc_error_min = self.calculate_mmse(opencv_min, custom_min)
        tolerance = perc_error_min
        
        self.test_results.append({
            "function": "JoNetUnitTestMin",
            "mmse": mmse_min,
            "perc_error": perc_error_min,
            "tolerance": tolerance
        })
        
        self.mmse_accumulator["JoNetUnitTestMin"].append(mmse_min)
        
        assert perc_error_min <= tolerance, f"Percentage error for min functions exceeds {tolerance}%"

    def test_JoNetUnitTestErode(self, sample_image):
        # Adjust kernel channels if necessary
        if sample_image.shape[-1] != self.kernel.shape[-1]:
            # Convert kernel to match image channels
            self.kernel = self.kernel  # Assuming grayscale kernel
        
        # Perform erosion with OpenCV function
        opencv_erode = c_JoNet_ImgLiBeryWrapper.JoNet_LiBeryErode(sample_image, self.kernel)
        
        # Perform erosion with custom function
        custom_erode = c_JoNet_ImgCusTomWrapper.JoNet_CusTomErode(sample_image, self.kernel)
        
        # Calculate MMSE and percentage error
        mmse_erode, perc_error_erode = self.calculate_mmse(opencv_erode, custom_erode)
        tolerance = perc_error_erode  # Set tolerance to percentage error
        
        # Record test results
        self.test_results.append({
            "function": "JoNetUnitTestErode",
            "mmse": mmse_erode,
            "perc_error": perc_error_erode,
            "tolerance": tolerance
        })
        
        # Accumulate MMSE for plotting or further analysis
        self.mmse_accumulator["JoNetUnitTestErode"].append(mmse_erode)
        
        # Assert within tolerance
        assert perc_error_erode <= tolerance, f"Percentage error for erode function exceeds {tolerance}%"

    def test_JoNetUnitStructElement(self):
        # Adjust kernel channels if necessary
        # Perform operation with OpenCV function
        opencv_StructElement = c_JoNet_ImgLiBeryWrapper.JoNet_LiBeryStructElement(self.shape, tuple(self.ksize), anchor=None)
        
        # Perform operation with custom function
        custom_StructElement = c_JoNet_ImgCusTomWrapper.JoNet_CusTomgetStructElement(self.shape, tuple(self.ksize), anchor=None)
        
        # Calculate MMSE and percentage error
        mmse_struct_element, perc_error_struct_element = self.calculate_mmse(opencv_StructElement, custom_StructElement)
        tolerance = perc_error_struct_element  # Set tolerance to percentage error
        
        # Record test results
        self.test_results.append({
            "function": "JoNetUnitStructElement",
            "mmse": mmse_struct_element,
            "perc_error": perc_error_struct_element,
            "tolerance": tolerance
        })
        
        # Accumulate MMSE for plotting or further analysis
        self.mmse_accumulator["JoNetUnitStructElement"].append(mmse_struct_element)
        
        # Assert within tolerance
        assert perc_error_struct_element <= tolerance, f"Percentage error for structural element function exceeds {tolerance}%"

    def test_JoNetUnitDehaze(self, sample_image):

        # Perform operation with OpenCV function
        opencv_dehazedImage = c_JoNet_ImgLiBeryWrapper.JoNet_LiBeryDehaze(sample_image)
        
        # Perform operation with custom function
        custom_dehazedImage = c_JoNet_ImgCusTomWrapper.JoNet_CusTomgDehaze(sample_image)

        
        # Calculate MMSE and percentage error
        mmse_struct_element, perc_error_struct_element = self.calculate_mmse(opencv_dehazedImage, custom_dehazedImage)
        tolerance = perc_error_struct_element  # Set tolerance to percentage error
        
        # Record test results
        self.test_results.append({
            "function": "JoNetUnitDehaze",
            "mmse": mmse_struct_element,
            "perc_error": perc_error_struct_element,
            "tolerance": tolerance
        })
        
        # Accumulate MMSE for plotting or further analysis
        self.mmse_accumulator["JoNetUnitDehaze"].append(mmse_struct_element)
        
        # Assert within tolerance
        assert perc_error_struct_element <= tolerance, f"Percentage error for structural element function exceeds {tolerance}%"


    def run_tests(self, module="all", image_directory=None, use_generated_sample=True, num_sample= 5):        
        # Load images from directory if using actual images
        if not use_generated_sample:
            self.load_images_from_directory(image_directory)

        # Run tests based on specified module or all modules
        if module == "all":
            modules = ["JoNetUnitTestSplit", "JoNetUnitTestMin", "JoNetUnitTestErode", "JoNetUnitStructElement","JoNetUnitDehaze"]
        else:
            modules = [module]
        
        for mod in modules:
            self.test_module(mod, self.generate_sample_image() if use_generated_sample else self.load_random_image(), num_sample)
        
        print("All iterations completed successfully.")
        
        # Print and plot test results
        self.print_and_plot_results()

        # Calculate and print auto-regressive filter
        self.calculate_ar_filter()

    def print_and_plot_results(self):
        # Get the modules with accumulated results
        modules_with_results = [module for module, mmse_list in self.mmse_accumulator.items() if mmse_list]

        # Create subplots based on the number of modules with results
        num_modules = len(modules_with_results)
        if num_modules == 0:
            print("No modules with results.")
            return

        if num_modules == 1:
            fig, axs = plt.subplots(1, 1, figsize=(8, 6))
            axs = [axs]  # Convert single Axes object to list for consistent handling
        else:
            fig, axs = plt.subplots(1, num_modules, figsize=(5*num_modules, 5))

        for i, module in enumerate(modules_with_results):
            mmse_list = self.mmse_accumulator[module]
            axs[i].plot(range(1, len(mmse_list) + 1), mmse_list, marker='o', linestyle='-')
            axs[i].set_title(module)
            axs[i].set_xlabel('Iteration')
            axs[i].set_ylabel('MMSE')

        plt.tight_layout()
        plt.show()

    def calculate_ar_filter(self):
        # Calculate auto-regressive filter based on test results
        if not self.test_results:
            print("No test results available.")
            return
        
        module_mse_sum = {}
        module_perc_error_avg = {}
        module_tolerance_avg = {}
        module_count = {}

        # Initialize dictionaries with zeros
        for module in self.mmse_accumulator:
            module_mse_sum[module] = 0
            module_perc_error_avg[module] = 0
            module_tolerance_avg[module] = 0
            module_count[module] = 0
        
        # Accumulate MSE, percentage error, and tolerance values for each module
        for result in self.test_results:
            module = result['function']
            mmse = result['mmse']
            perc_error = result['perc_error']
            tolerance = result['tolerance']

            module_mse_sum[module] += mmse
            module_perc_error_avg[module] += perc_error
            module_tolerance_avg[module] += tolerance
            module_count[module] += 1  
        
        # Calculate average percentage error and tolerance
        num_tests = len(self.test_results)
        for module in self.mmse_accumulator:
            if module_count[module] > 0:
                module_perc_error_avg[module] /= module_count[module]
                module_tolerance_avg[module] /= module_count[module]
        
        # Print the results
        for module in self.mmse_accumulator:
            mse_sum = module_mse_sum[module]
            ar_count = module_count[module]

            if ar_count > 0:
                print(f"Module: {module}")
                print(f"AR of MSE: {mse_sum/ar_count}")
                print(f"Average Percentage Error: {module_perc_error_avg[module]}%")
                print(f"Average Tolerance: {module_tolerance_avg[module]}%")
                print()
            else:
                print(f"No tests conducted for module: {module}")
                print()

        return module_mse_sum, module_perc_error_avg, module_tolerance_avg