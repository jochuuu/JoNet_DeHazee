import cv2
import numpy as np

class c_JoNet_ImgCusTomWrapper:

    @staticmethod
    def JoNet_CusTomSplit(image):
        if image.shape[-1] != 3:
            raise ValueError("Input image should have 3 channels (RGB)")

        # Get image dimensions
        height, width, _ = image.shape

        # Create three-dimensional arrays for each channel
        b = np.empty((height, width), dtype=np.uint8)
        g = np.empty((height, width), dtype=np.uint8)
        r = np.empty((height, width), dtype=np.uint8)

        # Splitting the image into channels
        b = image[:, :, 0]  # Blue channel
        g = image[:, :, 1]  # Green channel
        r = image[:, :, 2]  # Red channel

        return b, g, r

    @staticmethod
    def JoNet_CusTomMin(*args, **kwargs):
        if len(args) == 0:
            raise ValueError("At least one array or scalar must be provided.")
        
        result = np.array(args[0])
        for arr in args[1:]:
            result = np.minimum(result, arr)
        
        return result

    @staticmethod
    def JoNet_CusTomErode(image, kernel, iterations=1):
        # Ensure image and kernel have the same number of dimensions
        if image.ndim != kernel.ndim:
            raise ValueError("Image and kernel must have the same number of dimensions")
        
        # Ensure kernel dimensions are smaller than or equal to image dimensions
        if any(k_dim > img_dim for k_dim, img_dim in zip(kernel.shape, image.shape)):
            raise ValueError("Kernel size cannot be larger than image dimensions")
        
        # Pad the image with zeros to handle borders
        pad_dims = tuple((k_dim // 2, k_dim // 2) for k_dim in kernel.shape)
        padded_image = np.pad(image, pad_dims, mode='constant')
        
        # Initialize eroded image
        eroded_image = np.copy(image)
        
        # Iterate over all dimensions of the image
        for index in np.ndindex(image.shape):
            # Calculate slice indices based on the current index and kernel shape
            slices = tuple(slice(index[dim], index[dim] + k_dim) for dim, k_dim in enumerate(kernel.shape))
            
            roi = padded_image[slices]
            
            # Ensure roi has the same number of dimensions as kernel for element-wise multiplication
            if roi.shape != kernel.shape:
                raise ValueError(f"ROI shape {roi.shape} does not match kernel shape {kernel.shape}")
            
            # Apply erosion logic
            if np.all(roi * kernel):
                eroded_image[index] = 255
            else:
                eroded_image[index] = 0
        
        # Perform erosion iterations
        for _ in range(iterations - 1):
            temp_image = np.copy(eroded_image)
            
            for index in np.ndindex(image.shape):
                slices = tuple(slice(index[dim], index[dim] + k_dim) for dim, k_dim in enumerate(kernel.shape))
                
                roi = padded_image[slices]
                
                # Ensure roi has the same number of dimensions as kernel for element-wise multiplication
                if roi.shape != kernel.shape:
                    raise ValueError(f"ROI shape {roi.shape} does not match kernel shape {kernel.shape}")
                
                # Apply erosion logic
                if np.all(roi * kernel):
                    temp_image[index] = 255
                else:
                    temp_image[index] = 0
            
            eroded_image = np.copy(temp_image)
        
        return eroded_image
    
    @staticmethod
    def JoNet_CusTomgetStructElement(shape, ksize, anchor=None):
        assert isinstance(shape, str), "Shape must be a string ('RECT', 'CROSS', or 'ELLIPSE')."
        assert shape in ['RECT', 'CROSS', 'ELLIPSE'], "Shape must be one of 'RECT', 'CROSS', or 'ELLIPSE'."
        assert isinstance(ksize, tuple) and len(ksize) == 2, "ksize must be a tuple of two integers (width, height)."
        assert all(isinstance(dim, int) and dim > 0 for dim in ksize), "Dimensions must be positive integers."
        
        if anchor is None:
            anchor = (ksize[0] // 2, ksize[1] // 2)
        else:
            assert isinstance(anchor, tuple) and len(anchor) == 2, "Anchor must be a tuple of two integers (x, y)."
            assert all(isinstance(a, int) and 0 <= a < ksizes for a, ksizes in zip(anchor, ksize)), "Anchor must be within kernel size."

        if shape == 'RECT':
            return np.ones(ksize, dtype=np.uint8)
        elif shape == 'CROSS':
            kernel = np.zeros(ksize, dtype=np.uint8)
            kernel[anchor[0], :] = 1  # Horizontal line
            kernel[:, anchor[1]] = 1  # Vertical line
            return kernel
        elif shape == 'ELLIPSE':
            kernel = np.zeros(ksize, dtype=np.uint8)
            center = tuple(np.array(ksize) // 2)
            axes = (ksize[0] // 2, ksize[1] // 2)
            cv2.ellipse(kernel, center, axes, 0, 0, 360, 1, -1)
            return kernel
        
    @staticmethod
    def JoNet_CusTomgDehaze(image, tmin=0.1, omega=0.95, radius=40):
        b, g, r = c_JoNet_ImgCusTomWrapper.JoNet_CusTomSplit(image.astype(np.float32) / 255.0)
        min_channel = c_JoNet_ImgCusTomWrapper.JoNet_CusTomMin(c_JoNet_ImgCusTomWrapper.JoNet_CusTomMin(r, g), b)
        dark_channel = c_JoNet_ImgCusTomWrapper.JoNet_CusTomErode(min_channel, np.ones((radius, radius), dtype=np.float32))

        num_pixels = dark_channel.size
        num_pixels_to_take = int(num_pixels * (1 - omega))
        flat_dark_channel = dark_channel.reshape(num_pixels)
        indices = flat_dark_channel.argsort()[-num_pixels_to_take:]
        atmospheric_light = np.maximum.reduce(image.reshape(num_pixels, 3)[indices], axis=0) / 255.0

        transmission_map = 1 - omega * dark_channel
        transmission_map = np.maximum(tmin, transmission_map)

        dehazed_image = np.empty_like(image, dtype=np.float32)
        for i in range(3):
            dehazed_image[:, :, i] = ((image[:, :, i].astype(np.float32) / 255.0 - atmospheric_light[i]) / transmission_map + atmospheric_light[i]) * 255.0

        dehazed_image = np.clip(dehazed_image, 0, 255).astype(np.uint8)

        return dehazed_image
