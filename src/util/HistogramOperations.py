import cv2
from matplotlib import pyplot as plt

class HistogramProcessor:

    @staticmethod
    def equalize_histogram(image_path_or_array):

        # Check if it is a path or an image array
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
        else:
            image = image_path_or_array

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Equalize the histogram
        equalized_image = cv2.equalizeHist(gray_image)

        return equalized_image
    
    @staticmethod
    def show_images(original_image, equalized_image):
        # Show the original image and the equalized image side by side
        cv2.imshow('Original Image', original_image)
        cv2.imshow('Equalized Image', equalized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    @staticmethod
    def plot_histograms(original_image, equalized_image):
        
        # Convert to grayscale if not
        if len(original_image.shape) == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_image
        
        if len(equalized_image.shape) == 3:
            equalized_gray = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)
        else:
            equalized_gray = equalized_image

        # Plot the histograms
        plt.figure(figsize=(10, 5))
        
        # Histogram of the original image
        plt.subplot(1, 2, 1)
        plt.hist(original_gray.ravel(), 256, [0,256])
        plt.title('Histogram for original image')
        
        # Histogram of the equalized image
        plt.subplot(1, 2, 2)
        plt.hist(equalized_gray.ravel(), 256, [0,256])
        plt.title('Histogram for equalized image')
        
        plt.show()
