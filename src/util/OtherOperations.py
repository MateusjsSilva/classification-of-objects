import os
import cv2

class Operations:

    # Function to display image
    @staticmethod
    def display(image, image_name, max_width=1200, max_height=800):

        # Resize stacked image to control maximum window size
        if  image.shape[1] > max_width or image.shape[0] > max_height:
            scale_factor = min(max_width / image.shape[1], max_height / image.shape[0])
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

        cv2.imshow(image_name, image)
        cv2.waitKey(0)

    # Function to save the image
    @staticmethod
    def save_image(image, save_path):

        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        cv2.imwrite(save_path, image)

    @staticmethod
    def open_image(image_path):
        # Upload image from given path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to open image in path: {image_path}")
        return image

    @staticmethod
    def resize_image(image, max_width=800, max_height=800):
        """
        Resizes the image to a specified maximum size.
        """
        height, width = image.shape[:2]
        if width > max_width or height > max_height:
            # Determines the resize ratio
            aspect_ratio = width / height
            if aspect_ratio > 1:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            # Resize the image
            image = cv2.resize(image, (new_width, new_height))
        return image