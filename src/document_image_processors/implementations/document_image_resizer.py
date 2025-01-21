from document_image_processors.interfaces import DocumentImageProcessor
from PIL import Image
from typing import Union, Dict
from utils import read_image
import logging

IMAGE_SIZES = {
    "LARGE": {"source": 2000, "target": 1024},
    "MEDIUM": {"source": 1024, "target": 800},
    "SMALL": {"source": 600, "target": 600},
}

class DocumentImageResizer(DocumentImageProcessor):
    def __init__(self, image_sizes: Dict[str, Dict[str, int]] = IMAGE_SIZES):
        """
        Initialize the DocumentImageResizer with configurable image sizes.
        
        :param image_sizes: A dictionary defining source and target dimensions for each size category.
        """
        self.image_sizes = image_sizes
        
    def process(self, image: Union[Image.Image, str]) -> Union[Image.Image, str]:
        """
        Dynamically resize image based on its original dimensions.
        Ensures the image is not excessively resized to avoid quality loss.
        """
        try:
            image = read_image(image)
            return self._resize_image(image)
        except Exception as e:
            logging.error(f"Error in resizing image: {e}")
            return image

    def _resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_dimension = max(width, height)

        # Determine the appropriate target size based on image dimensions
        target_size = self._get_target_size(max_dimension)
        return self._resize_with_options(image, target_size, maintain_aspect=True)

    def _get_target_size(self, max_dimension: int) -> int:
        """
        Determine the target size based on the maximum dimension of the image.
        """
        for size_category, dimensions in self.image_sizes.items():
            if max_dimension > dimensions["source"]:
                return dimensions["target"]
        return self.image_sizes["SMALL"]["target"]  # Default to SMALL if no match

    def _resize_with_options(self, image: Image.Image, target_size: int, maintain_aspect: bool = True) -> Image.Image:
        """
        Resize the image to a target size dynamically, preserving aspect ratio if specified.
        """
        if maintain_aspect:
            # Calculate new dimensions while maintaining aspect ratio
            width, height = image.size
            scaling_factor = target_size / float(max(width, height))
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
        else:
            # Directly resize to the exact target size (square)
            resized_image = image.resize((target_size, target_size), Image.ANTIALIAS)

        return resized_image