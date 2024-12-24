from document_image_processors.interfaces import DocumentImageProcessor
from PIL import Image
from typing import Union
from utils import read_image
import logging

TARGET_FORMAT = "RGB"

class DocumentFormatConverter(DocumentImageProcessor):
    
    
    def __init__(self, target_format: str = TARGET_FORMAT):
        """
        Initialize the DocumentFormatConverter with a target format.
        
        :param target_format: The target format to convert the image to.
        """
        self.target_format = target_format

    
    def process(self, image: Union[Image.Image, str]) -> Union[Image.Image, str]:
        """
        Convert the image to a different format.
        """
        try:
            image = read_image(image)
            return image.convert(self.target_format)
        except Exception as e:
            logging.error(f"Error in converting image format: {e}")
            return image