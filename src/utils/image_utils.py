import base64
import os
from typing import Union
from io import BytesIO
from PIL import Image
import logging
from utils.file_utils import read_image


def encode_image_to_base64(image: Union[Image.Image, str], image_format: str = "JPEG") -> str:
    """
    Encode a PIL Image or an image file path to a base64 string.

    :param image: PIL.Image.Image object or a file path to the image.
    :param image_format: Desired format for the image (e.g., 'JPEG', 'JPG', 'PNG').
    :return: Base64-encoded string of the image.
    """
    
    image = read_image(image)
    buffered = BytesIO()
    image.save(buffered, format=image_format.upper())
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def handle_errors(default_value):
    """
    A decorator to handle errors in methods and return a default value if an exception occurs.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                return default_value
        return wrapper
    return decorator
