
import json
import os
from io import BytesIO
from typing import Union
from PIL import Image
import base64
from datetime import datetime


IMAGE_INPUT_FILE_PATH = "../img/inputs"
IMAGE_OUTPUT_FILE_PATH = "../img/outputs"

JSON_OUTPUT_FILE_PATH = "../results/json"

def save_as_json(data, output_file_name = None):
    """
    Save a dictionary as a JSON file.

    :param data: Dictionary to save.
    :param file_path: Path to save the JSON file.
    """
    if not output_file_name:
        output_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        
    file_path = os.path.join(JSON_OUTPUT_FILE_PATH, output_file_name)
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
        
        
def read_image(image : Union[Image.Image, str]) -> Image.Image:
    """
    Read an image from a PIL.Image.Image object or a file path.

    :param image: PIL.Image.Image object or a file path to the image.
    :return: PIL.Image.Image object.
    """
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str):  # Assume it's a file path
        img_path = os.path.join(IMAGE_INPUT_FILE_PATH, image)
        return Image.open(img_path)
    else:
        raise ValueError("Unsupported image type. Provide a PIL.Image.Image or a file path.")

def save_image(image: [Image.Image, str], output_name: str, image_format: str = "JPEG") -> None:
    """
    Save a PIL Image to a file.

    :param image: PIL.Image.Image object to save or base64-encoded image string.
    :param save_path: Path to save the image.
    :param image_format: Desired format for the image (e.g., 'JPEG', 'JPG', 'PNG').
    """
    
    save_path = os.path.join(IMAGE_OUTPUT_FILE_PATH, output_name)
    
    if isinstance(image, Image.Image):
        image.save(save_path, format=image_format.upper())
    elif isinstance(image, str):
        image = Image.open(BytesIO(base64.b64decode(image)))
        image.save(save_path, format=image_format.upper())
    else:
        raise ValueError("Unsupported image type. Provide a PIL.Image.Image or a base64-encoded image string.")
