from abc import ABC, abstractmethod
from pydantic import BaseModel
from PIL import Image
from typing import Union
class DocumentImageProcessor(ABC):
    
    @abstractmethod
    def process(self, image: Union[Image.Image, str]) -> Union[Image.Image, str]:
        """Process the image and return the processed image"""
        pass