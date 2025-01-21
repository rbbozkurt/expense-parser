from typing import Generic, TypeVar, Union, List, Dict
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from PIL import Image
from utils import StrEnum
from abc import ABC, abstractmethod


class StatusCodes(StrEnum):
    """
    Enum for response status codes.

    Attributes:
        OK (str): classification was successful.
        INVALID_IMAGE (str): The image format is invalid.
        NO_TEXT (str): No text detected in the image.
        NO_DOCUMENT (str): No document detected in the image.
        UNSUPPORTED_TYPE (str): The document type is unsupported.
        FAKE_DOCUMENT (str): A fake document was detected.
        UNKNOWN_ERROR (str): An unknown error occurred.
    """
    OK = "OK"
    INVALID_IMAGE = "INVALID_IMAGE"
    NO_TEXT = "NO_TEXT"
    NO_DOCUMENT = "NO_DOCUMENT"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    FAKE_DOCUMENT = "FAKE_DOCUMENT"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

# Define a generic type variable for schema
SchemaType = TypeVar("SchemaType", bound=BaseModel)

class DocumentImageClassifierOutput(GenericModel, Generic[SchemaType]):
    """
    Unified response format for document classification.

    Attributes:
        status (StatusCodes): The status of the classification process.
        details (Union[SchemaType, str]): Classification result or error message.
    """
    status: StatusCodes
    details: Union[SchemaType, str] = Field(
        ..., description="Classification result or error details."
    )
    
class ClassifierSchema(BaseModel):
    """
    Schema for receipt classification.

    Attributes:
        document_type (str): The type of document classified.
    """
    document_type: str


# Function to generate error messages dynamically
def generate_error_messages(supported_document_types: List[str]) -> Dict[StatusCodes, str]:
    return {
        StatusCodes.NO_TEXT: "No text detected in the image. Please provide an image with visible text.",
        StatusCodes.NO_DOCUMENT: "No document detected in the image. Please provide an image containing a document.",
        StatusCodes.UNSUPPORTED_TYPE: f"Unsupported document type. Please provide an image of one of the following: {', '.join(doc_type for doc_type in supported_document_types)}.",
        StatusCodes.EXTRACTION_FAILED: "Failed to extract data from the document. Please try again.",
        StatusCodes.FAKE_DOCUMENT: "Fake document detected. Please provide an image of a real document.",
        StatusCodes.UNKNOWN_ERROR: "An unknown error occurred. Please try again later.",
        StatusCodes.INVALID_IMAGE: "Invalid image format. Please provide an image in JPEG or PNG format."
    }
    

class DocumentImageClassifier(ABC, Generic[SchemaType]):
    """
    Abstract base class for document image classifiers.
    """
    def __init__(self, supported_document_types: List[str]):
        print("Supported Document Types: ", supported_document_types)
        self.supported_document_types = supported_document_types

    @abstractmethod
    def classify(self, image: Union[Image.Image, str]) -> DocumentImageClassifierOutput[SchemaType]:
        """
        Classify the document type and return the classification output.

        :param image: The input image (PIL.Image.Image).
        :return: Classification output.
        """
        pass
    
    @property
    @abstractmethod
    def summary(self) -> Dict[str, str]:
        """
        Return a summary of the classifier configuration.

        :return: A dictionary with the classifier configuration.
        """
        pass

    def does_support(self, document_type: str) -> bool:
        """
        Check if the classifier supports a given document type.

        :param document_type: The document type to check.
        :return: True if the document type is supported, False otherwise.
        """
        return document_type in self.supported_document_types
