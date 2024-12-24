from typing import List, Type
from document_image_classifiers.interfaces.document_image_classifier import (
    ClassifierSchema,
    StatusCodes,
    DocumentImageClassifierOutput,
)


class DocumentImageGPTClassifierOutput(DocumentImageClassifierOutput[ClassifierSchema]):
    """
    Specialized response format for document classification.
    """
    pass


DOCUMENT_IMAGE_GPT_CLASSIFIER_CONFIG = {
    "model": "gpt-4o-2024-08-06",
    "supported_document_types": ["receipt"],
    "response_format": DocumentImageGPTClassifierOutput,
    "default_response": DocumentImageGPTClassifierOutput(
        status=StatusCodes.UNKNOWN_ERROR,
        details="An unknown error occurred. Please try again later.",
    ),
}
