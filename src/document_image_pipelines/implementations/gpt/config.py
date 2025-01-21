from typing import List
from pydantic import BaseModel, Field
from document_image_parsers import DocumentImageParser, DocumentImageGPTReceiptParser
from document_image_classifiers import DocumentImageClassifier, DocumentImageGPTClassifier
from document_image_processors import DocumentImageProcessor, DocumentFormatConverter, DocumentImageResizer
from env_config import OPENAI_API_KEY

_SUPPORTED_DOCUMENT_TYPES = ["receipt"]

_PROCESSORS = [
        DocumentFormatConverter(),
        DocumentImageResizer()
    ]

_CLASSIFIERS = [
        DocumentImageGPTClassifier(
            openai_api_key=OPENAI_API_KEY,
            supported_document_types=_SUPPORTED_DOCUMENT_TYPES
        )
    ]

_PARSERS = [
        DocumentImageGPTReceiptParser(
            openai_api_key=OPENAI_API_KEY,
            target_document_type="receipt"
        )
    ]

DOCUMENT_IMAGE_GPT_PIPELINE_CONFIG = {
    "processors": _PROCESSORS,
    "classifiers": _CLASSIFIERS,
    "parsers": _PARSERS,
}
