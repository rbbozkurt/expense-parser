from PIL import Image
from typing import List, Union
from document_image_pipelines.interfaces import DocumentImagePipeline, DocumentImagePipelineOutput
from document_image_pipelines.implementations.gpt.config import DOCUMENT_IMAGE_GPT_PIPELINE_CONFIG
from document_image_parsers import DocumentImageParser, DocumentImageParserOutput
from document_image_classifiers import DocumentImageClassifier, DocumentImageClassifierOutput
from document_image_processors import DocumentImageProcessor


class DocumentImageGptPipeline(DocumentImagePipeline):
    """
    GPT-based document image pipeline implementation.
    """

    def __init__(self,
                processors: List[DocumentImageProcessor] = DOCUMENT_IMAGE_GPT_PIPELINE_CONFIG["processors"],
                classifiers: List[DocumentImageClassifier] = DOCUMENT_IMAGE_GPT_PIPELINE_CONFIG["classifiers"],
                parsers: List[DocumentImageParser] = DOCUMENT_IMAGE_GPT_PIPELINE_CONFIG["parsers"]):
        
        super().__init__(
            processors=processors,
            classifiers=classifiers,
            parsers=parsers
        )

    def process(self, document_image: Union[Image.Image, str]) -> DocumentImagePipelineOutput:
        """
        Override the process method if any specific behavior for GPT-based pipeline is required.
        Otherwise, it will use the base class's implementation.
        """
        return super().process(document_image)

    def validate_classifier_and_parser_document_types(self) -> (bool, List[str]):
        """
        Override this method if additional validation logic specific to GPT-based pipeline is required.
        """
        return super().validate_classifier_and_parser_document_types()