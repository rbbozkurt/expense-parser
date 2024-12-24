from abc import ABC, abstractmethod
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Union, Dict
from document_image_parsers import DocumentImageParser, DocumentImageParserOutput
from document_image_parsers import StatusCodes as ParserStatusCodes
from document_image_classifiers import DocumentImageClassifier, DocumentImageClassifierOutput
from document_image_classifiers import StatusCodes as ClassifierStatusCodes
from document_image_processors import DocumentImageProcessor
from utils import read_image, StrEnum


class StatusCodes(StrEnum):
    FINISHED_SUCCESS = "FINISHED_SUCCESS"
    ERROR = "ERROR"

ERROR_MESSAGE = {
    StatusCodes.FINISHED_SUCCESS: "Document image processing finished successfully.",
    StatusCodes.ERROR: "Error in processing document image.",
}

class DocumentImagePipelineOutput(BaseModel):
    status: StatusCodes
    details: Union[DocumentImageParserOutput, str] = Field(
        default=None,
        description="Details of the document image processing."
    )

class DocumentImagePipeline(ABC):
    def __init__(self,
                processors: List[DocumentImageProcessor] = [],
                classifiers: List[DocumentImageClassifier] = [],
                parsers: List[DocumentImageParser] = []):
        
        self.processors = processors
        self.classifiers = classifiers
        self.parsers = parsers

    @abstractmethod
    def process(self, document_image: Union[Image.Image, str]) -> DocumentImagePipelineOutput:
        """
        Process the document image through the pipeline.
        """
        if not document_image:
            raise ValueError("An image must be provided.")

        is_valid, missing_documents = self.validate_classifier_and_parser_document_types()
        if not is_valid:
            return DocumentImagePipelineOutput(
                status=StatusCodes.ERROR,
                details=f"Missing parsers for document types: {missing_documents}",
            )
        try:
            document_image = read_image(document_image)
        except Exception as e:
            return DocumentImagePipelineOutput(
                status=StatusCodes.ERROR,
                details=f"Error in reading image: {e}",
            )

        try:
            # Apply processors
            for processor in self.processors:
                document_image = processor.process(document_image)
        except Exception as e:
            return DocumentImagePipelineOutput(
                status=StatusCodes.ERROR,
                details=f"Error in processing image: {e}",
            )

        # Classify the document type
        classification_results : List[DocumentImageClassifierOutput] = [classifier.classify(document_image) for classifier in self.classifiers]
        document_type = max(set(classification_results), key=classification_results.count)

        if document_type.details != ClassifierStatusCodes.OK:
            return DocumentImagePipelineOutput(
                status=StatusCodes.ERROR,
                details=f"Error in classifying document type: {document_type.details}",
            )
        
        # Find the appropriate parser
        parser = next(
            (parser for parser in self.parsers if parser.target_document_type == document_type),
            None
        )
        if not parser:
            return DocumentImagePipelineOutput(
                status=StatusCodes.ERROR,
                details=f"No parser found for document type: {document_type}",
            )

        
        # Parse the document
        parser_result : DocumentImageParserOutput = parser.parse(document_image)
        if parser_result.status != ParserStatusCodes.OK:
            return DocumentImagePipelineOutput(
                status= StatusCodes.ERROR,
                details=f"Error in parsing document: {parser_result.details}",
            )
            
        return DocumentImagePipelineOutput(
            status=StatusCodes.FINISHED_SUCCESS,
            details=parser_result,
        )
        
        
    @abstractmethod
    def validate_classifier_and_parser_document_types(self) -> (bool, List[str]):
        """
        Validates that all document types returned by classifiers are supported by parsers.

        Returns:
            tuple:
                - bool: True if all document types returned by classifiers are supported by parsers, False otherwise.
                - List[str]: A list of missing document types (if any).
        """
        # Get supported document types from parsers
        supported_document_types = [parser.target_document_type for parser in self.parsers]

        # Collect all document types supported by classifiers
        classifier_document_types = [
            doc_type for classifier in self.classifiers for doc_type in classifier.supported_document_types
        ]

        # Identify missing document types
        missing_document_types = list(set(classifier_document_types) - set(supported_document_types))

        # Return a tuple indicating validity and missing document types
        return not bool(missing_document_types), missing_document_types

    def add_processor(self, processor: DocumentImageProcessor):
        if processor in self.processors:
            raise ValueError("Processor is already in the pipeline.")
        self.processors.append(processor)

    def add_processors(self, processors: List[DocumentImageProcessor]):
        for processor in processors:
            if processor in self.processors:
                raise ValueError(f"Processor {processor} is already in the pipeline.")
        self.processors.extend(processors)

    def add_classifier(self, classifier: DocumentImageClassifier):
        if classifier in self.classifiers:
            raise ValueError("Classifier is already in the pipeline.")
        
        if not classifier.supported_document_types:
            raise ValueError("Classifier must support at least one document type.")
        self.classifiers.append(classifier)

    def add_classifiers(self, classifiers: List[DocumentImageClassifier]):
        for classifier in classifiers:
            if classifier in self.classifiers:
                raise ValueError(f"Classifier {classifier} is already in the pipeline.")
            if not classifier.supported_document_types:
                raise ValueError(f"Classifier {classifier} must support at least one document type.")
        self.classifiers.extend(classifiers)

    def add_parser(self, parser: DocumentImageParser):
        if parser in self.parsers:
            raise ValueError("Parser is already in the pipeline.")
        if not parser.target_document_type:
            raise ValueError("Parser must have a target document type.")
        self.parsers.append(parser)

    def add_parsers(self, parsers: List[DocumentImageParser]):
        for parser in parsers:
            if parser in self.parsers:
                raise ValueError(f"Parser {parser} is already in the pipeline.")
            if not parser.target_document_type:
                raise ValueError(f"Parser {parser} must have a target document type.")
        self.parsers.extend(parsers)