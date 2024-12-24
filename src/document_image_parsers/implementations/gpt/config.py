from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from document_image_parsers.interfaces import StatusCodes, DocumentImageParserOutput, generate_error_messages, ReceiptSchema


class DocumentImageReceiptParserOutput(DocumentImageParserOutput[ReceiptSchema]):
    """
    Specialized response format for receipt parsing.
    """
    pass

# Initialize configuration
RECEIPT_PARSER_CONFIG = {
    "model": "gpt-4o-2024-08-06",
    "target_document_type": "receipt",
    "response_format": DocumentImageReceiptParserOutput,
    "default_response": DocumentImageReceiptParserOutput(
        status=StatusCodes.UNKNOWN_ERROR,
        details="An unknown error occurred. Please try again later."
    )
}