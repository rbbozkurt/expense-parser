from PIL import Image
from typing import Dict, List, Optional, Union, Generic, TypeVar
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from utils import StrEnum
from abc import ABC, abstractmethod
class TaxBreakdown(BaseModel):
    """
    Represents a tax breakdown for a receipt.

    Attributes:
        rate (float): Tax rate percentage.
        amount (float): Tax amount.
    """
    rate: float
    amount: float

class Discount(BaseModel):
    """
    Represents a discount applied to a receipt.

    Attributes:
        amount (float): Discount amount.
        description (str): Description of the discount.
    """
    amount: float
    description: str

class ItemizedReceiptItem(BaseModel):
    """
    Represents an item in an itemized receipt.

    Attributes:
        item_name (str): Name of the item.
        quantity (float): Quantity of the item purchased.
        item_total_price (float): Total price for this item.
    """
    item_name: str
    quantity: float
    item_total_price: float

class ReceiptSchema(BaseModel):
    """
    Schema for the structured output of expense receipt parsing.

    Attributes:
        store_name (str): Name of the store.
        store_address (Optional[str]): Address of the store.
        store_registration_number (Optional[str]): Registration number of the store.
        country (str): Country code (ISO 3166-1 alpha-2).
        language (str): Language code (ISO 639-1).
        timestamp (int): UNIX timestamp of the receipt.
        receipt_number (str): Unique identifier for the receipt.
        items (Optional[List[ItemizedReceiptItem]]): List of items purchased.
        subtotal (Optional[float]): Subtotal amount of the receipt.
        discount (Optional[Discount]): Discount details if applicable.
        tax (Optional[List[TaxBreakdown]]): Tax breakdown if applicable.
        total (float): Total amount of the receipt.
        currency (str): Currency code (ISO 4217).
        payment_method (str): Payment method used.
        metadata (Optional[dict]): Additional metadata about the receipt.
    """
    store_name: str
    store_address: Optional[str] = None
    store_registration_number: Optional[str] = None
    country: str
    language: str
    timestamp: int
    receipt_number: str
    items: Optional[List[ItemizedReceiptItem]] = []
    subtotal: Optional[float] = None
    discount: Optional[Discount] = None
    tax: Optional[List[TaxBreakdown]] = []
    total: float
    currency: str
    payment_method: str
    metadata: Optional[dict] = None

class StatusCodes(StrEnum):
    """
    Enum for response status codes.

    Attributes:
        OK: Indicates successful parsing.
        INVALID_IMAGE: Indicates an invalid image format.
        NO_TEXT: Indicates no text detected in the image.
        UNSUPPORTED_FORMAT: Indicates the document format is unsupported.
        EXTRACTION_FAILED: Indicates failure to extract data from the document.
        UNKNOWN_ERROR: Indicates an unknown error occurred.
    """
    OK = "OK"
    INVALID_IMAGE = "INVALID_IMAGE"
    NO_TEXT = "NO_TEXT"
    NO_DOCUMENT = "NO_DOCUMENT"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    FAKE_DOCUMENT = "FAKE_DOCUMENT"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

def generate_error_messages(target_doc_type : str):
    return{
        StatusCodes.INVALID_IMAGE: "Invalid image format. Please provide an image in JPEG or PNG format.",
        StatusCodes.NO_TEXT: "No text detected in the image. Please provide an image with visible text.",
        StatusCodes.NO_DOCUMENT: f"No document detected in the image. Please provide an image containing a {target_doc_type}.",
        StatusCodes.UNSUPPORTED_FORMAT: f"The provided document format is not supported for parsing. Please provide an image of a(n) {target_doc_type}.",
        StatusCodes.EXTRACTION_FAILED: "Failed to extract data from the document. Please try again.",
        StatusCodes.FAKE_DOCUMENT: "Fake document detected. Please provide an image of a real document.",
        StatusCodes.UNKNOWN_ERROR: "An unknown error occurred. Please try again later.",
}

# Define a generic type variable for schema
SchemaType = TypeVar("SchemaType", bound=BaseModel)

class DocumentImageParserOutput(GenericModel, Generic[SchemaType]):
    """
    Unified response format for document parsing.

    Attributes:
        status (StatusCodes): The status of the parsing operation.
        details (Union[SchemaType, str]): Parsed document data if successful; error message otherwise.
    """
    status: StatusCodes
    details: Union[SchemaType, str] = Field(
        ..., description="Parsed document data if status is OK; error message otherwise."
    )

class DocumentImageParser(ABC, Generic[SchemaType]):
    """
    Base class for document image parsers.

    Attributes:
        target_document_type (str): The document type this parser handles.
    """
    
    def __init__(self, target_document_type: str):
        self.target_document_type = target_document_type
        
    @abstractmethod
    def parse(self, image: Union[Image.Image, str]) -> DocumentImageParserOutput[SchemaType]:
        """
        Parse the given document image to extract structured data.

        :param image: The input image (PIL.Image.Image).
        :return: A DocumentImageParserOutput containing the parsed data or an error message.
        """
        pass
