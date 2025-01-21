from PIL import Image
from typing import Union, List, Dict
from openai import OpenAI, OpenAIError
from tiktoken import encoding_for_model
from document_image_parsers.interfaces import (
    DocumentImageParser,
    ReceiptSchema,
    StatusCodes,
    DocumentImageParserOutput,
    generate_error_messages,
)
from document_image_parsers.implementations.gpt.config import (RECEIPT_PARSER_CONFIG, DocumentImageReceiptParserOutput)
from utils import encode_image_to_base64, handle_errors

class DocumentImageGPTReceiptParser(DocumentImageParser[ReceiptSchema]):
    """
    A parser that uses OpenAI's GPT API to extract structured data from expense receipts.
    """
    def __init__(self,
                openai_api_key: str, 
                model: str = RECEIPT_PARSER_CONFIG["model"],
                target_document_type: str = RECEIPT_PARSER_CONFIG["target_document_type"],
                response_format = RECEIPT_PARSER_CONFIG["response_format"],
                default_response = RECEIPT_PARSER_CONFIG["default_response"]):
        if not openai_api_key:
            raise ValueError("An OpenAI API key must be provided.")

        super().__init__(target_document_type=target_document_type)
        self.model = model
        self.response_format = response_format
        self.default_response = default_response
        self._openai_api_key = openai_api_key
        self._client = OpenAI(api_key=openai_api_key)
        self._error_messages = generate_error_messages(target_document_type)
        self._developer_system_prompt = self._generate_developer_system_prompt()
        self._assistant_prompts = self._generate_assistant_prompts()

        # Initialize token counters
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        
    @handle_errors(default_value=RECEIPT_PARSER_CONFIG["default_response"])
    def parse(self, image: Union[Image.Image, str]) -> DocumentImageReceiptParserOutput:
        """
        Parse the given receipt document to extract structured data.
        """
        try:
            base64_image = encode_image_to_base64(image)
        except ValueError as e:
            return DocumentImageReceiptParserOutput(
                status=StatusCodes.INVALID_IMAGE,
                details=f"Error encoding image: {e}"
            )

        messages = self._generate_user_prompt(base64_image)
        input_tokens = self._count_tokens(messages)
        self._last_input_tokens = input_tokens
        self._total_input_tokens += input_tokens

        try:
            response = self._client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=self.response_format,
            )
        except OpenAIError as e:
            return DocumentImageReceiptParserOutput(
                status=StatusCodes.UNKNOWN_ERROR,
                details=f"Error from OpenAI API: {e}"
            )

        self._total_output_tokens += response['usage']['completion_tokens']

        try:
            return DocumentImageReceiptParserOutput(**response["data"])
        except KeyError as e:
            return DocumentImageReceiptParserOutput(
                status=StatusCodes.EXTRACTION_FAILED,
                details=f"Missing required field: {e}"
            )

    def _generate_developer_system_prompt(self) -> str:
        """
        Generate the system prompt for the parser.
        """
        status_code_descriptions = "\n".join(
            f"- {code}: {description}" for code, description in self._error_messages.items()
        )
        return (
            f"You are a receipt parser. Parse receipts into structured JSON.\n\n"
            f"Target Document Type: {self.target_document_type}\n\n"
            f"Error Codes:\n{status_code_descriptions}\n"
        )

    def _generate_assistant_prompts(self) -> List[Dict[str, str]]:
        """
        Generate assistant examples for supported scenarios.
        """
        return [
            {
                "role": "assistant",
                "content": {
                    "status": StatusCodes.OK,
                    "details": {
                        "store_name": "Example Store",
                        "total": 100.0,
                        "currency": "USD"
                    }
                }
            },
            *[
                {
                    "role": "assistant",
                    "content": {"status": code, "details": description}
                } for code, description in self._error_messages.items()
            ]
        ]

    def _generate_user_prompt(self, base64_image: str) -> List[Dict]:
        """
        Generate a structured user prompt for GPT input.
        """
        return [
            {"role": "developer", "content": self._developer_system_prompt},
            *self._assistant_prompts,
            {"role": "user", "content": f"Parse the following receipt: data:image/jpeg;base64,{base64_image}"}
        ]

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count the number of tokens in the GPT request messages.
        """
        encoding = encoding_for_model(self.model)
        return sum(len(encoding.encode(value)) for message in messages for value in message.values())

    def __str__(self) -> str:
        """
        Override the string representation of the parser.
        """
        return (
            f"DocumentImageGPTReceiptParser:\n"
            f"- Model: {self.model}\n"
            f"- Target Document Type: {self.target_document_type}\n"
            f"- Total Input Tokens: {self._total_input_tokens}\n"
            f"- Total Output Tokens: {self._total_output_tokens}\n"
            f"- Error Messages: {', '.join(self._error_messages.values())}\n"
        )

