from typing import List, Union, Dict
from PIL import Image
from openai import OpenAI, OpenAIError
from document_image_classifiers.interfaces import (
    DocumentImageClassifier,
    generate_error_messages,
    ClassifierSchema,
    StatusCodes,
)
from document_image_classifiers.implementations.gpt.config import (
    DOCUMENT_IMAGE_GPT_CLASSIFIER_CONFIG,
    DocumentImageGPTClassifierOutput,
)
from utils import encode_image_to_base64, handle_errors
from tiktoken import encoding_for_model


class DocumentImageGPTClassifier(DocumentImageClassifier[ClassifierSchema]):
    """
    GPT-based classifier for document images.
    """

    def __init__(self,
                openai_api_key: str,
                model: str = DOCUMENT_IMAGE_GPT_CLASSIFIER_CONFIG["model"],
                supported_document_types: List[str] = DOCUMENT_IMAGE_GPT_CLASSIFIER_CONFIG["supported_document_types"],
                response_format: type = DOCUMENT_IMAGE_GPT_CLASSIFIER_CONFIG["response_format"],
                default_response: DocumentImageGPTClassifierOutput = DOCUMENT_IMAGE_GPT_CLASSIFIER_CONFIG["default_response"]):
        
        if not openai_api_key:
            raise ValueError("An OpenAI API key must be provided.")
        
        super().__init__(supported_document_types= supported_document_types)
        self.model = model
        self.response_format = response_format
        self.default_response = default_response
        self._openai_api_key = openai_api_key
        self._client = OpenAI(api_key=openai_api_key)
        self._error_messages = generate_error_messages(self.supported_document_types)
        self._assistant_prompts = self._generate_assistant_prompts()
        self._developer_system_prompt = self._generate_developer_system_prompt()
        
        # Initialize token counters
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_input_tokens = 0
        self._last_output_tokens = 0

    @handle_errors(default_value=DOCUMENT_IMAGE_GPT_CLASSIFIER_CONFIG["default_response"])
    def classify(self, image: Union[Image.Image, str]) -> DocumentImageGPTClassifierOutput:
        """
        Classify the document type or return an error status with an explanation.
        """
        try:
            base64_image = encode_image_to_base64(image)
        except ValueError as e:
            return DocumentImageGPTClassifierOutput(
                status=StatusCodes.INVALID_IMAGE,
                details=f"Error in encoding image: {e}",
            )

        messages = self.user_prompt(base64_image)
        
        # Count input tokens
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
            return DocumentImageGPTClassifierOutput(
                status=StatusCodes.UNKNOWN_ERROR,
                details=f"Error in OpenAI API: {e}",
            )

        # Count output tokens
        output_tokens = response['usage']['completion_tokens']
        self._last_output_tokens = output_tokens
        self._total_output_tokens += output_tokens

        # Validate and construct the unified output
        try:
            validated_data = DocumentImageGPTClassifierOutput(**response["data"])
            return validated_data
        except KeyError as e:
            return DocumentImageGPTClassifierOutput(
                status=StatusCodes.EXTRACTION_FAILED,
                details=f"Missing required field: {str(e)}",
            )

    @property
    def summary(self) -> Dict[str, str]:
        """
        Summary of the GPT classifier's configuration.
        """
        return {
            "model": self.model,
            "supported_document_types": ", ".join(self.supported_document_types),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count the number of tokens in the input messages.
        """
        encoding = encoding_for_model(self.model)
        return sum(len(encoding.encode(value)) for message in messages for value in message.values())

    def _generate_assistant_prompts(self) -> List[DocumentImageGPTClassifierOutput]:
        """
        Generate assistant examples dynamically for supported document types and status codes.
        """
        examples = []
        for doc_type in self.supported_document_types:
            examples.append(
                DocumentImageGPTClassifierOutput(
                    status=StatusCodes.OK,
                    details=ClassifierSchema(document_type=doc_type)
                )
            )
        for status in StatusCodes:
            if status != StatusCodes.OK:
                examples.append(
                    DocumentImageGPTClassifierOutput(
                        status=status,
                        details=self._error_messages[status]
                    )
                )
        return examples

    def _generate_developer_system_prompt(self) -> str:
        """
        Generate the system prompt dynamically.
        """
        supported_types = ", ".join(self.supported_document_types)
        status_code_descriptions = "\n".join(
            f"- {status}: {desc}" for status, desc in self._error_messages.items()
        )

        return (
            f"You are a document type classifier using GPT. You will receive an image containing a document.\n\n"
            f"Supported Document Types: {supported_types}\n"
            f"Status Codes:\n{status_code_descriptions}\n"
            f"Return a JSON object with 'status' and 'details' fields."
        )

    def user_prompt(self, base64_image: str) -> List[Dict]:
        """
        Generate the structured messages for GPT input.
        """
        return [
            {"role": "developer", "content": self._developer_system_prompt},
            *[{"role": "assistant", "content": example} for example in self._assistant_prompts],
            {
                "role": "user",
                "content": (
                    f"Classify the type of this document: data:image/jpeg;base64,{base64_image}"
                ),
            },
        ]
