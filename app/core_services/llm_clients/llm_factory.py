"""
LLM Factory
===========

This module provides a factory for creating and using various Large Language Model (LLM) clients.
It abstracts the specific client implementations and provides a unified interface for text generation.
"""

import logging
from typing import Any, Dict, Protocol, Type, Tuple, List

from app.core_services.llm_clients.cohere_llm import CohereClient
from app.core_services.llm_clients.groq_llm import GroqClient
from app.schemas.llm import LLMRequest, LLMResponse
from app.utils.files import encode_image_to_data_url

log = logging.getLogger(__name__)

class LLMClient(Protocol):
    """
    A protocol defining the common interface for all LLM clients.
    """
    def generate(self, model_id: str, prompt: Any, temperature: float) -> Dict[str, Any]:
        ...

class LLMFactory:
    """
    A factory for accessing different LLM clients and models.
    """

    def __init__(self):
        """
        Initializes the LLMFactory and sets up the model-to-client mapping.
        """
        self._model_map: Dict[str, Tuple[Type[LLMClient], str]] = {
            # Cohere Models (Text only)
            "cohere_command_r_plus": (CohereClient, "command-r-plus-08-2024"),
            "cohere_command_a": (CohereClient, "command-a-03-2025"),

            # Groq Models (Text only)
            "groq_llama3_8b": (GroqClient, "llama3-8b-8192"),
            "groq_gemma_7b": (GroqClient, "gemma-7b-it"),
            "groq_mixtral_8x7b": (GroqClient, "mixtral-8x7b-32768"),
            
            # Groq Multi-modal model
            "groq_maverick": (GroqClient, "meta-llama/llama-4-maverick-17b-128e-instruct"),
        }
        log.info("LLMFactory initialized with %d models", len(self._model_map))

    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generates a response from the specified LLM using a standardized request schema.

        Args:
            request: An `LLMRequest` object containing all necessary information.

        Returns:
            An `LLMResponse` object with the model's output.

        Raises:
            ValueError: If the model ID is unsupported or if a multi-modal request
                        is sent to a text-only model.
        """
        log.info(f"Received generation request for model: {request.model_id}")

        if request.model_id not in self._model_map:
            log.error(f"Invalid model_id: {request.model_id}. Not found in model map.")
            raise ValueError(f"Model '{request.model_id}' is not supported.")

        client_class, specific_model_name = self._model_map[request.model_id]

        # --- Construct the prompt for the client ---
        prompt_for_client: Any
        if request.images:
            if request.model_id != "groq_maverick":
                raise ValueError(f"Model {request.model_id} does not support multi-modal input. Only 'groq_maverick' supports images.")
            
            # Build multi-modal prompt
            content_list = [{"type": "text", "text": request.user_prompt}]
            for image_path in request.images:
                try:
                    # Check if it's already a data URL or a local path
                    if image_path.startswith("data:image"):
                        data_url = image_path
                    else:
                        data_url = encode_image_to_data_url(image_path)
                    
                    content_list.append({"type": "image_url", "image_url": {"url": data_url}})
                except (FileNotFoundError, ValueError) as e:
                    log.error(f"Failed to process image {image_path}: {e}", exc_info=True)
                    raise
            prompt_for_client = content_list
        else:
            # For text-only prompts, the user_prompt is the content
            prompt_for_client = request.user_prompt
        
        try:
            log.debug(f"Instantiating client: {client_class.__name__}")
            client_instance = client_class()

            client_response = client_instance.generate(
                model_id=specific_model_name,
                prompt=prompt_for_client,
                temperature=request.temperature
            )

            # Wrap the client's dict response into the standard LLMResponse schema
            return LLMResponse(
                response=client_response.get("response", {}),
                model_id=request.model_id,
                # 'usage' would be extracted from client_response if available
                usage=client_response.get("usage")
            )
        except Exception as e:
            log.error(f"Failed to generate response using model {request.model_id}: {e}", exc_info=True)
            raise
