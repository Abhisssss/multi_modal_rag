"""Cohere LLM Client
=================

This module provides a client for interacting with Cohere's language models
using the V2 API. It's designed for use within the application's LLM factory.
"""

import cohere
import json
import logging
from typing import Any, Dict, List

from app.core.config import settings

# Get a logger instance for this module
log = logging.getLogger(__name__)

class CohereClient:
    """
    A client for interacting with Cohere's V2 Large Language Models.

    This client handles communication with the Cohere API, including authentication,
    request formatting, and robust response parsing. It uses the `ClientV2`
    interface as specified.
    """

    def __init__(self):
        """
        Initializes the CohereClient.

        The API key is read from the application's settings. It will raise a
        ValueError if the COHERE_API_KEY is not configured, preventing the
        application from starting with an invalid state.
        """
        if not settings.COHERE_API_KEY:
            log.error("COHERE_API_KEY is not configured.")
            raise ValueError("COHERE_API_KEY is not set in the environment or .env file.")
        
        log.info("Initializing CohereClient.")
        self.client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)

    def generate(self, model_id: str, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Generates a response from a specified Cohere model.

        This method sends a prompt to the Cohere chat API and expects a JSON object
        in response, which is then parsed and returned in a standardized format.

        Args:
            model_id: The ID of the Cohere model to use (e.g., 'command-r-plus-08-2024').
            prompt: The user's input prompt to the model.
            temperature: The sampling temperature for the generation, controlling randomness.

        Returns:
            A dictionary containing the model's parsed JSON response.

        Raises:
            Exception: Propagates exceptions from the API call or response parsing
                       after logging the error details.
        """
        log.info(f"Generating response from Cohere model: {model_id}")
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            response = self.client.chat(
                model=model_id,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            
            response_text = response.message.content[0].text
            log.debug(f"Received raw response from Cohere: {response_text}")
            
            response_json = json.loads(response_text)
            
            return {"response": response_json}

        except cohere.ApiError as e:
            log.error(f"Cohere API error for model {model_id}: {e}", exc_info=True)
            raise Exception(f"Cohere API error: {e}") from e
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            log.error(f"Failed to parse Cohere response for model {model_id}: {e}", exc_info=True)
            raise Exception(f"Failed to parse Cohere response: {e}") from e
        except Exception as e:
            log.error(f"An unexpected error occurred with model {model_id}: {e}", exc_info=True)
            raise Exception(f"An unexpected error occurred: {e}") from e

# --- Example Usage ---
#
# if __name__ == '__main__':
#     # Configure basic logging for demonstration
#     logging.basicConfig(level=logging.INFO)
#
#     # In a real FastAPI app, you would have a .env file with the API key.
#     # For this example, you might need to set it manually if you run this script directly.
#     # from dotenv import load_dotenv
#     # load_dotenv()
#
#     try:
#         cohere_client = CohereClient()
#         
#         prompt = 'Say hello, in json only, like this : {"response": "hello"}'
#         
#         print("--- Testing Cohere command-r-plus-08-2024 ---")
#         response_r_plus = cohere_client.generate(
#             model_id='command-r-plus-08-2024',
#             prompt=prompt
#         )
#         print(response_r_plus)
#         
#         print("\n--- Testing Cohere command-a-03-2025 ---")
#         response_a = cohere_client.generate(
#             model_id='command-a-03-2025',
#             prompt=prompt,
#             temperature=0.0
#         )
#         print(response_a)
#
#     except (ValueError, Exception) as e:
#         log.error(f"An error occurred during execution: {e}")