"""Groq LLM Client
==============

This module provides a client for interacting with Groq's language models.
It supports both text and multi-modal inputs and is designed for use within
the application's LLM factory.
"""

import groq
import json
import logging
from typing import Any, Dict, List, Union

from app.core.config import settings

# Get a logger instance for this module
log = logging.getLogger(__name__)

class GroqClient:
    """
    A client for interacting with Groq's Large Language Models.

    This client manages communication with the Groq API, including authentication,
    handling different content types (text and multi-modal), and providing
    standardized output.
    """

    def __init__(self):
        """
        Initializes the GroqClient.

        Reads the API key from the application's settings. Raises a ValueError
        if the GROQ_API_KEY is not configured to ensure the client is always
        properly authenticated.
        """
        if not settings.GROQ_API_KEY:
            log.error("GROQ_API_KEY is not configured.")
            raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")
        
        log.info("Initializing GroqClient.")
        self.client = groq.Groq(api_key=settings.GROQ_API_KEY)

    def generate(self, model_id: str, prompt: Union[str, List[Dict[str, Any]]], temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generates a response from a specified Groq model.

        This method supports both simple text prompts and complex multi-modal
        prompts, sending the request to the Groq chat API and parsing the
        JSON response.

        Args:
            model_id: The ID of the Groq model to use (e.g., 'llama3-8b-8192').
            prompt: The user's prompt, which can be a simple string for text
                    or a list of dicts for multi-modal content following the
                    OpenAI message format.
            temperature: The sampling temperature for the generation.

        Returns:
            A dictionary containing the model's parsed JSON response.

        Raises:
            Exception: Propagates exceptions from the API or response parsing
                       after logging the error details.
        """
        log.info(f"Generating response from Groq model: {model_id}")
        try:
            # Construct the message payload based on the prompt type
            if isinstance(prompt, str):
                content = prompt
            elif isinstance(prompt, list):
                content = prompt
            else:
                raise TypeError("Prompt must be a string or a list of content blocks.")

            messages = [{"role": "user", "content": content}]

            # --- Diagnostic Logging ---
            # To debug the "No picture provided" error, we log the exact payload.
            # The data URL for the image is truncated to keep logs readable.
            try:
                debug_messages = json.loads(json.dumps(messages)) # Deep copy
                if isinstance(debug_messages[0]['content'], list):
                    for part in debug_messages[0]['content']:
                        if part.get('type') == 'image_url':
                            url = part['image_url']['url']
                            part['image_url']['url'] = f"{url[:50]}...[TRUNCATED]"
                log.debug(f"Payload to Groq API: {json.dumps(debug_messages, indent=2)}")
            except Exception as e:
                log.warning(f"Could not serialize debug payload: {e}")
            # --- End Diagnostic Logging ---

            completion = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            response_text = completion.choices[0].message.content
            log.debug(f"Received raw response from Groq: {response_text}")
            
            response_json = json.loads(response_text)
            
            return {"response": response_json}

        except groq.APIError as e:
            log.error(f"Groq API error for model {model_id}: {e}", exc_info=True)
            raise Exception(f"Groq API error: {e}") from e
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            log.error(f"Failed to parse Groq response for model {model_id}: {e}", exc_info=True)
            raise Exception(f"Failed to parse Groq response: {e}") from e
        except Exception as e:
            log.error(f"An unexpected error occurred with model {model_id}: {e}", exc_info=True)
            raise Exception(f"An unexpected error occurred: {e}") from e

# --- Example Usage ---
#
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#
#     try:
#         groq_client = GroqClient()
#         
#         # --- Text-only example ---
#         text_prompt = 'What are the top 3 benefits of using FastAPI? Respond in JSON.'
#         print("--- Testing Groq llama3-8b-8192 (text) ---")
#         response_text = groq_client.generate(
#             model_id='llama3-8b-8192',
#             prompt=text_prompt
#         )
#         print(response_text)
#         
#         # --- Multi-modal example (requires a model that supports it) ---
#         # Note: This requires providing a base64-encoded image.
#         # The following is a placeholder for the image data.
#         # image_base64 = "data:image/png;base64,iVBORw0KGgo..."
#
#         # multi_modal_prompt = [
#         #     {"type": "text", "text": "What is in this image? Respond in JSON."},
#         #     {"type": "image_url", "image_url": {"url": image_base64}},
#         # ]
#         # print("\n--- Testing Groq LLaMA-3.1-405B-i1 (multi-modal) ---")
#         # response_multi_modal = groq_client.generate(
#         #     model_id='llama-3.1-405b-i1', # Example model, check Groq for current multi-modal models
#         #     prompt=multi_modal_prompt
#         # )
#         # print(response_multi_modal)
#
#     except (ValueError, Exception) as e:
#         log.error(f"An error occurred during execution: {e}")