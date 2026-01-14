"""Cohere Embeddings Client
=========================

This module provides a client for generating embeddings using Cohere's embed-v4.0 model.
It supports both text and image embeddings.
"""

import cohere
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings

log = logging.getLogger(__name__)


class CohereEmbeddings:
    """
    A client for generating embeddings using Cohere's embed-v4.0 model.

    This client supports both text and image embeddings, providing a unified
    interface for generating vector representations.
    """

    def __init__(self):
        """
        Initializes the CohereEmbeddings client.

        Raises:
            ValueError: If COHERE_API_KEY is not configured.
        """
        if not settings.COHERE_API_KEY:
            log.error("COHERE_API_KEY is not configured.")
            raise ValueError("COHERE_API_KEY is not set in the environment or .env file.")

        log.info("Initializing CohereEmbeddings client.")
        self.client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        self.model = "embed-v4.0"

    def embed_texts(
        self,
        texts: List[str],
        input_type: str = "search_document",
        truncate: str = "NONE"
    ) -> Dict[str, Any]:
        """
        Generates embeddings for a list of text strings.

        Args:
            texts: A list of text strings to embed.
            input_type: The type of input - 'search_document', 'search_query', or 'classification'.
            truncate: How to handle texts that exceed the model's context length.
                      Options: 'NONE', 'START', 'END'.

        Returns:
            A dictionary containing the embeddings and metadata.

        Raises:
            Exception: If the API call fails.
        """
        log.info(f"Generating text embeddings for {len(texts)} text(s).")
        try:
            response = self.client.embed(
                model=self.model,
                texts=texts,
                input_type=input_type,
                truncate=truncate
            )

            # Extract embeddings - handle both list and object response formats
            embeddings = response.embeddings
            if hasattr(embeddings, 'float'):
                embeddings_list = embeddings.float
            else:
                embeddings_list = embeddings

            return {
                "id": response.id,
                "embeddings": embeddings_list,
                "texts": response.texts,
                "meta": {
                    "api_version": response.meta.api_version if response.meta else None,
                    "billed_units": response.meta.billed_units if response.meta else None,
                }
            }

        except cohere.ApiError as e:
            log.error(f"Cohere API error during text embedding: {e}", exc_info=True)
            raise Exception(f"Cohere API error: {e}") from e
        except Exception as e:
            log.error(f"Unexpected error during text embedding: {e}", exc_info=True)
            raise Exception(f"Unexpected error: {e}") from e

    def embed_images(
        self,
        image_data_urls: List[str],
        embedding_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generates embeddings for a list of images.

        Args:
            image_data_urls: A list of base64 encoded image data URLs
                            (e.g., "data:image/jpeg;base64,...").
            embedding_types: The types of embeddings to return. Default: ['float'].

        Returns:
            A dictionary containing the embeddings and metadata.

        Raises:
            Exception: If the API call fails.
        """
        log.info(f"Generating image embeddings for {len(image_data_urls)} image(s).")

        if embedding_types is None:
            embedding_types = ["float"]

        try:
            # Build image inputs in the format required by Cohere
            image_inputs = [
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
                for image_url in image_data_urls
            ]

            response = self.client.embed(
                model=self.model,
                input_type="image",
                embedding_types=embedding_types,
                inputs=image_inputs
            )

            # Extract embeddings
            embeddings = response.embeddings
            if hasattr(embeddings, 'float'):
                embeddings_list = embeddings.float
            else:
                embeddings_list = embeddings

            return {
                "id": response.id,
                "embeddings": embeddings_list,
                "images": response.images if hasattr(response, 'images') else [],
                "meta": {
                    "api_version": response.meta.api_version if response.meta else None,
                    "billed_units": response.meta.billed_units if response.meta else None,
                }
            }

        except cohere.ApiError as e:
            log.error(f"Cohere API error during image embedding: {e}", exc_info=True)
            raise Exception(f"Cohere API error: {e}") from e
        except Exception as e:
            log.error(f"Unexpected error during image embedding: {e}", exc_info=True)
            raise Exception(f"Unexpected error: {e}") from e
