"""Cohere Reranker Client
========================

This module provides a client for reranking documents using Cohere's rerank-v4.0-pro model.
"""

import cohere
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings

log = logging.getLogger(__name__)


class CohereReranker:
    """
    A client for reranking documents using Cohere's rerank model.

    This client takes a query and a list of documents, and returns the documents
    reranked by relevance to the query.
    """

    def __init__(self):
        """
        Initializes the CohereReranker client.

        Raises:
            ValueError: If COHERE_API_KEY is not configured.
        """
        if not settings.COHERE_API_KEY:
            log.error("COHERE_API_KEY is not configured.")
            raise ValueError("COHERE_API_KEY is not set in the environment or .env file.")

        log.info("Initializing CohereReranker client.")
        self.client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        self.model = "rerank-v4.0-pro"

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Reranks a list of documents based on their relevance to a query.

        Args:
            query: The query string to rank documents against.
            documents: A list of document strings to rerank.
            top_n: The number of top results to return. If None, returns all documents.

        Returns:
            A dictionary containing:
                - results: List of reranked results with index, score, and document text.
                - query: The original query.
                - meta: API metadata.

        Raises:
            Exception: If the API call fails.
        """
        log.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")

        if top_n is None:
            top_n = len(documents)

        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n
            )

            # Build structured results
            results = []
            for i, result in enumerate(response.results):
                results.append({
                    "rank": i + 1,
                    "index": result.index,
                    "relevance_score": result.relevance_score,
                    "document": documents[result.index]
                })

            return {
                "id": response.id if hasattr(response, 'id') else None,
                "results": results,
                "query": query,
                "meta": {
                    "api_version": response.meta.api_version if response.meta else None,
                    "billed_units": response.meta.billed_units if response.meta else None,
                }
            }

        except cohere.ApiError as e:
            log.error(f"Cohere API error during reranking: {e}", exc_info=True)
            raise Exception(f"Cohere API error: {e}") from e
        except Exception as e:
            log.error(f"Unexpected error during reranking: {e}", exc_info=True)
            raise Exception(f"Unexpected error: {e}") from e
