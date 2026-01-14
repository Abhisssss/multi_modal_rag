"""
Chat API Routes
===============

This module defines the API routes for RAG-based chat functionalities.
These endpoints will handle the actual multi-modal RAG pipeline for
document-grounded conversations.

Note: For testing individual core services (LLMs, embeddings, rerankers),
use the endpoints in routes_core_services.py instead.
"""

import logging

from fastapi import APIRouter

# --- Router and Logger Setup ---
router = APIRouter()
log = logging.getLogger(__name__)


# --- RAG Chat Endpoints ---
# TODO: Implement RAG chat endpoints here
# These will combine:
# - Document retrieval using embeddings
# - Reranking for relevance
# - LLM generation with retrieved context

@router.get("/chat/status")
async def chat_status():
    """
    Returns the status of the RAG chat service.

    This is a placeholder endpoint. RAG chat endpoints will be implemented here.
    """
    return {
        "status": "pending_implementation",
        "message": "RAG chat endpoints coming soon. Use /api/v1/llm/generate for direct LLM access.",
        "available_features": [
            "LLM generation: POST /api/v1/llm/generate",
            "LLM with files: POST /api/v1/llm/generate-with-files",
            "Text embeddings: POST /api/v1/embeddings/text",
            "Image embeddings: POST /api/v1/embeddings/image",
            "Reranking: POST /api/v1/reranker",
        ]
    }
