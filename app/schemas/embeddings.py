"""
Pydantic Schemas for Embeddings Service
========================================

This module defines the Pydantic models for standardizing the input and output
of the embeddings generation services.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class EmbeddingInputType(str, Enum):
    """The type of input for text embeddings."""
    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"
    CLASSIFICATION = "classification"


class TruncateOption(str, Enum):
    """How to handle texts that exceed the model's context length."""
    NONE = "NONE"
    START = "START"
    END = "END"


class TextEmbeddingRequest(BaseModel):
    """Request schema for text embedding generation."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="A list of text strings to embed."
    )
    input_type: EmbeddingInputType = Field(
        default=EmbeddingInputType.SEARCH_DOCUMENT,
        description="The type of input - 'search_document', 'search_query', or 'classification'."
    )
    truncate: TruncateOption = Field(
        default=TruncateOption.NONE,
        description="How to handle texts that exceed the model's context length."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Hello world", "How are you?"],
                "input_type": "search_document",
                "truncate": "NONE"
            }
        }


class ImageEmbeddingRequest(BaseModel):
    """Request schema for image embedding generation."""
    images: List[str] = Field(
        ...,
        min_length=1,
        description="A list of base64 encoded image data URLs (e.g., 'data:image/jpeg;base64,...')."
    )
    embedding_types: Optional[List[str]] = Field(
        default=["float"],
        description="The types of embeddings to return. Default: ['float']."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "images": ["data:image/jpeg;base64,/9j/4AAQSkZJRg..."],
                "embedding_types": ["float"]
            }
        }


class TruncatedEmbedding(BaseModel):
    """A truncated embedding vector for display purposes."""
    first_values: List[float] = Field(
        ...,
        description="First few values of the embedding vector."
    )
    last_values: List[float] = Field(
        ...,
        description="Last few values of the embedding vector."
    )
    total_dimensions: int = Field(
        ...,
        description="Total number of dimensions in the full embedding."
    )


class EmbeddingResponse(BaseModel):
    """Response schema for embedding generation."""
    id: Optional[str] = Field(
        None,
        description="The unique ID of the embedding request."
    )
    embeddings_truncated: List[TruncatedEmbedding] = Field(
        ...,
        description="Truncated embeddings showing first and last values with total dimensions."
    )
    embeddings_full: Optional[List[List[float]]] = Field(
        None,
        description="Full embeddings (only included if include_full=true)."
    )
    texts: Optional[List[str]] = Field(
        None,
        description="The input texts (for text embeddings)."
    )
    images: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Image metadata (for image embeddings)."
    )
    meta: Optional[Dict[str, Any]] = Field(
        None,
        description="API metadata including version and billing info."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "da6e531f-54c6-4a73-bf92-f60566d8d753",
                "embeddings_truncated": [
                    {
                        "first_values": [0.016296387, -0.008354187, 0.023456789],
                        "last_values": [0.047332764, 0.0023212433, 0.0052719116],
                        "total_dimensions": 1024
                    }
                ],
                "texts": ["Hello world"],
                "meta": {
                    "api_version": {"version": "2"},
                    "billed_units": {"input_tokens": 2}
                }
            }
        }
