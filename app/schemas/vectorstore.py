"""
Pydantic Schemas for Vector Store Service
==========================================

This module defines the Pydantic models for standardizing the input and output
of the Pinecone vector store operations.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class VectorData(BaseModel):
    """A single vector with its ID, values, and optional metadata."""
    id: str = Field(..., description="Unique identifier for the vector.")
    values: List[float] = Field(..., description="The vector embedding values.")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata associated with the vector."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "values": [0.1, 0.2, 0.3, 0.4, 0.5],
                "metadata": {"source": "document.pdf", "page": 1, "chunk_index": 0}
            }
        }


class UpsertRequest(BaseModel):
    """Request schema for upserting vectors."""
    vectors: List[VectorData] = Field(
        ...,
        min_length=1,
        description="List of vectors to upsert."
    )
    namespace: str = Field(
        default="",
        description="The namespace to upsert vectors into. Empty string for default namespace."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "vectors": [
                    {"id": "vec_1", "values": [0.1, 0.2, 0.3], "metadata": {"type": "text"}},
                    {"id": "vec_2", "values": [0.4, 0.5, 0.6], "metadata": {"type": "image"}}
                ],
                "namespace": "documents"
            }
        }


class UpsertResponse(BaseModel):
    """Response schema for upsert operations."""
    upserted_count: int = Field(..., description="Number of vectors successfully upserted.")
    namespace: str = Field(..., description="The namespace vectors were upserted to.")

    class Config:
        json_schema_extra = {
            "example": {
                "upserted_count": 2,
                "namespace": "documents"
            }
        }


class QueryRequest(BaseModel):
    """Request schema for querying vectors."""
    vector: List[float] = Field(
        ...,
        min_length=1,
        description="The query vector (dense embedding)."
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=10000,
        description="Number of top results to return."
    )
    namespace: str = Field(
        default="",
        description="The namespace to query. Empty string for default namespace."
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in results."
    )
    include_values: bool = Field(
        default=False,
        description="Whether to include vector values in results."
    )
    filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata filter for the query."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
                "top_k": 5,
                "namespace": "documents",
                "include_metadata": True,
                "include_values": False,
                "filter": {"type": {"$eq": "text"}}
            }
        }


class QueryMatch(BaseModel):
    """A single match from a query result."""
    id: str = Field(..., description="The ID of the matched vector.")
    score: float = Field(..., description="Similarity score (higher is more similar).")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata of the matched vector (if requested)."
    )
    values: Optional[List[float]] = Field(
        None,
        description="Vector values (if requested)."
    )


class QueryResponse(BaseModel):
    """Response schema for query operations."""
    matches: List[QueryMatch] = Field(..., description="List of matching vectors.")
    namespace: str = Field(..., description="The namespace that was queried.")
    top_k: int = Field(..., description="The requested number of results.")
    total_matches: int = Field(..., description="Actual number of matches returned.")

    class Config:
        json_schema_extra = {
            "example": {
                "matches": [
                    {"id": "vec_1", "score": 0.95, "metadata": {"type": "text", "page": 1}},
                    {"id": "vec_3", "score": 0.87, "metadata": {"type": "text", "page": 3}}
                ],
                "namespace": "documents",
                "top_k": 5,
                "total_matches": 2
            }
        }


class DeleteRequest(BaseModel):
    """Request schema for deleting vectors."""
    ids: Optional[List[str]] = Field(
        None,
        description="List of vector IDs to delete."
    )
    namespace: str = Field(
        default="",
        description="The namespace to delete from."
    )
    delete_all: bool = Field(
        default=False,
        description="If True, deletes all vectors in the namespace."
    )
    filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata filter for deletion."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ids": ["vec_1", "vec_2"],
                "namespace": "documents",
                "delete_all": False
            }
        }


class DeleteResponse(BaseModel):
    """Response schema for delete operations."""
    deleted_ids: Optional[List[str]] = Field(None, description="IDs that were deleted.")
    deleted_count: Optional[int] = Field(None, description="Number of vectors deleted.")
    deleted_all: bool = Field(default=False, description="Whether all vectors were deleted.")
    namespace: str = Field(..., description="The namespace deleted from.")

    class Config:
        json_schema_extra = {
            "example": {
                "deleted_ids": ["vec_1", "vec_2"],
                "deleted_count": 2,
                "deleted_all": False,
                "namespace": "documents"
            }
        }


class IndexStats(BaseModel):
    """Statistics about the Pinecone index."""
    dimension: Optional[int] = Field(None, description="Vector dimension of the index.")
    total_vector_count: int = Field(..., description="Total number of vectors in the index.")
    namespaces: Dict[str, Dict[str, int]] = Field(
        ...,
        description="Vector counts per namespace."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dimension": 1536,
                "total_vector_count": 10000,
                "namespaces": {
                    "documents": {"vector_count": 5000},
                    "images": {"vector_count": 5000}
                }
            }
        }
