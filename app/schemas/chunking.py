"""
Pydantic Schemas for Chunking Service
======================================

This module defines the Pydantic models for standardizing the input and output
of the text chunking services.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""
    chunk_index: int = Field(..., description="Index of this chunk (0-indexed).")
    start_token: int = Field(..., description="Starting token position in original text.")
    end_token: int = Field(..., description="Ending token position in original text.")
    token_count: int = Field(..., description="Number of tokens in this chunk.")
    total_tokens: int = Field(..., description="Total tokens in the original text.")
    total_chunks: int = Field(..., description="Total number of chunks created.")
    document_index: Optional[int] = Field(None, description="Index of source document (for batch chunking).")


class Chunk(BaseModel):
    """A single text chunk."""
    id: str = Field(..., description="Unique identifier for the chunk.")
    text: str = Field(..., description="The chunk text content.")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata.")


class ChunkRequest(BaseModel):
    """Request schema for chunking a single text."""
    text: str = Field(
        ...,
        min_length=1,
        description="The text to chunk."
    )
    chunk_size: int = Field(
        default=512,
        ge=50,
        le=8192,
        description="Maximum tokens per chunk. Default: 512 (optimized for RAG)."
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=1000,
        description="Overlapping tokens between chunks. Default: 50 (~10% overlap)."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata to attach to all chunks."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a long document that needs to be chunked for RAG...",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "metadata": {"source": "document.pdf", "page": 1}
            }
        }


class ChunkBatchRequest(BaseModel):
    """Request schema for chunking multiple texts."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of texts to chunk."
    )
    chunk_size: int = Field(
        default=512,
        ge=50,
        le=8192,
        description="Maximum tokens per chunk."
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=1000,
        description="Overlapping tokens between chunks."
    )
    metadata_list: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Optional list of metadata (one per text)."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["First document text...", "Second document text..."],
                "chunk_size": 512,
                "chunk_overlap": 50,
                "metadata_list": [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
            }
        }


class ChunkResponse(BaseModel):
    """Response schema for chunking operations."""
    chunks: List[Chunk] = Field(..., description="List of text chunks.")
    total_chunks: int = Field(..., description="Total number of chunks created.")
    config: Dict[str, Any] = Field(..., description="Chunker configuration used.")

    class Config:
        json_schema_extra = {
            "example": {
                "chunks": [
                    {
                        "id": "chunk_abc123def456",
                        "text": "This is the first chunk...",
                        "metadata": {
                            "chunk_index": 0,
                            "start_token": 0,
                            "end_token": 512,
                            "token_count": 512,
                            "total_tokens": 1500,
                            "total_chunks": 3
                        }
                    }
                ],
                "total_chunks": 3,
                "config": {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "encoding_name": "cl100k_base"
                }
            }
        }


class TokenCountRequest(BaseModel):
    """Request schema for counting tokens."""
    text: str = Field(..., min_length=1, description="Text to count tokens for.")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is some text to count tokens for."
            }
        }


class TokenCountResponse(BaseModel):
    """Response schema for token counting."""
    token_count: int = Field(..., description="Number of tokens in the text.")
    text_length: int = Field(..., description="Character length of the text.")
    encoding: str = Field(..., description="Encoding used for tokenization.")

    class Config:
        json_schema_extra = {
            "example": {
                "token_count": 9,
                "text_length": 38,
                "encoding": "cl100k_base"
            }
        }
