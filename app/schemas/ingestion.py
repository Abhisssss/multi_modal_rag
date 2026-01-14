"""
Pydantic Schemas for Document Ingestion
========================================

This module defines the Pydantic models for standardizing the input and output
of the document ingestion pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class ChunkMetadataInfo(BaseModel):
    """Metadata info for a single chunk."""
    vector_id: str = Field(..., description="Unique vector ID in Pinecone.")
    chunk_index: int = Field(..., description="Index of this chunk (0-indexed).")
    text_preview: str = Field(..., description="Preview of the chunk text (first 100 chars).")
    token_count: int = Field(..., description="Number of tokens in the chunk.")
    page_number: int = Field(..., description="Estimated page number of the chunk.")


class ProcessingStats(BaseModel):
    """Statistics from document processing."""
    markdown_length: int = Field(..., description="Length of extracted markdown text.")
    estimated_tokens: int = Field(..., description="Estimated total tokens in document.")
    total_blocks: int = Field(..., description="Total text blocks extracted.")
    text_batches_processed: int = Field(default=0, description="Number of text batches processed.")
    image_batches_processed: int = Field(default=0, description="Number of image batches processed.")


class IngestDocumentRequest(BaseModel):
    """Request schema for document ingestion (JSON body, used with file upload)."""
    namespace: str = Field(
        default="",
        description="Pinecone namespace for storing vectors. Empty string for default."
    )
    extract_images: bool = Field(
        default=True,
        description="Whether to extract images from the PDF."
    )
    chunk_size: int = Field(
        default=312,
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
    additional_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional metadata to include with each chunk."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "namespace": "documents",
                "extract_images": True,
                "chunk_size": 312,
                "chunk_overlap": 50,
                "additional_metadata": {"category": "research", "author": "John Doe"}
            }
        }


class IngestDocumentResponse(BaseModel):
    """Response schema for document ingestion."""
    document_id: str = Field(..., description="Unique identifier for the ingested document.")
    filename: str = Field(..., description="Original filename of the document.")
    pdf_path: str = Field(..., description="Path to the stored PDF file.")
    image_dir: Optional[str] = Field(None, description="Path to extracted images directory.")
    total_chunks: int = Field(..., description="Total number of text chunks created.")
    total_text_vectors: int = Field(default=0, description="Total text vectors upserted to Pinecone.")
    total_image_vectors: int = Field(default=0, description="Total image vectors upserted to Pinecone.")
    total_vectors: int = Field(..., description="Total number of vectors upserted to Pinecone (text + image).")
    total_images: int = Field(..., description="Total number of images extracted from PDF.")
    namespace: str = Field(..., description="Pinecone namespace where vectors are stored.")
    processing_stats: ProcessingStats = Field(..., description="Document processing statistics.")
    chunks_metadata: List[ChunkMetadataInfo] = Field(
        ...,
        description="Metadata for each chunk (truncated for large documents)."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "research_paper.pdf",
                "pdf_path": "storage/pdfs/research_paper.pdf",
                "image_dir": "storage/images/550e8400-e29b-41d4-a716-446655440000",
                "total_chunks": 25,
                "total_text_vectors": 25,
                "total_image_vectors": 5,
                "total_vectors": 30,
                "total_images": 5,
                "namespace": "documents",
                "processing_stats": {
                    "markdown_length": 15000,
                    "estimated_tokens": 3750,
                    "total_blocks": 45,
                    "text_batches_processed": 1,
                    "image_batches_processed": 1
                },
                "chunks_metadata": [
                    {
                        "vector_id": "research_paper_chunk_0000",
                        "chunk_index": 0,
                        "text_preview": "This is the beginning of the document...",
                        "token_count": 312,
                        "page_number": 1
                    }
                ]
            }
        }


class DeleteDocumentRequest(BaseModel):
    """Request schema for deleting a document from the vector store."""
    document_id: str = Field(..., description="Document ID to delete.")
    namespace: str = Field(
        default="",
        description="Pinecone namespace to delete from."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "namespace": "documents"
            }
        }


class DeleteDocumentResponse(BaseModel):
    """Response schema for document deletion."""
    document_id: str = Field(..., description="Document ID that was deleted.")
    namespace: str = Field(..., description="Namespace deleted from.")
    status: str = Field(..., description="Deletion status.")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "namespace": "documents",
                "status": "deleted"
            }
        }


class DocumentStatsResponse(BaseModel):
    """Response schema for document statistics."""
    document_id: str = Field(..., description="Document ID.")
    namespace: str = Field(..., description="Namespace queried.")
    index_stats: Dict[str, Any] = Field(..., description="Index statistics from Pinecone.")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "namespace": "documents",
                "index_stats": {
                    "dimension": 1024,
                    "total_vector_count": 100,
                    "namespaces": {
                        "documents": {"vector_count": 100}
                    }
                }
            }
        }
