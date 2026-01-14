"""
Chat API Routes
===============

This module defines the API routes for RAG-based chat functionalities.
These endpoints handle document ingestion and the multi-modal RAG pipeline
for document-grounded conversations.

Note: For testing individual core services (LLMs, embeddings, rerankers),
use the endpoints in routes_core_services.py instead.
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core_services.ingestion.document_ingestion import DocumentIngestionPipeline
from app.schemas.ingestion import (
    ChunkMetadataInfo,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DocumentStatsResponse,
    IngestDocumentResponse,
    ProcessingStats,
)

# --- Router and Logger Setup ---
router = APIRouter()
log = logging.getLogger(__name__)


# --- Document Ingestion Endpoints ---

@router.post(
    "/ingest",
    response_model=IngestDocumentResponse,
    summary="Ingest a PDF document",
    description="Upload and process a PDF document for RAG. Extracts text and images, "
                "chunks the content, generates embeddings, and stores vectors in Pinecone.",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    namespace: str = Form(default="", description="Pinecone namespace for vectors"),
    extract_images: bool = Form(default=True, description="Extract images from PDF"),
    chunk_size: int = Form(default=312, ge=50, le=8192, description="Max tokens per chunk"),
    chunk_overlap: int = Form(default=50, ge=0, le=1000, description="Overlap tokens between chunks"),
):
    """
    Ingest a PDF document into the RAG system.

    This endpoint:
    1. Accepts a PDF file upload
    2. Parses the PDF to extract text (markdown) and images
    3. Chunks the text with configurable size and overlap
    4. Generates embeddings using Cohere embed-v4.0
    5. Upserts vectors to Pinecone with comprehensive metadata

    The metadata stored with each vector includes:
    - id: filename_chunk_XXXX (e.g., research_paper_chunk_0001)
    - text: The actual chunk text content
    - page_number: Estimated page number
    - filename: Original filename
    - file_path: Path to stored PDF
    - document_id: Unique document identifier
    - chunk_index: Position of chunk in document
    - token_count: Number of tokens in chunk
    """
    log.info(f"Received document ingestion request: {file.filename}")

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    try:
        # Read file content
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Initialize pipeline with requested settings
        pipeline = DocumentIngestionPipeline(
            storage_base_dir="storage",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=50,
        )

        # Run ingestion
        result = pipeline.ingest_pdf(
            file_source=file_content,
            filename=file.filename,
            namespace=namespace,
            extract_images=extract_images,
        )

        # Build response
        response = IngestDocumentResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            pdf_path=result["pdf_path"],
            image_dir=result.get("image_dir"),
            total_chunks=result["total_chunks"],
            total_vectors=result["total_vectors"],
            total_images=result["total_images"],
            namespace=result["namespace"],
            processing_stats=ProcessingStats(**result["processing_stats"]),
            chunks_metadata=[
                ChunkMetadataInfo(**chunk) for chunk in result["chunks_metadata"][:50]
            ],  # Limit to first 50 chunks in response
        )

        log.info(f"Document ingestion successful: {result['document_id']}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Document ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {str(e)}"
        )


@router.delete(
    "/document",
    response_model=DeleteDocumentResponse,
    summary="Delete a document from vector store",
    description="Delete all vectors associated with a document from Pinecone.",
)
async def delete_document(request: DeleteDocumentRequest):
    """
    Delete all vectors for a document from Pinecone.

    This removes all chunked vectors associated with the given document_id.
    Note: This does not delete the stored PDF or images from local storage.
    """
    log.info(f"Received document deletion request: {request.document_id}")

    try:
        pipeline = DocumentIngestionPipeline()

        result = pipeline.delete_document(
            document_id=request.document_id,
            namespace=request.namespace,
        )

        return DeleteDocumentResponse(
            document_id=result["document_id"],
            namespace=result["namespace"],
            status=result["status"],
        )

    except Exception as e:
        log.error(f"Document deletion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {str(e)}"
        )


@router.get(
    "/document/{document_id}/stats",
    response_model=DocumentStatsResponse,
    summary="Get document statistics",
    description="Get vector store statistics for a document.",
)
async def get_document_stats(
    document_id: str,
    namespace: str = "",
):
    """
    Get statistics for an ingested document.

    Returns index statistics from Pinecone for the given namespace.
    """
    log.info(f"Received document stats request: {document_id}")

    try:
        pipeline = DocumentIngestionPipeline()

        result = pipeline.get_document_stats(
            document_id=document_id,
            namespace=namespace,
        )

        return DocumentStatsResponse(
            document_id=result["document_id"],
            namespace=result["namespace"],
            index_stats=result["index_stats"],
        )

    except Exception as e:
        log.error(f"Failed to get document stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document stats: {str(e)}"
        )


# --- RAG Chat Status ---

@router.get("/status")
async def chat_status():
    """
    Returns the status of the RAG chat service.
    """
    return {
        "status": "active",
        "message": "RAG document ingestion is available.",
        "available_endpoints": [
            "POST /api/v1/chat/ingest - Ingest PDF documents",
            "DELETE /api/v1/chat/document - Delete document vectors",
            "GET /api/v1/chat/document/{document_id}/stats - Get document stats",
        ],
        "core_services": [
            "POST /api/v1/llm/generate - LLM text generation",
            "POST /api/v1/llm/generate-with-files - LLM with images",
            "POST /api/v1/embeddings/text - Text embeddings",
            "POST /api/v1/embeddings/image - Image embeddings",
            "POST /api/v1/reranker - Reranking",
        ]
    }
