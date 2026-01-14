"""
Chat API Routes
===============

This module defines the API routes for RAG-based chat functionalities.
These endpoints handle document ingestion, retrieval, and the multi-modal RAG pipeline
for document-grounded conversations.

Note: For testing individual core services (LLMs, embeddings, rerankers),
use the endpoints in routes_core_services.py instead.
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core_services.ingestion.document_ingestion import DocumentIngestionPipeline
from app.retrieval.pipeline import RetrievalPipeline
from app.response_generation.text_generator import RAGTextGenerator
from app.response_generation.multimodal_generator import MultiModalRAGGenerator
from app.schemas.ingestion import (
    ChunkMetadataInfo,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DocumentStatsResponse,
    IngestDocumentResponse,
    ProcessingStats,
)
from app.schemas.rag import (
    AvailableModelsResponse,
    ContextUsed,
    MultiModalRAGAnswer,
    MultiModalRAGQueryRequest,
    MultiModalRAGQueryResponse,
    RAGAnswer,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedChunk,
)

# --- Router and Logger Setup ---
router = APIRouter()
log = logging.getLogger(__name__)


# --- RAG Query Endpoints ---

@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Query the RAG system",
    description="Ask a question and get an answer based on ingested documents. "
                "Uses retrieval, reranking, and LLM generation.",
)
async def rag_query(request: RAGQueryRequest):
    """
    Query the RAG system for an answer based on ingested documents.

    This endpoint:
    1. Embeds the user query using Cohere embeddings
    2. Retrieves top_k candidates from Pinecone
    3. Reranks to top_n using Cohere reranker
    4. Generates answer using the specified LLM
    5. Returns structured JSON response

    The answer includes:
    - Generated answer with confidence level
    - Source chunk IDs used
    - Retrieved chunks with scores
    - Token usage statistics
    """
    log.info(f"Received RAG query: '{request.query[:50]}...'")

    try:
        # Step 1 & 2 & 3: Retrieve and rerank
        retrieval_pipeline = RetrievalPipeline()
        retrieval_result = retrieval_pipeline.retrieve(
            query=request.query,
            namespace=request.namespace,
            top_k=request.top_k,
            top_n=request.top_n,
            filter=request.filter,
            use_reranker=request.use_reranker,
        )

        retrieved_chunks = retrieval_result["chunks"]

        if not retrieved_chunks:
            log.warning("No chunks retrieved. Returning empty response.")
            return RAGQueryResponse(
                query=request.query,
                answer=RAGAnswer(
                    answer=None,
                    confidence=None,
                    sources=[],
                    reason="No relevant documents found in the knowledge base.",
                ),
                raw_response="",
                model_id=request.model_id,
                retrieved_chunks=[],
                context_used=[],
                retrieval_stats={
                    "total_retrieved": 0,
                    "total_after_rerank": 0,
                    "namespace": request.namespace or "default",
                },
                usage=None,
            )

        # Step 4: Generate answer using LLM
        text_generator = RAGTextGenerator()
        generation_result = text_generator.generate(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            model_id=request.model_id,
            temperature=request.temperature,
        )

        # Build response
        response_chunks = []
        for chunk in retrieved_chunks:
            response_chunks.append(
                RetrievedChunk(
                    id=chunk["id"],
                    text=chunk["text"],
                    vector_score=chunk.get("vector_score") or chunk.get("score"),
                    rerank_score=chunk.get("rerank_score"),
                    rank=chunk.get("rank"),
                    metadata=chunk.get("metadata", {}),
                )
            )

        context_used = [
            ContextUsed(id=ctx["id"], text_preview=ctx["text_preview"])
            for ctx in generation_result["context_used"]
        ]

        # Parse answer from generation result
        answer_data = generation_result["answer"]
        rag_answer = RAGAnswer(
            answer=answer_data.get("answer"),
            confidence=answer_data.get("confidence"),
            sources=answer_data.get("sources", []),
            reason=answer_data.get("reason"),
        )

        response = RAGQueryResponse(
            query=request.query,
            answer=rag_answer,
            raw_response=generation_result["raw_response"],
            model_id=generation_result["model_id"],
            retrieved_chunks=response_chunks,
            context_used=context_used,
            retrieval_stats={
                "total_retrieved": retrieval_result["total_retrieved"],
                "total_after_rerank": retrieval_result["total_after_rerank"],
                "namespace": retrieval_result["namespace"],
            },
            usage=generation_result.get("usage"),
        )

        log.info(f"RAG query successful. Answer confidence: {rag_answer.confidence}")
        return response

    except Exception as e:
        log.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )


@router.post(
    "/query-multimodal",
    response_model=MultiModalRAGQueryResponse,
    summary="Multi-modal RAG query with text and images",
    description="Ask a question using both text context and images retrieved via vector search. "
                "Uses retrieval, text-only reranking, and vision-capable LLM (groq_maverick).",
)
async def multimodal_rag_query(request: MultiModalRAGQueryRequest):
    """
    Multi-modal RAG query that retrieves both text chunks and images via vector search.

    This endpoint:
    1. Embeds the user query using Cohere embeddings
    2. Searches Pinecone for top_k text chunks (filter: type="text")
    3. Searches Pinecone for top_k images (filter: type="image")
    4. Reranks text chunks only to get top_n
    5. Loads images from stored paths (from metadata)
    6. Sends text context + images to groq_maverick (vision LLM)
    7. Returns answer with image references (e.g., "As shown in Image 1...")

    Note: Images must be ingested during PDF processing with embed_images=True.
    The images are retrieved based on query similarity, not uploaded by users.
    """
    log.info(f"Received multi-modal RAG query: '{request.query[:50]}...'")

    try:
        # Step 1-4: Retrieve text chunks and images via vector search
        retrieval_pipeline = RetrievalPipeline()
        retrieval_result = retrieval_pipeline.retrieve_multimodal(
            query=request.query,
            namespace=request.namespace,
            text_top_k=request.text_top_k,
            text_top_n=request.text_top_n,
            image_top_k=request.image_top_k,
            use_reranker=request.use_reranker,
            filter=request.filter,
        )

        text_chunks = retrieval_result["text_chunks"]
        images = retrieval_result["images"]

        log.info(f"Retrieved {len(text_chunks)} text chunks and {len(images)} images")

        if not text_chunks and not images:
            log.warning("No chunks or images retrieved for multi-modal query.")
            return MultiModalRAGQueryResponse(
                query=request.query,
                answer=MultiModalRAGAnswer(
                    answer=None,
                    confidence=None,
                    sources=[],
                    images_referenced=[],
                    reason="No relevant documents or images found in the knowledge base.",
                ),
                raw_response="",
                model_id="groq_maverick",
                retrieved_chunks=[],
                context_used=[],
                images_used=[],
                retrieval_stats={
                    "total_text_retrieved": 0,
                    "total_text_after_rerank": 0,
                    "total_images_retrieved": 0,
                    "namespace": request.namespace or "default",
                },
                usage=None,
            )

        # Step 5-6: Generate multi-modal response
        mm_generator = MultiModalRAGGenerator()
        generation_result = mm_generator.generate(
            query=request.query,
            text_chunks=text_chunks,
            images=images,
            temperature=request.temperature,
        )

        # Build response
        response_chunks = []
        for chunk in text_chunks:
            response_chunks.append(
                RetrievedChunk(
                    id=chunk["id"],
                    text=chunk["text"],
                    vector_score=chunk.get("vector_score") or chunk.get("score"),
                    rerank_score=chunk.get("rerank_score"),
                    rank=chunk.get("rank"),
                    metadata=chunk.get("metadata", {}),
                )
            )

        context_used = [
            ContextUsed(id=ctx["id"], text_preview=ctx["text_preview"])
            for ctx in generation_result["context_used"]
        ]

        # Extract image paths used
        images_used = [
            img.get("image_path", "") for img in generation_result.get("images_used", [])
        ]

        # Parse answer
        answer_data = generation_result["answer"]
        mm_answer = MultiModalRAGAnswer(
            answer=answer_data.get("answer"),
            confidence=answer_data.get("confidence"),
            sources=answer_data.get("sources", []),
            images_referenced=answer_data.get("images_referenced", []),
            reason=answer_data.get("reason"),
        )

        response = MultiModalRAGQueryResponse(
            query=request.query,
            answer=mm_answer,
            raw_response=generation_result["raw_response"],
            model_id=generation_result["model_id"],
            retrieved_chunks=response_chunks,
            context_used=context_used,
            images_used=images_used,
            retrieval_stats={
                "total_text_retrieved": retrieval_result["total_text_retrieved"],
                "total_text_after_rerank": retrieval_result["total_text_after_rerank"],
                "total_images_retrieved": retrieval_result["total_images_retrieved"],
                "namespace": retrieval_result["namespace"],
            },
            usage=generation_result.get("usage"),
        )

        log.info(
            f"Multi-modal RAG query successful. "
            f"Text chunks: {len(text_chunks)}, Images: {len(images)}, "
            f"Images referenced: {mm_answer.images_referenced}"
        )
        return response

    except Exception as e:
        log.error(f"Multi-modal RAG query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Multi-modal RAG query failed: {str(e)}"
        )


@router.get(
    "/models",
    response_model=AvailableModelsResponse,
    summary="Get available LLM models",
    description="List all available LLM models for RAG generation.",
)
async def get_available_models():
    """
    Get list of available LLM models for RAG.
    """
    try:
        generator = RAGTextGenerator()
        models = generator.get_available_models()

        return AvailableModelsResponse(
            models=models,
            default_model="groq_maverick",
        )

    except Exception as e:
        log.error(f"Failed to get models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get models: {str(e)}"
        )


# --- Document Ingestion Endpoints ---

@router.post(
    "/ingest",
    response_model=IngestDocumentResponse,
    summary="Ingest a PDF document",
    description="Upload and process a PDF document for RAG. Extracts text and images, "
                "chunks the content, generates embeddings for both text and images, "
                "and stores vectors in Pinecone with type metadata (text/image).",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    namespace: str = Form(default="", description="Pinecone namespace for vectors"),
    extract_images: bool = Form(default=True, description="Extract images from PDF"),
    embed_images: bool = Form(default=True, description="Embed images and store in vector DB (for multi-modal RAG)"),
    chunk_size: int = Form(default=312, ge=50, le=8192, description="Max tokens per chunk"),
    chunk_overlap: int = Form(default=50, ge=0, le=1000, description="Overlap tokens between chunks"),
):
    """
    Ingest a PDF document into the RAG system.

    This endpoint:
    1. Accepts a PDF file upload
    2. Parses the PDF to extract text (markdown) and images
    3. Chunks the text with configurable size and overlap
    4. Generates text embeddings using Cohere embed-v4.0
    5. Generates image embeddings using Cohere embed-v4.0 (if embed_images=True)
    6. Upserts all vectors to Pinecone with type metadata (text/image)

    Text vector metadata:
    - type: "text"
    - id: filename_chunk_XXXX (e.g., research_paper_chunk_0001)
    - text: The actual chunk text content
    - page_number: Estimated page number
    - filename: Original filename
    - document_id: Unique document identifier

    Image vector metadata:
    - type: "image"
    - id: filename_img_XXXX (e.g., research_paper_img_0001)
    - image_path: Path to stored image file
    - page_number: Page where image was found
    - width, height: Image dimensions
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
            embed_images=embed_images,
        )

        # Build response
        response = IngestDocumentResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            pdf_path=result["pdf_path"],
            image_dir=result.get("image_dir"),
            total_chunks=result["total_chunks"],
            total_text_vectors=result.get("total_text_vectors", 0),
            total_image_vectors=result.get("total_image_vectors", 0),
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
        "message": "RAG system is fully operational (Text + Multi-Modal).",
        "available_endpoints": {
            "query": "POST /api/v1/chat/query - Text RAG query",
            "query_multimodal": "POST /api/v1/chat/query-multimodal - Multi-modal RAG with text + images",
            "models": "GET /api/v1/chat/models - List available LLM models",
            "ingest": "POST /api/v1/chat/ingest - Ingest PDF documents (text + image embeddings)",
            "delete": "DELETE /api/v1/chat/document - Delete document vectors",
            "stats": "GET /api/v1/chat/document/{document_id}/stats - Get document stats",
        },
        "pipeline_flow": {
            "text_rag": "Question → Embed → Pinecone (top 10 text) → Rerank (top 3) → LLM → Answer",
            "multimodal_rag": "Question → Embed → Pinecone (top 10 text + top 5 images) → Rerank text (top 3) → Vision LLM → Answer with image references",
        },
        "ingestion_flow": "PDF → Parse (text + images) → Chunk text → Embed (text + images) → Store in Pinecone (type: text/image)",
    }
