"""
Pydantic Schemas for RAG Query
==============================

This module defines the Pydantic models for standardizing the input and output
of the RAG query pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with scores and metadata."""
    id: str = Field(..., description="Unique identifier of the chunk.")
    text: str = Field(..., description="The chunk text content.")
    vector_score: Optional[float] = Field(None, description="Pinecone similarity score.")
    rerank_score: Optional[float] = Field(None, description="Reranker relevance score.")
    rank: Optional[int] = Field(None, description="Final rank after reranking.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata.")


class ContextUsed(BaseModel):
    """Summary of context provided to LLM."""
    id: str = Field(..., description="Chunk ID.")
    text_preview: str = Field(..., description="Preview of chunk text (first 100 chars).")


class RAGAnswer(BaseModel):
    """Structured answer from RAG."""
    answer: Optional[str] = Field(None, description="The generated answer.")
    confidence: Optional[str] = Field(None, description="Confidence level: high, medium, low.")
    sources: List[str] = Field(default_factory=list, description="Chunk IDs used for answer.")
    reason: Optional[str] = Field(None, description="Reason if answer is null.")


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query."""
    query: str = Field(
        ...,
        min_length=1,
        description="The user's question or query."
    )
    namespace: str = Field(
        default="",
        description="Pinecone namespace to search. Empty string for default."
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of chunks to retrieve from Pinecone."
    )
    top_n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of chunks after reranking for context."
    )
    model_id: str = Field(
        default="groq_maverick",
        description="LLM model for answer generation."
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Generation temperature (lower = more factual)."
    )
    filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional Pinecone metadata filter."
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to apply reranking."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the main topic of the document?",
                "namespace": "documents",
                "top_k": 10,
                "top_n": 3,
                "model_id": "groq_maverick",
                "temperature": 0.3,
                "use_reranker": True
            }
        }


class MultiModalRAGQueryRequest(BaseModel):
    """Request schema for multi-modal RAG query (text + images via vector search)."""
    query: str = Field(
        ...,
        min_length=1,
        description="The user's question or query."
    )
    namespace: str = Field(
        default="",
        description="Pinecone namespace to search. Empty string for default."
    )
    text_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of text chunks to retrieve from Pinecone."
    )
    text_top_n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of text chunks after reranking for context."
    )
    image_top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of images to retrieve from Pinecone (max 5 sent to LLM)."
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Generation temperature (lower = more factual)."
    )
    filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional base Pinecone metadata filter (type filter added automatically)."
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to apply reranking on text chunks."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What does the architecture diagram show?",
                "namespace": "documents",
                "text_top_k": 10,
                "text_top_n": 3,
                "image_top_k": 5,
                "temperature": 0.3,
                "use_reranker": True
            }
        }


class RAGQueryResponse(BaseModel):
    """Response schema for RAG query."""
    query: str = Field(..., description="Original user query.")
    answer: RAGAnswer = Field(..., description="Structured answer from LLM.")
    raw_response: str = Field(..., description="Raw LLM response text.")
    model_id: str = Field(..., description="LLM model used for generation.")
    retrieved_chunks: List[RetrievedChunk] = Field(
        ...,
        description="Chunks retrieved and used for context."
    )
    context_used: List[ContextUsed] = Field(
        ...,
        description="Summary of context provided to LLM."
    )
    retrieval_stats: Dict[str, Any] = Field(
        ...,
        description="Statistics from retrieval pipeline."
    )
    usage: Optional[Dict[str, Any]] = Field(
        None,
        description="Token usage from LLM (if available)."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the main topic of the document?",
                "answer": {
                    "answer": "The main topic is machine learning applications.",
                    "confidence": "high",
                    "sources": ["doc_chunk_0001", "doc_chunk_0002"]
                },
                "raw_response": "{\"answer\": \"...\", \"confidence\": \"high\", ...}",
                "model_id": "groq_maverick",
                "retrieved_chunks": [
                    {
                        "id": "doc_chunk_0001",
                        "text": "Machine learning is a subset of AI...",
                        "vector_score": 0.92,
                        "rerank_score": 0.95,
                        "rank": 1,
                        "metadata": {"filename": "ml_paper.pdf", "page_number": 1}
                    }
                ],
                "context_used": [
                    {"id": "doc_chunk_0001", "text_preview": "Machine learning is a subset..."}
                ],
                "retrieval_stats": {
                    "total_retrieved": 10,
                    "total_after_rerank": 3,
                    "namespace": "documents"
                },
                "usage": {"prompt_tokens": 500, "completion_tokens": 100}
            }
        }


class AvailableModelsResponse(BaseModel):
    """Response schema for available models endpoint."""
    models: List[str] = Field(..., description="List of available model IDs.")
    default_model: str = Field(..., description="Default model for RAG.")

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    "groq_maverick",
                    "groq_llama3_8b",
                    "cohere_command_r_plus"
                ],
                "default_model": "groq_maverick"
            }
        }


# --- Multi-Modal RAG Schemas ---

class MultiModalRAGAnswer(BaseModel):
    """Structured answer from multi-modal RAG with image references."""
    answer: Optional[str] = Field(None, description="The generated answer with image references.")
    confidence: Optional[str] = Field(None, description="Confidence level: high, medium, low.")
    sources: List[str] = Field(default_factory=list, description="Chunk IDs used for answer.")
    images_referenced: List[str] = Field(
        default_factory=list,
        description="Images referenced in the answer (e.g., ['Image 1', 'Image 2'])."
    )
    reason: Optional[str] = Field(None, description="Reason if answer is null.")


class ImageInfo(BaseModel):
    """Information about an image used in multi-modal RAG."""
    index: int = Field(..., description="Image index (1-based).")
    path: str = Field(..., description="Path or identifier of the image.")
    filename: str = Field(..., description="Filename of the image.")


class MultiModalRAGQueryResponse(BaseModel):
    """Response schema for multi-modal RAG query."""
    query: str = Field(..., description="Original user query.")
    answer: MultiModalRAGAnswer = Field(..., description="Structured answer with image references.")
    raw_response: str = Field(..., description="Raw LLM response text.")
    model_id: str = Field(default="groq_maverick", description="LLM model used (always groq_maverick).")
    retrieved_chunks: List[RetrievedChunk] = Field(
        ...,
        description="Text chunks retrieved and used for context."
    )
    context_used: List[ContextUsed] = Field(
        ...,
        description="Summary of text context provided to LLM."
    )
    images_used: List[str] = Field(
        ...,
        description="List of image paths sent to the LLM."
    )
    retrieval_stats: Dict[str, Any] = Field(
        ...,
        description="Statistics from retrieval pipeline."
    )
    usage: Optional[Dict[str, Any]] = Field(
        None,
        description="Token usage from LLM (if available)."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What does the architecture diagram show?",
                "answer": {
                    "answer": "As shown in Image 1, the architecture consists of three layers...",
                    "confidence": "high",
                    "sources": ["doc_chunk_0001"],
                    "images_referenced": ["Image 1"]
                },
                "raw_response": "{\"answer\": \"As shown in Image 1...\", ...}",
                "model_id": "groq_maverick",
                "retrieved_chunks": [
                    {
                        "id": "doc_chunk_0001",
                        "text": "The system architecture includes...",
                        "vector_score": 0.92,
                        "rerank_score": 0.95,
                        "rank": 1,
                        "metadata": {"filename": "architecture.pdf", "page_number": 1}
                    }
                ],
                "context_used": [
                    {"id": "doc_chunk_0001", "text_preview": "The system architecture includes..."}
                ],
                "images_used": [
                    "storage/images/abc123/page_1_img_1.png"
                ],
                "retrieval_stats": {
                    "total_retrieved": 10,
                    "total_after_rerank": 3,
                    "namespace": "documents"
                },
                "usage": {"prompt_tokens": 800, "completion_tokens": 150}
            }
        }
