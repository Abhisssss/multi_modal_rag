"""
Core Services API Routes
========================

This module defines the API routes for testing core services including
LLMs, embeddings, reranking, and PDF parsing functionalities.
"""

import logging
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.core_services.embeddings.cohere_embeddings import CohereEmbeddings
from app.core_services.file_parsers.pdf_parser import PDFParser
from app.core_services.llm_clients.llm_factory import LLMFactory
from app.core_services.rerankers.cohere_reranker import CohereReranker
from app.core_services.storage.temp_file__store import TempFileStore
from app.core_services.vectorstores.pinecone_client import PineconeVectorStore
from app.schemas.embeddings import (
    TextEmbeddingRequest,
    EmbeddingResponse,
    TruncatedEmbedding,
)
from app.schemas.llm import LLMRequest, LLMResponse
from app.schemas.pdf import (
    PDFParseResponse,
    PDFTextResponse,
    PDFImagesResponse,
    PDFInfo,
    PDFStats,
    TextBlock,
    TextBlockMetadata,
    ExtractedImage,
    ImageExtractionResult,
)
from app.schemas.reranker import RerankerRequest, RerankerResponse
from app.schemas.vectorstore import (
    UpsertRequest,
    UpsertResponse,
    QueryRequest,
    QueryResponse,
    QueryMatch,
    DeleteRequest,
    DeleteResponse,
    IndexStats,
)
from app.utils.files import encode_image_to_data_url

# --- Router and Logger Setup ---
router = APIRouter()
log = logging.getLogger(__name__)

# --- Constants ---
TRUNCATE_SHOW_COUNT = 5  # Number of values to show at start and end


# --- Dynamic Model ID Enum for LLMs ---
try:
    llm_factory_for_enum = LLMFactory()
    ModelIdEnum = Enum(
        "ModelIdEnum",
        {key: key for key in llm_factory_for_enum._model_map.keys()}
    )
except Exception as e:
    log.error(f"Failed to create ModelIdEnum from LLMFactory: {e}", exc_info=True)
    ModelIdEnum = Enum("ModelIdEnum", {"fallback_model": "fallback_model"})


# --- Dependencies ---

def get_llm_factory():
    """Dependency to get an instance of LLMFactory."""
    return LLMFactory()


def get_embeddings_client():
    """Dependency to get an instance of CohereEmbeddings."""
    return CohereEmbeddings()


def get_reranker_client():
    """Dependency to get an instance of CohereReranker."""
    return CohereReranker()


def get_temp_file_store():
    """Dependency to get an instance of TempFileStore."""
    return TempFileStore()


def get_pdf_parser():
    """Dependency to get an instance of PDFParser."""
    return PDFParser()


def get_vectorstore():
    """Dependency to get an instance of PineconeVectorStore."""
    return PineconeVectorStore()


# --- Helper Functions ---

def truncate_embeddings(embeddings: list, show_count: int = TRUNCATE_SHOW_COUNT) -> list:
    """
    Truncates embedding vectors for display, showing first and last values.
    """
    truncated = []
    for embedding in embeddings:
        truncated.append(TruncatedEmbedding(
            first_values=embedding[:show_count],
            last_values=embedding[-show_count:],
            total_dimensions=len(embedding)
        ))
    return truncated


def truncate_markdown(markdown: str, max_chars: int = 2000) -> str:
    """
    Truncates markdown content for display, showing first and last portions.
    """
    if len(markdown) <= max_chars:
        return markdown

    half = max_chars // 2
    return f"{markdown[:half]}\n\n... [TRUNCATED - {len(markdown)} total characters] ...\n\n{markdown[-half:]}"


# --- LLM Endpoints ---

@router.post("/llm/generate", response_model=LLMResponse)
async def generate_llm_response(
    request: LLMRequest,
    llm_factory: LLMFactory = Depends(get_llm_factory),
):
    """
    Generates a response from a Large Language Model (JSON body).

    **Request Body:**
    - **user_prompt**: The main prompt or query from the user.
    - **model_id**: The model identifier (e.g., 'groq_llama3_8b', 'cohere_command_r_plus', 'groq_maverick').
    - **temperature**: Sampling temperature (0.0 - 2.0). Default: 0.7
    - **images**: Optional list of base64 encoded image data URLs (only for 'groq_maverick').
    """
    log.info(f"Received LLM generation request for model: {request.model_id}")

    try:
        response = llm_factory.generate(request)
        return response
    except ValueError as e:
        log.warning(f"Validation error during generation: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"An unexpected error occurred during generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while generating the response.")


@router.post("/llm/generate-with-files", response_model=LLMResponse)
async def generate_llm_response_with_files(
    user_prompt: str = Form(
        ...,
        description="The main prompt or query from the user.",
        example="What are the top 3 benefits of using FastAPI?",
    ),
    model_id: ModelIdEnum = Form(
        ...,
        description="The identifier for the desired LLM model."
    ),
    temperature: float = Form(
        0.7,
        ge=0.0,
        le=2.0,
        description="The sampling temperature for generation."
    ),
    images: Optional[List[UploadFile]] = File(
        None,
        description="Optional image file(s) for multi-modal generation. Only supported by 'groq_maverick' model."
    ),
    llm_factory: LLMFactory = Depends(get_llm_factory),
    temp_file_store: TempFileStore = Depends(get_temp_file_store),
):
    """
    Generates a response from a Large Language Model with file upload support.

    - **Model Selection**: Choose from available models via dropdown.
    - **Text Input**: Provide a user prompt.
    - **Image Upload**: Upload image files (only for 'groq_maverick' model).
    - **Temperature**: Adjust the creativity of the response.
    """
    log.info(f"Received LLM generation request (with files) for model: {model_id.value}")

    image_paths = []
    actual_images = [img for img in (images or []) if img.filename]

    if actual_images:
        if model_id.value != "groq_maverick":
            log.warning(f"Images provided but model {model_id.value} doesn't support multi-modal input. Ignoring images.")
        else:
            log.info(f"Handling {len(actual_images)} uploaded image(s).")
            try:
                image_paths = temp_file_store.save_multiple(actual_images)
            except IOError as e:
                log.error(f"File saving failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    llm_request = LLMRequest(
        user_prompt=user_prompt,
        model_id=model_id.value,
        images=image_paths if image_paths else None,
        temperature=temperature,
    )

    try:
        response = llm_factory.generate(llm_request)
        return response
    except ValueError as e:
        log.warning(f"Validation error during generation: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"An unexpected error occurred during generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while generating the response.")


# --- Embeddings Endpoints ---

@router.post("/embeddings/text", response_model=EmbeddingResponse)
async def generate_text_embeddings(
    request: TextEmbeddingRequest,
    include_full: bool = Query(
        False,
        description="If true, includes the full embedding vectors in the response."
    ),
    embeddings_client: CohereEmbeddings = Depends(get_embeddings_client),
):
    """
    Generates embeddings for text inputs.

    **Note:** Embedding vectors are truncated by default (showing first and last 5 values).
    Set `include_full=true` to get complete vectors.

    **Request Body:**
    - **texts**: List of text strings to embed.
    - **input_type**: Type of input ('search_document', 'search_query', 'classification').
    - **truncate**: How to handle long texts ('NONE', 'START', 'END').
    """
    log.info(f"Received text embedding request for {len(request.texts)} text(s).")

    try:
        result = embeddings_client.embed_texts(
            texts=request.texts,
            input_type=request.input_type.value,
            truncate=request.truncate.value
        )

        truncated = truncate_embeddings(result["embeddings"])

        return EmbeddingResponse(
            id=result.get("id"),
            embeddings_truncated=truncated,
            embeddings_full=result["embeddings"] if include_full else None,
            texts=result.get("texts"),
            meta=result.get("meta")
        )

    except ValueError as e:
        log.warning(f"Validation error during text embedding: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Error during text embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings/image", response_model=EmbeddingResponse)
async def generate_image_embeddings(
    images: List[UploadFile] = File(
        ...,
        description="Image file(s) to generate embeddings for."
    ),
    include_full: bool = Query(
        False,
        description="If true, includes the full embedding vectors in the response."
    ),
    embeddings_client: CohereEmbeddings = Depends(get_embeddings_client),
    temp_file_store: TempFileStore = Depends(get_temp_file_store),
):
    """
    Generates embeddings for uploaded image files.

    Upload one or more image files to generate vector embeddings using Cohere's embed-v4.0 model.

    **Note:** Embedding vectors are truncated by default (showing first and last 5 values).
    Set `include_full=true` to get complete vectors.
    """
    log.info(f"Received image embedding request for {len(images)} image(s).")

    # Filter out empty uploads
    actual_images = [img for img in images if img.filename]
    if not actual_images:
        raise HTTPException(status_code=400, detail="No valid image files provided.")

    try:
        # Save uploaded files temporarily
        image_paths = temp_file_store.save_multiple(actual_images)

        # Convert to base64 data URLs
        image_data_urls = []
        for path in image_paths:
            data_url = encode_image_to_data_url(path)
            image_data_urls.append(data_url)

        # Generate embeddings
        result = embeddings_client.embed_images(
            image_data_urls=image_data_urls,
            embedding_types=["float"]
        )

        truncated = truncate_embeddings(result["embeddings"])

        return EmbeddingResponse(
            id=result.get("id"),
            embeddings_truncated=truncated,
            embeddings_full=result["embeddings"] if include_full else None,
            images=result.get("images"),
            meta=result.get("meta")
        )

    except ValueError as e:
        log.warning(f"Validation error during image embedding: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Error during image embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Reranker Endpoints ---

@router.post("/reranker", response_model=RerankerResponse)
async def rerank_documents(
    request: RerankerRequest,
    reranker_client: CohereReranker = Depends(get_reranker_client),
):
    """
    Reranks documents based on their relevance to a query.

    **Request Body:**
    - **query**: The query string to rank documents against.
    - **documents**: List of document strings to rerank.
    - **top_n**: Number of top results to return (optional, defaults to all).
    """
    log.info(f"Received rerank request for {len(request.documents)} documents.")

    try:
        result = reranker_client.rerank(
            query=request.query,
            documents=request.documents,
            top_n=request.top_n
        )

        return RerankerResponse(
            id=result.get("id"),
            results=result["results"],
            query=result["query"],
            meta=result.get("meta")
        )

    except ValueError as e:
        log.warning(f"Validation error during reranking: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Error during reranking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- PDF Parser Endpoints ---

@router.post("/pdf/parse", response_model=PDFParseResponse)
async def parse_pdf_full(
    pdf_file: UploadFile = File(..., description="PDF file to parse."),
    extract_images: bool = Query(
        True,
        description="Whether to extract images from the PDF."
    ),
    min_image_width: int = Query(
        50,
        ge=1,
        description="Minimum image width to extract (filters out tiny images)."
    ),
    min_image_height: int = Query(
        50,
        ge=1,
        description="Minimum image height to extract."
    ),
    include_full_markdown: bool = Query(
        False,
        description="If true, includes full markdown in response (can be large)."
    ),
    include_all_blocks: bool = Query(
        False,
        description="If true, includes all text blocks (can be large)."
    ),
    pdf_parser: PDFParser = Depends(get_pdf_parser),
):
    """
    Performs full PDF parsing: saves the PDF, extracts text and images.

    This endpoint:
    - Saves the PDF to long-term storage (`storage/pdfs/`)
    - Extracts text as markdown (optimized for RAG)
    - Extracts structured text blocks with metadata
    - Extracts images and saves them (`storage/images/<document_id>/`)

    **Note:** By default, markdown and text blocks are truncated for display.
    Use query parameters to include full content.
    """
    log.info(f"Received PDF parse request: {pdf_file.filename}")

    if not pdf_file.filename or not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    try:
        # Read file content
        pdf_bytes = await pdf_file.read()

        # Parse the PDF
        result = pdf_parser.parse_full(
            file_source=pdf_bytes,
            filename=pdf_file.filename,
            extract_images=extract_images,
            min_image_width=min_image_width,
            min_image_height=min_image_height,
        )

        # Build response with optional truncation
        markdown_full = result["markdown"]
        markdown_truncated = truncate_markdown(markdown_full) if len(markdown_full) > 2000 else None

        # Convert text blocks to schema objects
        text_blocks = [
            TextBlock(
                text=block["text"],
                metadata=TextBlockMetadata(**block["metadata"])
            )
            for block in result["text_blocks"]
        ]

        # Truncate text blocks for display (first 10)
        text_blocks_truncated = text_blocks[:10] if len(text_blocks) > 10 and not include_all_blocks else None

        # Build images result if available
        images_result = None
        if result["images"]:
            images_result = ImageExtractionResult(
                document_id=result["images"]["document_id"],
                output_dir=result["images"]["output_dir"],
                images=[ExtractedImage(**img) for img in result["images"]["images"]],
                total_images=result["images"]["total_images"]
            )

        return PDFParseResponse(
            document_id=result["document_id"],
            pdf_info=PDFInfo(**result["pdf_info"]),
            markdown=markdown_full if include_full_markdown else (markdown_truncated or markdown_full[:2000]),
            markdown_truncated=markdown_truncated,
            text_blocks=text_blocks if include_all_blocks else text_blocks[:10],
            text_blocks_truncated=text_blocks_truncated,
            images=images_result,
            stats=PDFStats(**result["stats"])
        )

    except FileNotFoundError as e:
        log.error(f"File not found: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        log.warning(f"Validation error during PDF parsing: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Error during PDF parsing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")


@router.post("/pdf/extract-text", response_model=PDFTextResponse)
async def extract_pdf_text(
    pdf_file: UploadFile = File(..., description="PDF file to extract text from."),
    include_full: bool = Query(
        False,
        description="If true, includes full markdown (can be large)."
    ),
    pdf_parser: PDFParser = Depends(get_pdf_parser),
):
    """
    Extracts text from a PDF as markdown.

    This is a lighter endpoint that only extracts text without saving
    the PDF or extracting images. Useful for quick text extraction.
    """
    log.info(f"Received PDF text extraction request: {pdf_file.filename}")

    if not pdf_file.filename or not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    try:
        pdf_bytes = await pdf_file.read()
        markdown = pdf_parser.parse_to_markdown(pdf_bytes)

        return PDFTextResponse(
            document_id="temp-" + pdf_file.filename,
            markdown=markdown if include_full else truncate_markdown(markdown),
            markdown_truncated=truncate_markdown(markdown) if len(markdown) > 2000 else None,
            stats={
                "markdown_length": len(markdown),
                "estimated_tokens": len(markdown) // 4
            }
        )

    except Exception as e:
        log.error(f"Error during PDF text extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")


@router.post("/pdf/extract-images", response_model=PDFImagesResponse)
async def extract_pdf_images(
    pdf_file: UploadFile = File(..., description="PDF file to extract images from."),
    min_width: int = Query(50, ge=1, description="Minimum image width to extract."),
    min_height: int = Query(50, ge=1, description="Minimum image height to extract."),
    pdf_parser: PDFParser = Depends(get_pdf_parser),
):
    """
    Extracts images from a PDF and saves them to storage.

    Images are saved to `storage/images/<document_id>/` with filenames
    indicating page number and image index.
    """
    log.info(f"Received PDF image extraction request: {pdf_file.filename}")

    if not pdf_file.filename or not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    try:
        pdf_bytes = await pdf_file.read()
        result = pdf_parser.extract_images(
            pdf_bytes,
            min_width=min_width,
            min_height=min_height
        )

        return PDFImagesResponse(
            document_id=result["document_id"],
            output_dir=result["output_dir"],
            images=[ExtractedImage(**img) for img in result["images"]],
            total_images=result["total_images"]
        )

    except Exception as e:
        log.error(f"Error during PDF image extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract images: {str(e)}")


# --- Vector Store Endpoints ---

@router.post("/vectorstore/upsert", response_model=UpsertResponse)
async def upsert_vectors(
    request: UpsertRequest,
    vectorstore: PineconeVectorStore = Depends(get_vectorstore),
):
    """
    Upserts vectors into the Pinecone vector store.

    **Request Body:**
    - **vectors**: List of vectors with id, values, and optional metadata.
    - **namespace**: Namespace to upsert into (empty string for default).

    Each vector must have:
    - **id**: Unique identifier
    - **values**: List of float values (the embedding)
    - **metadata**: Optional dictionary of metadata
    """
    log.info(f"Received upsert request for {len(request.vectors)} vectors.")

    try:
        # Convert Pydantic models to dicts
        vectors_data = [
            {
                "id": v.id,
                "values": v.values,
                "metadata": v.metadata
            }
            for v in request.vectors
        ]

        result = vectorstore.upsert(
            vectors=vectors_data,
            namespace=request.namespace
        )

        return UpsertResponse(
            upserted_count=result["upserted_count"],
            namespace=result["namespace"]
        )

    except ValueError as e:
        log.warning(f"Validation error during upsert: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Error during upsert: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upsert vectors: {str(e)}")


@router.post("/vectorstore/query", response_model=QueryResponse)
async def query_vectors(
    request: QueryRequest,
    vectorstore: PineconeVectorStore = Depends(get_vectorstore),
):
    """
    Queries the vector store for similar vectors.

    **Request Body:**
    - **vector**: The query vector (dense embedding).
    - **top_k**: Number of top results to return.
    - **namespace**: Namespace to query (empty string for default).
    - **include_metadata**: Whether to include metadata in results.
    - **include_values**: Whether to include vector values in results.
    - **filter**: Optional metadata filter.

    Returns a list of matching vectors with their similarity scores.
    """
    log.info(f"Received query request with top_k={request.top_k}")

    try:
        result = vectorstore.query(
            vector=request.vector,
            top_k=request.top_k,
            namespace=request.namespace,
            include_metadata=request.include_metadata,
            include_values=request.include_values,
            filter=request.filter
        )

        matches = [
            QueryMatch(
                id=m["id"],
                score=m["score"],
                metadata=m.get("metadata"),
                values=m.get("values")
            )
            for m in result["matches"]
        ]

        return QueryResponse(
            matches=matches,
            namespace=result["namespace"],
            top_k=result["top_k"],
            total_matches=result["total_matches"]
        )

    except ValueError as e:
        log.warning(f"Validation error during query: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to query vectors: {str(e)}")


@router.post("/vectorstore/delete", response_model=DeleteResponse)
async def delete_vectors(
    request: DeleteRequest,
    vectorstore: PineconeVectorStore = Depends(get_vectorstore),
):
    """
    Deletes vectors from the vector store.

    **Request Body:**
    - **ids**: List of vector IDs to delete (optional).
    - **namespace**: Namespace to delete from.
    - **delete_all**: If True, deletes all vectors in the namespace.
    - **filter**: Optional metadata filter for deletion.

    Must specify either 'ids', 'delete_all=True', or 'filter'.
    """
    log.info(f"Received delete request for namespace: {request.namespace or 'default'}")

    try:
        result = vectorstore.delete(
            ids=request.ids,
            namespace=request.namespace,
            delete_all=request.delete_all,
            filter=request.filter
        )

        return DeleteResponse(
            deleted_ids=result.get("deleted_ids"),
            deleted_count=result.get("count"),
            deleted_all=result.get("deleted") == "all",
            namespace=result["namespace"]
        )

    except ValueError as e:
        log.warning(f"Validation error during delete: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Error during delete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {str(e)}")


@router.get("/vectorstore/stats", response_model=IndexStats)
async def get_index_stats(
    vectorstore: PineconeVectorStore = Depends(get_vectorstore),
):
    """
    Gets statistics about the Pinecone index.

    Returns dimension, total vector count, and per-namespace counts.
    """
    log.info("Received index stats request.")

    try:
        result = vectorstore.describe_index_stats()

        return IndexStats(
            dimension=result["dimension"],
            total_vector_count=result["total_vector_count"],
            namespaces=result["namespaces"]
        )

    except Exception as e:
        log.error(f"Error fetching index stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get index stats: {str(e)}")


# --- Health Check Endpoints ---

@router.get("/health/llm")
async def check_llm_health():
    """
    Health check for the LLM service.

    Returns the list of available models.
    """
    try:
        factory = LLMFactory()
        return {
            "status": "healthy",
            "service": "llm",
            "available_models": list(factory._model_map.keys())
        }
    except Exception as e:
        log.error(f"LLM health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"LLM service unhealthy: {str(e)}")


@router.get("/health/embeddings")
async def check_embeddings_health():
    """
    Health check for the embeddings service.
    """
    try:
        client = CohereEmbeddings()
        result = client.embed_texts(
            texts=["health check"],
            input_type="search_document"
        )
        return {
            "status": "healthy",
            "service": "embeddings",
            "model": client.model,
            "embedding_dimensions": len(result["embeddings"][0]) if result["embeddings"] else 0
        }
    except Exception as e:
        log.error(f"Embeddings health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Embeddings service unhealthy: {str(e)}")


@router.get("/health/reranker")
async def check_reranker_health():
    """
    Health check for the reranker service.
    """
    try:
        client = CohereReranker()
        result = client.rerank(
            query="test query",
            documents=["test document 1", "test document 2"],
            top_n=1
        )
        return {
            "status": "healthy",
            "service": "reranker",
            "model": client.model,
            "test_result": result["results"][0] if result["results"] else None
        }
    except Exception as e:
        log.error(f"Reranker health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Reranker service unhealthy: {str(e)}")


@router.get("/health/pdf-parser")
async def check_pdf_parser_health():
    """
    Health check for the PDF parser service.
    """
    try:
        parser = PDFParser()
        return {
            "status": "healthy",
            "service": "pdf-parser",
            "pdf_storage": str(parser.pdf_storage),
            "image_storage": str(parser.image_storage),
            "pdf_storage_exists": parser.pdf_storage.exists(),
            "image_storage_exists": parser.image_storage.exists(),
        }
    except Exception as e:
        log.error(f"PDF parser health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"PDF parser service unhealthy: {str(e)}")


@router.get("/health/vectorstore")
async def check_vectorstore_health():
    """
    Health check for the Pinecone vector store service.
    """
    try:
        client = PineconeVectorStore()
        stats = client.describe_index_stats()
        return {
            "status": "healthy",
            "service": "vectorstore",
            "index_name": client.index_name,
            "dimension": stats["dimension"],
            "total_vector_count": stats["total_vector_count"],
            "namespaces": list(stats["namespaces"].keys())
        }
    except Exception as e:
        log.error(f"Vectorstore health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Vectorstore service unhealthy: {str(e)}")
