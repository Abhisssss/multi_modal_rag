"""Document Ingestion Pipeline
=============================

A complete pipeline for ingesting PDF documents into the RAG system.
Handles text extraction, chunking, embedding, and vector storage.
Supports multi-modal ingestion with both text and image embeddings.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.core_services.chunking.default_chunker import TokenChunker
from app.core_services.embeddings.cohere_embeddings import CohereEmbeddings
from app.core_services.file_parsers.pdf_parser import PDFParser
from app.core_services.vectorstores.pinecone_client import PineconeVectorStore
from app.utils.files import encode_image_to_data_url

log = logging.getLogger(__name__)

# Default batch size for embedding and upserting
DEFAULT_BATCH_SIZE = 50
IMAGE_BATCH_SIZE = 5  # Smaller batch for images due to size


class DocumentIngestionPipeline:
    """
    A pipeline for ingesting documents into the RAG system.

    This pipeline handles the complete workflow:
    1. PDF parsing (text and image extraction)
    2. Text chunking with overlap
    3. Batch embedding generation (text and images)
    4. Vector storage with metadata (type: text/image)
    """

    def __init__(
        self,
        storage_base_dir: str = "storage",
        chunk_size: int = 312,
        chunk_overlap: int = 50,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the DocumentIngestionPipeline.

        Args:
            storage_base_dir: Base directory for storing files.
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Overlapping tokens between chunks.
            batch_size: Number of items per batch for embedding/upserting.
        """
        self.storage_base_dir = storage_base_dir
        self.batch_size = batch_size

        # Initialize core services
        self.pdf_parser = PDFParser(
            storage_base_dir=storage_base_dir,
            pdf_storage_dir="pdfs",
            image_storage_dir="images",
        )
        self.chunker = TokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embeddings = CohereEmbeddings()
        self.vector_store = PineconeVectorStore()

        log.info(
            f"DocumentIngestionPipeline initialized: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}, batch_size={batch_size}"
        )

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for use in vector IDs."""
        name = Path(filename).stem
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.lower().strip("_")

    def _create_vector_id(self, filename: str, chunk_index: int, vector_type: str = "text") -> str:
        """Create a unique vector ID from filename, index, and type."""
        sanitized = self._sanitize_filename(filename)
        if vector_type == "image":
            return f"{sanitized}_img_{chunk_index:04d}"
        return f"{sanitized}_chunk_{chunk_index:04d}"

    def ingest_pdf(
        self,
        file_source: Union[str, Path, bytes],
        filename: str,
        namespace: str = "",
        extract_images: bool = True,
        embed_images: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a PDF document into the RAG system.

        This method performs the complete ingestion workflow:
        1. Parse PDF and extract text/images
        2. Chunk the extracted text
        3. Generate text embeddings in batches
        4. Generate image embeddings (if enabled)
        5. Upsert all vectors to Pinecone with metadata

        Args:
            file_source: PDF file as path or bytes.
            filename: Original filename of the PDF.
            namespace: Pinecone namespace for vectors.
            extract_images: Whether to extract images from PDF.
            embed_images: Whether to embed images and store in vector DB.
            additional_metadata: Optional extra metadata to include.

        Returns:
            Dict containing ingestion results and statistics.
        """
        log.info(f"Starting document ingestion for: {filename}")

        result = {
            "document_id": None,
            "filename": filename,
            "pdf_path": None,
            "image_dir": None,
            "total_chunks": 0,
            "total_text_vectors": 0,
            "total_image_vectors": 0,
            "total_vectors": 0,
            "total_images": 0,
            "namespace": namespace or "default",
            "chunks_metadata": [],
            "images_metadata": [],
            "processing_stats": {},
        }

        try:
            # Step 1: Parse PDF (extract text and images)
            log.info("Step 1: Parsing PDF...")
            pdf_result = self.pdf_parser.parse_full(
                file_source=file_source,
                filename=filename,
                extract_images=extract_images,
            )

            result["document_id"] = pdf_result["document_id"]
            result["pdf_path"] = pdf_result["pdf_info"]["file_path"]

            extracted_images = []
            if extract_images and pdf_result.get("images"):
                result["image_dir"] = pdf_result["images"]["output_dir"]
                result["total_images"] = pdf_result["images"]["total_images"]
                extracted_images = pdf_result["images"]["images"]

            markdown_text = pdf_result["markdown"]
            text_blocks = pdf_result["text_blocks"]

            log.info(
                f"PDF parsed: {len(markdown_text)} chars, "
                f"{len(text_blocks)} blocks, {result['total_images']} images"
            )

            # Step 2: Chunk the markdown text
            log.info("Step 2: Chunking text...")
            chunk_metadata_base = {
                "document_id": result["document_id"],
                "filename": filename,
                "file_path": result["pdf_path"],
            }

            if additional_metadata:
                chunk_metadata_base.update(additional_metadata)

            chunks = self.chunker.chunk_text(
                text=markdown_text,
                metadata=chunk_metadata_base,
            )

            result["total_chunks"] = len(chunks)
            log.info(f"Created {len(chunks)} chunks from document")

            # Step 3: Embed and upsert TEXT chunks
            if chunks:
                log.info(f"Step 3: Embedding and upserting TEXT chunks in batches of {self.batch_size}...")
                text_vectors_count, chunks_metadata = self._embed_and_upsert_texts(
                    chunks=chunks,
                    filename=filename,
                    document_id=result["document_id"],
                    pdf_path=result["pdf_path"],
                    text_blocks=text_blocks,
                    namespace=namespace,
                )
                result["total_text_vectors"] = text_vectors_count
                result["chunks_metadata"] = chunks_metadata

            # Step 4: Embed and upsert IMAGES
            if embed_images and extracted_images:
                log.info(f"Step 4: Embedding and upserting {len(extracted_images)} IMAGES...")
                image_vectors_count, images_metadata = self._embed_and_upsert_images(
                    images=extracted_images,
                    filename=filename,
                    document_id=result["document_id"],
                    namespace=namespace,
                )
                result["total_image_vectors"] = image_vectors_count
                result["images_metadata"] = images_metadata

            result["total_vectors"] = result["total_text_vectors"] + result["total_image_vectors"]
            result["processing_stats"] = {
                "markdown_length": len(markdown_text),
                "estimated_tokens": pdf_result.get("stats", {}).get("estimated_tokens", 0),
                "total_blocks": len(text_blocks),
                "text_batches_processed": (len(chunks) + self.batch_size - 1) // self.batch_size if chunks else 0,
                "image_batches_processed": (len(extracted_images) + IMAGE_BATCH_SIZE - 1) // IMAGE_BATCH_SIZE if extracted_images else 0,
            }

            log.info(
                f"Document ingestion complete: "
                f"{result['total_text_vectors']} text vectors, "
                f"{result['total_image_vectors']} image vectors"
            )

            return result

        except Exception as e:
            log.error(f"Document ingestion failed: {e}", exc_info=True)
            raise

    def _embed_and_upsert_texts(
        self,
        chunks: List[Dict[str, Any]],
        filename: str,
        document_id: str,
        pdf_path: str,
        text_blocks: List[Dict[str, Any]],
        namespace: str,
    ) -> tuple:
        """Embed text chunks and upsert to Pinecone."""
        total_upserted = 0
        chunks_metadata = []

        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]

            log.info(f"Processing text batch: chunks {batch_start}-{batch_end - 1}")

            # Extract texts for embedding
            batch_texts = [chunk["text"] for chunk in batch_chunks]

            # Generate embeddings
            embed_result = self.embeddings.embed_texts(
                texts=batch_texts,
                input_type="search_document",
            )
            embeddings = embed_result["embeddings"]

            # Prepare vectors for Pinecone
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                chunk_index = batch_start + i
                vector_id = self._create_vector_id(filename, chunk_index, "text")

                # Build metadata with type="text"
                metadata = {
                    "type": "text",  # Important for filtering
                    "text": chunk["text"],
                    "chunk_index": chunk_index,
                    "document_id": document_id,
                    "filename": filename,
                    "file_path": pdf_path,
                    "token_count": chunk["metadata"].get("token_count", 0),
                    "total_chunks": len(chunks),
                }

                # Estimate page number
                if text_blocks:
                    metadata["page_number"] = self._estimate_page_number(chunk["text"], text_blocks)
                else:
                    metadata["page_number"] = 1

                # Add additional metadata from chunk
                for key, value in chunk["metadata"].items():
                    if key not in metadata:
                        metadata[key] = value

                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata,
                })

                chunks_metadata.append({
                    "vector_id": vector_id,
                    "chunk_index": chunk_index,
                    "text_preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
                    "token_count": metadata["token_count"],
                    "page_number": metadata["page_number"],
                })

            # Upsert batch
            upsert_result = self.vector_store.upsert(vectors=vectors, namespace=namespace)
            total_upserted += upsert_result["upserted_count"]
            log.info(f"Upserted {upsert_result['upserted_count']} text vectors")

        return total_upserted, chunks_metadata

    def _embed_and_upsert_images(
        self,
        images: List[Dict[str, Any]],
        filename: str,
        document_id: str,
        namespace: str,
    ) -> tuple:
        """Embed images and upsert to Pinecone."""
        total_upserted = 0
        images_metadata = []

        for batch_start in range(0, len(images), IMAGE_BATCH_SIZE):
            batch_end = min(batch_start + IMAGE_BATCH_SIZE, len(images))
            batch_images = images[batch_start:batch_end]

            log.info(f"Processing image batch: images {batch_start}-{batch_end - 1}")

            # Convert images to data URLs
            image_data_urls = []
            valid_images = []

            for img_info in batch_images:
                try:
                    image_path = img_info["file_path"]
                    data_url = encode_image_to_data_url(image_path)
                    image_data_urls.append(data_url)
                    valid_images.append(img_info)
                except Exception as e:
                    log.warning(f"Failed to encode image {img_info.get('file_path')}: {e}")
                    continue

            if not image_data_urls:
                continue

            # Generate image embeddings
            embed_result = self.embeddings.embed_images(image_data_urls=image_data_urls)
            embeddings = embed_result["embeddings"]

            # Prepare vectors for Pinecone
            vectors = []
            for i, (img_info, embedding) in enumerate(zip(valid_images, embeddings)):
                image_index = batch_start + i
                vector_id = self._create_vector_id(filename, image_index, "image")

                # Build metadata with type="image"
                metadata = {
                    "type": "image",  # Important for filtering
                    "image_path": img_info["file_path"],
                    "image_filename": img_info["filename"],
                    "image_index": image_index,
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": img_info.get("page_number", 1),
                    "width": img_info.get("width", 0),
                    "height": img_info.get("height", 0),
                }

                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata,
                })

                images_metadata.append({
                    "vector_id": vector_id,
                    "image_index": image_index,
                    "image_path": img_info["file_path"],
                    "page_number": metadata["page_number"],
                })

            # Upsert batch
            upsert_result = self.vector_store.upsert(vectors=vectors, namespace=namespace)
            total_upserted += upsert_result["upserted_count"]
            log.info(f"Upserted {upsert_result['upserted_count']} image vectors")

        return total_upserted, images_metadata

    def _estimate_page_number(
        self, chunk_text: str, text_blocks: List[Dict[str, Any]]
    ) -> int:
        """Estimate the page number for a chunk based on text matching."""
        if not text_blocks:
            return 1

        chunk_start = chunk_text[:50].strip().lower()
        best_page = 1
        best_score = 0

        for block in text_blocks:
            block_text = block.get("text", "").lower()
            if chunk_start in block_text:
                return block.get("metadata", {}).get("page_number", 1)

            overlap = sum(1 for c in chunk_start if c in block_text)
            if overlap > best_score:
                best_score = overlap
                best_page = block.get("metadata", {}).get("page_number", 1)

        return best_page

    def get_document_stats(self, document_id: str, namespace: str = "") -> Dict[str, Any]:
        """Get statistics for an ingested document."""
        try:
            index_stats = self.vector_store.describe_index_stats()
            return {
                "document_id": document_id,
                "namespace": namespace or "default",
                "index_stats": index_stats,
            }
        except Exception as e:
            log.error(f"Failed to get document stats: {e}", exc_info=True)
            raise

    def delete_document(self, document_id: str, namespace: str = "") -> Dict[str, Any]:
        """Delete all vectors (text and images) for a document."""
        try:
            result = self.vector_store.delete(
                filter={"document_id": {"$eq": document_id}},
                namespace=namespace,
            )
            log.info(f"Deleted vectors for document: {document_id}")
            return {
                "document_id": document_id,
                "namespace": namespace or "default",
                "status": "deleted",
                "result": result,
            }
        except Exception as e:
            log.error(f"Failed to delete document: {e}", exc_info=True)
            raise
