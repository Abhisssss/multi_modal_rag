"""Document Ingestion Pipeline
=============================

A complete pipeline for ingesting PDF documents into the RAG system.
Handles text extraction, chunking, embedding, and vector storage.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.core_services.chunking.default_chunker import TokenChunker
from app.core_services.embeddings.cohere_embeddings import CohereEmbeddings
from app.core_services.file_parsers.pdf_parser import PDFParser
from app.core_services.vectorstores.pinecone_client import PineconeVectorStore

log = logging.getLogger(__name__)

# Default batch size for embedding and upserting
DEFAULT_BATCH_SIZE = 50


class DocumentIngestionPipeline:
    """
    A pipeline for ingesting documents into the RAG system.

    This pipeline handles the complete workflow:
    1. PDF parsing (text and image extraction)
    2. Text chunking with overlap
    3. Batch embedding generation
    4. Vector storage with metadata
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
        # Remove extension
        name = Path(filename).stem
        # Replace non-alphanumeric characters with underscores
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        # Remove consecutive underscores
        name = re.sub(r"_+", "_", name)
        return name.lower().strip("_")

    def _create_vector_id(self, filename: str, chunk_index: int) -> str:
        """Create a unique vector ID from filename and chunk index."""
        sanitized = self._sanitize_filename(filename)
        return f"{sanitized}_chunk_{chunk_index:04d}"

    def ingest_pdf(
        self,
        file_source: Union[str, Path, bytes],
        filename: str,
        namespace: str = "",
        extract_images: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a PDF document into the RAG system.

        This method performs the complete ingestion workflow:
        1. Parse PDF and extract text/images
        2. Chunk the extracted text
        3. Generate embeddings in batches
        4. Upsert vectors to Pinecone with metadata

        Args:
            file_source: PDF file as path or bytes.
            filename: Original filename of the PDF.
            namespace: Pinecone namespace for vectors.
            extract_images: Whether to extract images from PDF.
            additional_metadata: Optional extra metadata to include with each chunk.

        Returns:
            Dict containing:
                - document_id: Unique identifier for the document.
                - filename: Original filename.
                - pdf_path: Path to stored PDF.
                - image_dir: Path to extracted images (if any).
                - total_chunks: Number of text chunks created.
                - total_vectors: Number of vectors upserted.
                - total_images: Number of images extracted.
                - namespace: Pinecone namespace used.
                - chunks_metadata: List of chunk metadata for reference.
        """
        log.info(f"Starting document ingestion for: {filename}")

        result = {
            "document_id": None,
            "filename": filename,
            "pdf_path": None,
            "image_dir": None,
            "total_chunks": 0,
            "total_vectors": 0,
            "total_images": 0,
            "namespace": namespace or "default",
            "chunks_metadata": [],
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

            if extract_images and pdf_result.get("images"):
                result["image_dir"] = pdf_result["images"]["output_dir"]
                result["total_images"] = pdf_result["images"]["total_images"]

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

            if not chunks:
                log.warning("No chunks created from document. Skipping embedding.")
                return result

            # Step 3 & 4: Embed and upsert in batches
            log.info(f"Step 3 & 4: Embedding and upserting in batches of {self.batch_size}...")

            total_upserted = 0
            chunks_metadata = []

            # Process chunks in batches
            for batch_start in range(0, len(chunks), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]

                log.info(f"Processing batch {batch_start // self.batch_size + 1}: chunks {batch_start}-{batch_end - 1}")

                # Extract texts for embedding
                batch_texts = [chunk["text"] for chunk in batch_chunks]

                # Generate embeddings for batch
                embed_result = self.embeddings.embed_texts(
                    texts=batch_texts,
                    input_type="search_document",
                )
                embeddings = embed_result["embeddings"]

                # Prepare vectors for Pinecone
                vectors = []
                for i, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_index = batch_start + i
                    vector_id = self._create_vector_id(filename, chunk_index)

                    # Build comprehensive metadata
                    metadata = {
                        "text": chunk["text"],
                        "chunk_index": chunk_index,
                        "document_id": result["document_id"],
                        "filename": filename,
                        "file_path": result["pdf_path"],
                        "token_count": chunk["metadata"].get("token_count", 0),
                        "total_chunks": result["total_chunks"],
                    }

                    # Add page number estimation based on position
                    # (approximate based on chunk position in document)
                    if text_blocks:
                        # Estimate page from text blocks if available
                        estimated_page = self._estimate_page_number(
                            chunk["text"], text_blocks
                        )
                        metadata["page_number"] = estimated_page
                    else:
                        metadata["page_number"] = 1

                    # Add any additional metadata from chunk
                    for key, value in chunk["metadata"].items():
                        if key not in metadata:
                            metadata[key] = value

                    vector = {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata,
                    }
                    vectors.append(vector)

                    # Store chunk metadata for response
                    chunks_metadata.append({
                        "vector_id": vector_id,
                        "chunk_index": chunk_index,
                        "text_preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
                        "token_count": metadata["token_count"],
                        "page_number": metadata["page_number"],
                    })

                # Upsert batch to Pinecone
                upsert_result = self.vector_store.upsert(
                    vectors=vectors,
                    namespace=namespace,
                )
                total_upserted += upsert_result["upserted_count"]
                log.info(f"Upserted {upsert_result['upserted_count']} vectors")

            result["total_vectors"] = total_upserted
            result["chunks_metadata"] = chunks_metadata
            result["processing_stats"] = {
                "markdown_length": len(markdown_text),
                "estimated_tokens": pdf_result.get("stats", {}).get("estimated_tokens", 0),
                "total_blocks": len(text_blocks),
                "batches_processed": (len(chunks) + self.batch_size - 1) // self.batch_size,
            }

            log.info(
                f"Document ingestion complete: {result['total_chunks']} chunks, "
                f"{result['total_vectors']} vectors upserted"
            )

            return result

        except Exception as e:
            log.error(f"Document ingestion failed: {e}", exc_info=True)
            raise

    def _estimate_page_number(
        self, chunk_text: str, text_blocks: List[Dict[str, Any]]
    ) -> int:
        """
        Estimate the page number for a chunk based on text matching.

        Uses fuzzy matching against text blocks to find the most likely page.
        """
        if not text_blocks:
            return 1

        # Take first 50 chars of chunk for matching
        chunk_start = chunk_text[:50].strip().lower()

        best_page = 1
        best_score = 0

        for block in text_blocks:
            block_text = block.get("text", "").lower()
            if chunk_start in block_text:
                # Found a match, return this page
                return block.get("metadata", {}).get("page_number", 1)

            # Simple overlap scoring
            overlap = sum(1 for c in chunk_start if c in block_text)
            if overlap > best_score:
                best_score = overlap
                best_page = block.get("metadata", {}).get("page_number", 1)

        return best_page

    def get_document_stats(self, document_id: str, namespace: str = "") -> Dict[str, Any]:
        """
        Get statistics for an ingested document.

        Args:
            document_id: The document identifier.
            namespace: The Pinecone namespace.

        Returns:
            Dict with document statistics from the vector store.
        """
        try:
            # Query for vectors with this document_id
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
        """
        Delete all vectors for a document from the vector store.

        Args:
            document_id: The document identifier.
            namespace: The Pinecone namespace.

        Returns:
            Dict with deletion status.
        """
        try:
            # Delete by filter on document_id
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
