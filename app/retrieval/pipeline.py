"""Retrieval Pipeline
====================

A retrieval pipeline that embeds queries, searches Pinecone, and reranks results
for optimal context selection in RAG workflows.
Supports both text-only and multi-modal (text + images) retrieval.
"""

import logging
from typing import Any, Dict, List, Optional

from app.core_services.embeddings.cohere_embeddings import CohereEmbeddings
from app.core_services.rerankers.cohere_reranker import CohereReranker
from app.core_services.vectorstores.pinecone_client import PineconeVectorStore

log = logging.getLogger(__name__)

# Default retrieval settings
DEFAULT_TEXT_TOP_K = 10  # Initial text retrieval from Pinecone
DEFAULT_TEXT_TOP_N = 3   # Final text results after reranking
DEFAULT_IMAGE_TOP_K = 5  # Number of images to retrieve


class RetrievalPipeline:
    """
    A retrieval pipeline for RAG that combines vector search with reranking.

    This pipeline supports:
    - Text-only retrieval with reranking
    - Multi-modal retrieval (text + images) for vision LLMs
    """

    def __init__(self):
        """
        Initialize the RetrievalPipeline.

        Initializes connections to:
        - CohereEmbeddings for query embedding
        - PineconeVectorStore for vector search
        - CohereReranker for result reranking
        """
        log.info("Initializing RetrievalPipeline...")

        self.embeddings = CohereEmbeddings()
        self.vector_store = PineconeVectorStore()
        self.reranker = CohereReranker()

        log.info("RetrievalPipeline initialized successfully.")

    def retrieve(
        self,
        query: str,
        namespace: str = "",
        top_k: int = DEFAULT_TEXT_TOP_K,
        top_n: int = DEFAULT_TEXT_TOP_N,
        filter: Optional[Dict[str, Any]] = None,
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant text chunks for a query using vector search and reranking.

        Args:
            query: The user's query string.
            namespace: Pinecone namespace to search in.
            top_k: Number of candidates to retrieve from Pinecone.
            top_n: Number of final results after reranking.
            filter: Optional Pinecone metadata filter.
            use_reranker: Whether to apply reranking (default: True).

        Returns:
            Dict containing:
                - query: Original query
                - chunks: List of retrieved chunks with text, score, and metadata
                - total_retrieved: Number of chunks initially retrieved
                - total_after_rerank: Number of chunks after reranking
                - namespace: Namespace searched
        """
        log.info(f"Retrieving for query: '{query[:50]}...' (top_k={top_k}, top_n={top_n})")

        result = {
            "query": query,
            "chunks": [],
            "total_retrieved": 0,
            "total_after_rerank": 0,
            "namespace": namespace or "default",
        }

        try:
            # Step 1: Embed the query
            log.info("Step 1: Embedding query...")
            embed_result = self.embeddings.embed_texts(
                texts=[query],
                input_type="search_query",
            )
            query_vector = embed_result["embeddings"][0]
            log.info(f"Query embedded. Vector dimension: {len(query_vector)}")

            # Step 2: Search Pinecone (text only if filter not specified)
            search_filter = filter.copy() if filter else {}
            # Add type filter for text if not already specified
            if "type" not in search_filter:
                search_filter["type"] = {"$eq": "text"}

            log.info(f"Step 2: Searching Pinecone for TEXT (top_k={top_k})...")
            search_result = self.vector_store.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                include_values=False,
                filter=search_filter,
            )

            matches = search_result.get("matches", [])
            result["total_retrieved"] = len(matches)
            log.info(f"Retrieved {len(matches)} text candidates from Pinecone.")

            if not matches:
                log.warning("No matches found in Pinecone.")
                return result

            # Extract chunk data
            chunks = []
            for match in matches:
                chunk = {
                    "id": match["id"],
                    "text": match.get("metadata", {}).get("text", ""),
                    "score": match["score"],
                    "metadata": match.get("metadata", {}),
                }
                chunks.append(chunk)

            # Step 3: Rerank (if enabled and we have chunks)
            if use_reranker and len(chunks) > 0:
                log.info(f"Step 3: Reranking to top {top_n}...")
                chunks = self._rerank_chunks(query, chunks, top_n)
                result["total_after_rerank"] = len(chunks)
                log.info(f"Reranked to {len(chunks)} final chunks.")
            else:
                chunks = chunks[:top_n]
                result["total_after_rerank"] = len(chunks)

            result["chunks"] = chunks
            return result

        except Exception as e:
            log.error(f"Retrieval failed: {e}", exc_info=True)
            raise

    def retrieve_multimodal(
        self,
        query: str,
        namespace: str = "",
        text_top_k: int = DEFAULT_TEXT_TOP_K,
        text_top_n: int = DEFAULT_TEXT_TOP_N,
        image_top_k: int = DEFAULT_IMAGE_TOP_K,
        use_reranker: bool = True,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve both text chunks and images for multi-modal RAG.

        This method:
        1. Embeds the user query
        2. Searches Pinecone for top_k text chunks (filter: type="text")
        3. Searches Pinecone for top_k images (filter: type="image")
        4. Reranks only text chunks to get top_n
        5. Returns both text chunks and images

        Args:
            query: The user's query string.
            namespace: Pinecone namespace to search in.
            text_top_k: Number of text candidates to retrieve.
            text_top_n: Number of text chunks after reranking.
            image_top_k: Number of images to retrieve.
            use_reranker: Whether to apply reranking on text.
            filter: Optional base filter (type filter will be added automatically).

        Returns:
            Dict containing:
                - query: Original query
                - text_chunks: List of text chunks (reranked)
                - images: List of retrieved images with metadata (including image_path)
                - total_text_retrieved: Initial text count
                - total_text_after_rerank: Text count after reranking
                - total_images_retrieved: Image count
                - namespace: Namespace searched
        """
        log.info(
            f"Multi-modal retrieval for query: '{query[:50]}...' "
            f"(text_top_k={text_top_k}, text_top_n={text_top_n}, image_top_k={image_top_k})"
        )

        result = {
            "query": query,
            "text_chunks": [],
            "images": [],
            "total_text_retrieved": 0,
            "total_text_after_rerank": 0,
            "total_images_retrieved": 0,
            "namespace": namespace or "default",
        }

        try:
            # Step 1: Embed the query
            log.info("Step 1: Embedding query...")
            embed_result = self.embeddings.embed_texts(
                texts=[query],
                input_type="search_query",
            )
            query_vector = embed_result["embeddings"][0]
            log.info(f"Query embedded. Vector dimension: {len(query_vector)}")

            # Base filter for additional constraints
            base_filter = filter.copy() if filter else {}

            # Step 2: Search for TEXT chunks
            log.info(f"Step 2: Searching Pinecone for TEXT (top_k={text_top_k})...")
            text_filter = {**base_filter, "type": {"$eq": "text"}}

            text_search_result = self.vector_store.query(
                vector=query_vector,
                top_k=text_top_k,
                namespace=namespace,
                include_metadata=True,
                include_values=False,
                filter=text_filter,
            )

            text_matches = text_search_result.get("matches", [])
            result["total_text_retrieved"] = len(text_matches)
            log.info(f"Retrieved {len(text_matches)} text candidates.")

            # Extract text chunk data
            text_chunks = []
            for match in text_matches:
                chunk = {
                    "id": match["id"],
                    "text": match.get("metadata", {}).get("text", ""),
                    "score": match["score"],
                    "metadata": match.get("metadata", {}),
                }
                text_chunks.append(chunk)

            # Step 3: Rerank text chunks
            if use_reranker and len(text_chunks) > 0:
                log.info(f"Step 3: Reranking text to top {text_top_n}...")
                text_chunks = self._rerank_chunks(query, text_chunks, text_top_n)
                result["total_text_after_rerank"] = len(text_chunks)
                log.info(f"Reranked to {len(text_chunks)} text chunks.")
            else:
                text_chunks = text_chunks[:text_top_n]
                result["total_text_after_rerank"] = len(text_chunks)

            result["text_chunks"] = text_chunks

            # Step 4: Search for IMAGES
            log.info(f"Step 4: Searching Pinecone for IMAGES (top_k={image_top_k})...")
            image_filter = {**base_filter, "type": {"$eq": "image"}}

            image_search_result = self.vector_store.query(
                vector=query_vector,
                top_k=image_top_k,
                namespace=namespace,
                include_metadata=True,
                include_values=False,
                filter=image_filter,
            )

            image_matches = image_search_result.get("matches", [])
            result["total_images_retrieved"] = len(image_matches)
            log.info(f"Retrieved {len(image_matches)} images.")

            # Extract image data
            images = []
            for match in image_matches:
                metadata = match.get("metadata", {})
                image = {
                    "id": match["id"],
                    "score": match["score"],
                    "image_path": metadata.get("image_path", ""),
                    "image_filename": metadata.get("image_filename", ""),
                    "page_number": metadata.get("page_number", 1),
                    "document_id": metadata.get("document_id", ""),
                    "filename": metadata.get("filename", ""),
                    "width": metadata.get("width", 0),
                    "height": metadata.get("height", 0),
                    "metadata": metadata,
                }
                images.append(image)

            result["images"] = images

            log.info(
                f"Multi-modal retrieval complete: "
                f"{len(text_chunks)} text chunks, {len(images)} images"
            )

            return result

        except Exception as e:
            log.error(f"Multi-modal retrieval failed: {e}", exc_info=True)
            raise

    def _rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using Cohere reranker.

        Args:
            query: The user's query.
            chunks: List of chunk dictionaries with 'text' field.
            top_n: Number of top results to return.

        Returns:
            List of reranked chunks with updated scores.
        """
        documents = [chunk["text"] for chunk in chunks]

        rerank_result = self.reranker.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
        )

        reranked_chunks = []
        for result in rerank_result["results"]:
            original_index = result["index"]
            original_chunk = chunks[original_index]

            reranked_chunk = {
                "id": original_chunk["id"],
                "text": original_chunk["text"],
                "vector_score": original_chunk["score"],
                "rerank_score": result["relevance_score"],
                "rank": result["rank"],
                "metadata": original_chunk["metadata"],
            }
            reranked_chunks.append(reranked_chunk)

        return reranked_chunks

    def retrieve_simple(
        self,
        query: str,
        namespace: str = "",
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Simple retrieval without reranking.

        Args:
            query: The user's query string.
            namespace: Pinecone namespace to search in.
            top_k: Number of results to return.
            filter: Optional Pinecone metadata filter.

        Returns:
            List of retrieved chunks.
        """
        result = self.retrieve(
            query=query,
            namespace=namespace,
            top_k=top_k,
            top_n=top_k,
            filter=filter,
            use_reranker=False,
        )
        return result["chunks"]
