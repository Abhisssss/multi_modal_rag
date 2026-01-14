"""Token-Based Chunker
=====================

A simple, efficient token-based text chunker with overlap support.
Optimized for RAG workflows where chunk size affects retrieval quality.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import tiktoken

log = logging.getLogger(__name__)

# Default values optimized for RAG
DEFAULT_CHUNK_SIZE = 312  # tokens - good balance for retrieval precision
DEFAULT_CHUNK_OVERLAP = 50  # tokens - ~10% overlap for context continuity
DEFAULT_ENCODING = "cl100k_base"  # GPT-4/Claude compatible encoding


class TokenChunker:
    """
    A token-based text chunker with configurable chunk size and overlap.

    This chunker splits text into chunks based on token count rather than
    character count, which provides more consistent chunk sizes for LLMs.

    Default settings (312 tokens, 50 overlap) are optimized for RAG:
    - 312 tokens fits well in most embedding models
    - 50 token overlap (~10%) maintains context continuity
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        encoding_name: str = DEFAULT_ENCODING,
    ):
        """
        Initialize the TokenChunker.

        Args:
            chunk_size: Maximum number of tokens per chunk. Default: 312
            chunk_overlap: Number of overlapping tokens between chunks. Default: 50
            encoding_name: Tiktoken encoding name. Default: 'cl100k_base' (GPT-4/Claude)

        Raises:
            ValueError: If chunk_overlap >= chunk_size or values are invalid.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name

        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            log.error(f"Failed to load tiktoken encoding '{encoding_name}': {e}")
            raise ValueError(f"Invalid encoding name: {encoding_name}") from e

        log.info(
            f"TokenChunker initialized: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, encoding={encoding_name}"
        )

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        return len(self.tokenizer.encode(text))

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Splits text into chunks based on token count with overlap.

        Args:
            text: The text to chunk.
            metadata: Optional metadata to include with each chunk.

        Returns:
            List of chunk dictionaries, each containing:
                - id: Unique chunk identifier
                - text: The chunk text
                - metadata: Chunk metadata including position info
        """
        if not text or not text.strip():
            log.warning("Empty text provided for chunking.")
            return []

        # Encode the entire text to tokens
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        log.info(f"Chunking text with {total_tokens} tokens...")

        chunks = []
        chunk_index = 0
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < total_tokens:
            # Get token slice for this chunk
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]

            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Build chunk metadata
            chunk_metadata = {
                "chunk_index": chunk_index,
                "start_token": start,
                "end_token": end,
                "token_count": len(chunk_tokens),
                "total_tokens": total_tokens,
                "total_chunks": None,  # Will be set after loop
            }

            # Merge with provided metadata
            if metadata:
                chunk_metadata.update(metadata)

            chunk = {
                "id": f"chunk_{uuid.uuid4().hex[:12]}",
                "text": chunk_text,
                "metadata": chunk_metadata,
            }
            chunks.append(chunk)

            chunk_index += 1
            start += step

            # Safety check to prevent infinite loop
            if step <= 0:
                break

        # Update total_chunks in all chunk metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = total_chunks

        log.info(f"Created {total_chunks} chunks from {total_tokens} tokens.")
        return chunks

    def chunk_texts(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunks multiple texts, maintaining document boundaries.

        Args:
            texts: List of texts to chunk.
            metadata_list: Optional list of metadata dicts (one per text).

        Returns:
            List of all chunks from all texts.
        """
        if metadata_list and len(metadata_list) != len(texts):
            raise ValueError("metadata_list must have same length as texts.")

        all_chunks = []
        for i, text in enumerate(texts):
            metadata = metadata_list[i] if metadata_list else None
            if metadata is None:
                metadata = {}
            metadata["document_index"] = i

            chunks = self.chunk_text(text, metadata=metadata)
            all_chunks.extend(chunks)

        log.info(f"Created {len(all_chunks)} total chunks from {len(texts)} texts.")
        return all_chunks

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the current chunker configuration.

        Returns:
            Dict with chunk_size, chunk_overlap, and encoding_name.
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "encoding_name": self.encoding_name,
        }
