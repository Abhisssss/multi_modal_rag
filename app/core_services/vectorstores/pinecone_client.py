"""Pinecone Vector Store Client
==============================

This module provides a client for interacting with Pinecone vector database.
It supports upserting vectors with metadata and querying for similar vectors.
"""

import logging
from typing import Any, Dict, List, Optional

from pinecone.grpc import PineconeGRPC as Pinecone

from app.core.config import settings

log = logging.getLogger(__name__)


class PineconeVectorStore:
    """
    A client for interacting with Pinecone vector database.

    This client handles:
    - Upserting vectors with metadata to specified namespaces
    - Querying vectors for similarity search
    - Batch operations for efficient data handling
    """

    def __init__(self):
        """
        Initializes the PineconeVectorStore client.

        Raises:
            ValueError: If required Pinecone settings are not configured.
        """
        if not settings.PINECONE_API_KEY:
            log.error("PINECONE_API_KEY is not configured.")
            raise ValueError("PINECONE_API_KEY is not set in the environment or .env file.")

        if not settings.PINECONE_HOST:
            log.error("PINECONE_HOST is not configured.")
            raise ValueError("PINECONE_HOST is not set in the environment or .env file.")

        log.info("Initializing PineconeVectorStore client.")
        self.client = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.client.Index(host=settings.PINECONE_HOST)
        self.index_name = settings.PINECONE_INDEX

        log.info(f"Connected to Pinecone index: {self.index_name}")

    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Upserts vectors into the Pinecone index.

        Each vector should be a dictionary with:
        - id: Unique identifier for the vector
        - values: List of floats representing the vector embedding
        - metadata: Optional dictionary of metadata

        Args:
            vectors: List of vector dictionaries to upsert.
                Example:
                [
                    {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"key": "value"}},
                    {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"key": "value"}}
                ]
            namespace: The namespace to upsert vectors into. Default is empty string.

        Returns:
            Dict containing upsert response with upserted_count.

        Raises:
            ValueError: If vectors list is empty or malformed.
            Exception: For any Pinecone API errors.
        """
        if not vectors:
            raise ValueError("Vectors list cannot be empty.")

        log.info(f"Upserting {len(vectors)} vectors to namespace: '{namespace or 'default'}'")

        try:
            # Validate vector format
            for i, vec in enumerate(vectors):
                if "id" not in vec:
                    raise ValueError(f"Vector at index {i} missing 'id' field.")
                if "values" not in vec:
                    raise ValueError(f"Vector at index {i} missing 'values' field.")
                if not isinstance(vec["values"], list):
                    raise ValueError(f"Vector at index {i} 'values' must be a list.")

            # Perform upsert
            response = self.index.upsert(
                vectors=vectors,
                namespace=namespace
            )

            result = {
                "upserted_count": response.upserted_count if hasattr(response, 'upserted_count') else len(vectors),
                "namespace": namespace or "default"
            }

            log.info(f"Successfully upserted {result['upserted_count']} vectors.")
            return result

        except Exception as e:
            log.error(f"Failed to upsert vectors: {e}", exc_info=True)
            raise Exception(f"Pinecone upsert error: {e}") from e

    def upsert_batch(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Upserts vectors in batches for large datasets.

        Args:
            vectors: List of vector dictionaries to upsert.
            namespace: The namespace to upsert vectors into.
            batch_size: Number of vectors per batch. Default is 100.

        Returns:
            Dict containing total upserted count and batch info.
        """
        if not vectors:
            raise ValueError("Vectors list cannot be empty.")

        log.info(f"Batch upserting {len(vectors)} vectors (batch_size={batch_size})")

        total_upserted = 0
        batches_processed = 0

        try:
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                result = self.upsert(batch, namespace=namespace)
                total_upserted += result["upserted_count"]
                batches_processed += 1
                log.debug(f"Processed batch {batches_processed}: {len(batch)} vectors")

            return {
                "total_upserted": total_upserted,
                "batches_processed": batches_processed,
                "namespace": namespace or "default"
            }

        except Exception as e:
            log.error(f"Failed during batch upsert: {e}", exc_info=True)
            raise

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: str = "",
        include_metadata: bool = True,
        include_values: bool = False,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Queries the Pinecone index for similar vectors.

        Args:
            vector: The query vector (dense embedding).
            top_k: Number of top results to return. Default is 10.
            namespace: The namespace to query. Default is empty string.
            include_metadata: Whether to include metadata in results. Default is True.
            include_values: Whether to include vector values in results. Default is False.
            filter: Optional metadata filter for the query.

        Returns:
            Dict containing:
                - matches: List of matching vectors with id, score, and optionally metadata/values.
                - namespace: The queried namespace.

        Raises:
            ValueError: If vector is empty.
            Exception: For any Pinecone API errors.
        """
        if not vector:
            raise ValueError("Query vector cannot be empty.")

        log.info(f"Querying namespace '{namespace or 'default'}' with top_k={top_k}")

        try:
            query_params = {
                "namespace": namespace,
                "vector": vector,
                "top_k": top_k,
                "include_metadata": include_metadata,
                "include_values": include_values
            }

            if filter:
                query_params["filter"] = filter

            response = self.index.query(**query_params)

            # Parse matches
            matches = []
            for match in response.matches:
                match_data = {
                    "id": match.id,
                    "score": match.score
                }
                if include_metadata and hasattr(match, 'metadata') and match.metadata:
                    match_data["metadata"] = dict(match.metadata)
                if include_values and hasattr(match, 'values') and match.values:
                    match_data["values"] = list(match.values)
                matches.append(match_data)

            result = {
                "matches": matches,
                "namespace": namespace or "default",
                "top_k": top_k,
                "total_matches": len(matches)
            }

            log.info(f"Query returned {len(matches)} matches.")
            return result

        except Exception as e:
            log.error(f"Failed to query vectors: {e}", exc_info=True)
            raise Exception(f"Pinecone query error: {e}") from e

    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: str = "",
        delete_all: bool = False,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deletes vectors from the Pinecone index.

        Args:
            ids: List of vector IDs to delete.
            namespace: The namespace to delete from.
            delete_all: If True, deletes all vectors in the namespace.
            filter: Optional metadata filter for deletion.

        Returns:
            Dict with deletion status.

        Raises:
            ValueError: If neither ids nor delete_all is specified.
            Exception: For any Pinecone API errors.
        """
        if not ids and not delete_all and not filter:
            raise ValueError("Must specify either 'ids', 'delete_all=True', or 'filter'.")

        log.info(f"Deleting vectors from namespace: '{namespace or 'default'}'")

        try:
            if delete_all:
                self.index.delete(delete_all=True, namespace=namespace)
                log.info(f"Deleted all vectors in namespace '{namespace or 'default'}'")
                return {"deleted": "all", "namespace": namespace or "default"}
            elif ids:
                self.index.delete(ids=ids, namespace=namespace)
                log.info(f"Deleted {len(ids)} vectors by ID.")
                return {"deleted_ids": ids, "count": len(ids), "namespace": namespace or "default"}
            elif filter:
                self.index.delete(filter=filter, namespace=namespace)
                log.info(f"Deleted vectors matching filter.")
                return {"deleted_by_filter": filter, "namespace": namespace or "default"}

        except Exception as e:
            log.error(f"Failed to delete vectors: {e}", exc_info=True)
            raise Exception(f"Pinecone delete error: {e}") from e

    def describe_index_stats(self) -> Dict[str, Any]:
        """
        Gets statistics about the Pinecone index.

        Returns:
            Dict containing index statistics including dimension, count, and namespaces.
        """
        log.info("Fetching index statistics.")

        try:
            stats = self.index.describe_index_stats()

            result = {
                "dimension": stats.dimension if hasattr(stats, 'dimension') else None,
                "total_vector_count": stats.total_vector_count if hasattr(stats, 'total_vector_count') else 0,
                "namespaces": {}
            }

            if hasattr(stats, 'namespaces') and stats.namespaces:
                for ns_name, ns_stats in stats.namespaces.items():
                    result["namespaces"][ns_name] = {
                        "vector_count": ns_stats.vector_count if hasattr(ns_stats, 'vector_count') else 0
                    }

            log.info(f"Index stats: {result['total_vector_count']} total vectors.")
            return result

        except Exception as e:
            log.error(f"Failed to fetch index stats: {e}", exc_info=True)
            raise Exception(f"Pinecone stats error: {e}") from e
