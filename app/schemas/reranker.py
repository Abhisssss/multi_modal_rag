"""
Pydantic Schemas for Reranker Service
======================================

This module defines the Pydantic models for standardizing the input and output
of the document reranking services.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class RerankerRequest(BaseModel):
    """Request schema for document reranking."""
    query: str = Field(
        ...,
        min_length=1,
        description="The query string to rank documents against."
    )
    documents: List[str] = Field(
        ...,
        min_length=1,
        description="A list of document strings to rerank."
    )
    top_n: Optional[int] = Field(
        None,
        ge=1,
        description="The number of top results to return. If None, returns all documents."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the capital of the United States?",
                "documents": [
                    "Carson City is the capital city of the American state of Nevada.",
                    "Washington, D.C. is the capital of the United States.",
                    "Capital punishment has existed in the United States since before the country was founded."
                ],
                "top_n": 2
            }
        }


class RerankedDocument(BaseModel):
    """A single reranked document result."""
    rank: int = Field(
        ...,
        description="The rank position (1-indexed)."
    )
    index: int = Field(
        ...,
        description="The original index of the document in the input list."
    )
    relevance_score: float = Field(
        ...,
        description="The relevance score from the reranker model."
    )
    document: str = Field(
        ...,
        description="The document text."
    )


class RerankerResponse(BaseModel):
    """Response schema for document reranking."""
    id: Optional[str] = Field(
        None,
        description="The unique ID of the rerank request."
    )
    results: List[RerankedDocument] = Field(
        ...,
        description="The reranked documents with scores."
    )
    query: str = Field(
        ...,
        description="The original query."
    )
    meta: Optional[Dict[str, Any]] = Field(
        None,
        description="API metadata including version and billing info."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "5807ee2e-0cda-445a-9ec8-864c60a06606",
                "results": [
                    {
                        "rank": 1,
                        "index": 1,
                        "relevance_score": 0.9876,
                        "document": "Washington, D.C. is the capital of the United States."
                    },
                    {
                        "rank": 2,
                        "index": 0,
                        "relevance_score": 0.2345,
                        "document": "Carson City is the capital city of the American state of Nevada."
                    }
                ],
                "query": "What is the capital of the United States?",
                "meta": {
                    "api_version": {"version": "2"},
                    "billed_units": {"search_units": 1}
                }
            }
        }
