"""
Pydantic Schemas for PDF Parser Service
========================================

This module defines the Pydantic models for standardizing the input and output
of the PDF parsing services.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class TextBlockMetadata(BaseModel):
    """Metadata for a text block extracted from a PDF."""
    page_number: int = Field(..., description="The page number (1-indexed).")
    block_number: int = Field(..., description="The block number on the page.")
    block_type: str = Field(..., description="Type of block ('text' or 'image').")
    bounding_box: List[float] = Field(..., description="Bounding box coordinates [x0, y0, x1, y1].")
    token_count: int = Field(..., description="Estimated token count for the text.")


class TextBlock(BaseModel):
    """A text block extracted from a PDF."""
    text: str = Field(..., description="The text content of the block.")
    metadata: TextBlockMetadata = Field(..., description="Metadata about the block.")


class ExtractedImage(BaseModel):
    """Information about an extracted image from a PDF."""
    page_number: int = Field(..., description="The page number where the image was found.")
    image_index: int = Field(..., description="The index of the image on the page.")
    file_path: str = Field(..., description="Path to the saved image file.")
    filename: str = Field(..., description="Filename of the saved image.")
    width: int = Field(..., description="Image width in pixels.")
    height: int = Field(..., description="Image height in pixels.")
    extension: str = Field(..., description="Image file extension (e.g., 'png', 'jpeg').")
    xref: int = Field(..., description="PDF internal reference ID for the image.")


class ImageExtractionResult(BaseModel):
    """Result of image extraction from a PDF."""
    document_id: str = Field(..., description="Unique identifier for the document.")
    output_dir: str = Field(..., description="Directory where images are saved.")
    images: List[ExtractedImage] = Field(..., description="List of extracted images.")
    total_images: int = Field(..., description="Total number of images extracted.")


class PDFInfo(BaseModel):
    """Information about a saved PDF file."""
    document_id: str = Field(..., description="Unique identifier for the document.")
    file_path: str = Field(..., description="Path to the saved PDF file.")
    filename: str = Field(..., description="Filename of the saved PDF.")


class PDFStats(BaseModel):
    """Statistics about PDF parsing."""
    markdown_length: Optional[int] = Field(None, description="Length of markdown content in characters.")
    estimated_tokens: Optional[int] = Field(None, description="Estimated token count for the content.")
    total_blocks: Optional[int] = Field(None, description="Total number of text blocks extracted.")
    total_images: Optional[int] = Field(None, description="Total number of images extracted.")


class PDFParseResponse(BaseModel):
    """Response from full PDF parsing."""
    document_id: str = Field(..., description="Unique identifier for the document.")
    pdf_info: PDFInfo = Field(..., description="Information about the saved PDF.")
    markdown: str = Field(..., description="Extracted text in markdown format.")
    markdown_truncated: Optional[str] = Field(
        None,
        description="Truncated markdown preview (first and last portions)."
    )
    text_blocks: List[TextBlock] = Field(..., description="Structured text blocks with metadata.")
    text_blocks_truncated: Optional[List[TextBlock]] = Field(
        None,
        description="First few text blocks (for preview)."
    )
    images: Optional[ImageExtractionResult] = Field(
        None,
        description="Image extraction results."
    )
    stats: PDFStats = Field(..., description="Parsing statistics.")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "pdf_info": {
                    "document_id": "550e8400-e29b-41d4-a716-446655440000",
                    "file_path": "storage/pdfs/document.pdf",
                    "filename": "document.pdf"
                },
                "markdown": "# Document Title\n\nThis is the content...",
                "markdown_truncated": "# Document Title\n\nThis is the content... [TRUNCATED] ...end of document.",
                "text_blocks": [],
                "images": {
                    "document_id": "550e8400-e29b-41d4-a716-446655440000",
                    "output_dir": "storage/images/550e8400-e29b-41d4-a716-446655440000",
                    "images": [],
                    "total_images": 5
                },
                "stats": {
                    "markdown_length": 15000,
                    "estimated_tokens": 3750,
                    "total_blocks": 45,
                    "total_images": 5
                }
            }
        }


class PDFTextResponse(BaseModel):
    """Response from PDF text extraction only."""
    document_id: str = Field(..., description="Unique identifier for the document.")
    markdown: str = Field(..., description="Extracted text in markdown format.")
    markdown_truncated: Optional[str] = Field(
        None,
        description="Truncated markdown preview."
    )
    stats: Dict[str, Any] = Field(..., description="Extraction statistics.")


class PDFImagesResponse(BaseModel):
    """Response from PDF image extraction only."""
    document_id: str = Field(..., description="Unique identifier for the document.")
    output_dir: str = Field(..., description="Directory where images are saved.")
    images: List[ExtractedImage] = Field(..., description="List of extracted images.")
    total_images: int = Field(..., description="Total number of images extracted.")
