"""PDF Parser
===========

A unified parser for extracting text and images from PDF files using PyMuPDF.
Supports markdown conversion for RAG workflows and image extraction for multi-modal processing.
"""

import io
import logging
import uuid
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fitz
import pymupdf4llm
from PIL import Image

log = logging.getLogger(__name__)


class PDFParser:
    """
    A unified parser for extracting structured text and images from PDF files.

    This parser uses PyMuPDF (fitz) for robust PDF processing and supports:
    - Text extraction as markdown (optimized for RAG workflows)
    - Text extraction as structured blocks with metadata
    - Image extraction with page and position information
    - Long-term storage of both PDF files and extracted images
    """

    def __init__(
        self,
        storage_base_dir: str = "storage",
        pdf_storage_dir: str = "pdfs",
        image_storage_dir: str = "images",
    ):
        """
        Initialize the PDFParser.

        Args:
            storage_base_dir: Base directory for all storage.
            pdf_storage_dir: Subdirectory for storing PDF files.
            image_storage_dir: Subdirectory for storing extracted images.
        """
        self.storage_base = Path(storage_base_dir)
        self.pdf_storage = self.storage_base / pdf_storage_dir
        self.image_storage = self.storage_base / image_storage_dir

        # Ensure directories exist
        self.pdf_storage.mkdir(parents=True, exist_ok=True)
        self.image_storage.mkdir(parents=True, exist_ok=True)

        log.info(f"PDFParser initialized. PDF storage: {self.pdf_storage}, Image storage: {self.image_storage}")

    def _count_tokens(self, text: str) -> int:
        """Estimate token count as ceil(len(text) / 4)."""
        return ceil(len(text) / 4)

    def _generate_document_id(self) -> str:
        """Generate a unique document ID."""
        return str(uuid.uuid4())

    def save_pdf(self, file_source: Union[str, Path, bytes], filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Saves a PDF file to long-term storage.

        Args:
            file_source: The PDF file as a path or bytes.
            filename: Optional filename. If not provided, generates a UUID-based name.

        Returns:
            Dict with document_id, file_path, and filename.
        """
        document_id = self._generate_document_id()

        if filename is None:
            filename = f"{document_id}.pdf"
        elif not filename.endswith('.pdf'):
            filename = f"{filename}.pdf"

        save_path = self.pdf_storage / filename

        try:
            if isinstance(file_source, bytes):
                with open(save_path, 'wb') as f:
                    f.write(file_source)
            elif isinstance(file_source, (str, Path)):
                source_path = Path(file_source)
                if not source_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {source_path}")
                # Copy file to storage
                with open(source_path, 'rb') as src, open(save_path, 'wb') as dst:
                    dst.write(src.read())
            else:
                raise ValueError("file_source must be a file path (str/Path) or bytes.")

            log.info(f"Saved PDF to storage: {save_path}")
            return {
                "document_id": document_id,
                "file_path": str(save_path),
                "filename": filename
            }
        except Exception as e:
            log.error(f"Failed to save PDF: {e}", exc_info=True)
            raise

    def parse_to_markdown(self, file_source: Union[str, Path, bytes]) -> str:
        """
        Parses a PDF file into markdown format.

        The markdown format preserves document structure (headings, lists, tables)
        which provides better context for LLMs in RAG workflows.

        Args:
            file_source: The PDF file as a path or bytes.

        Returns:
            A string containing the extracted content in markdown format.

        Raises:
            ValueError: If the file_source is not a valid type.
            Exception: For any errors during parsing.
        """
        log.info(f"Parsing PDF to markdown. Source type: {type(file_source).__name__}")

        try:
            if isinstance(file_source, bytes):
                # pymupdf4llm can handle bytes via fitz
                doc = fitz.open(stream=file_source, filetype="pdf")
                md_text = pymupdf4llm.to_markdown(doc)
                doc.close()
            elif isinstance(file_source, (str, Path)):
                file_path = Path(file_source)
                if not file_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {file_path}")
                md_text = pymupdf4llm.to_markdown(str(file_path))
            else:
                raise ValueError("file_source must be a file path (str/Path) or bytes.")

            if not md_text.strip():
                log.warning("PDF parsing resulted in empty content.")
                return ""

            log.info(f"Successfully parsed PDF to markdown. Length: {len(md_text)} characters.")
            return md_text

        except Exception as e:
            log.error(f"Failed to parse PDF to markdown: {e}", exc_info=True)
            raise

    def parse_to_blocks(self, file_source: Union[str, Path, bytes]) -> List[Dict[str, Any]]:
        """
        Extracts text from a PDF as structured blocks with metadata.

        Each block contains text content and metadata including page number,
        bounding box, block type, and estimated token count.

        Args:
            file_source: The PDF file as a path or bytes.

        Returns:
            A list of dictionaries, each containing 'text' and 'metadata'.

        Raises:
            ValueError: If the file_source is not a valid type.
            Exception: For any errors during parsing.
        """
        log.info(f"Extracting text blocks from PDF. Source type: {type(file_source).__name__}")

        try:
            if isinstance(file_source, bytes):
                doc = fitz.open(stream=file_source, filetype="pdf")
            elif isinstance(file_source, (str, Path)):
                file_path = Path(file_source)
                if not file_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {file_path}")
                doc = fitz.open(str(file_path))
            else:
                raise ValueError("file_source must be a file path (str/Path) or bytes.")

            text_blocks = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("blocks")

                for block in blocks:
                    # Block format: (x0, y0, x1, y1, text, block_no, block_type)
                    text = block[4].strip() if len(block) > 4 else ""
                    if text:
                        text_blocks.append({
                            "text": text,
                            "metadata": {
                                "page_number": page_num + 1,
                                "block_number": block[5] if len(block) > 5 else 0,
                                "block_type": "text" if (len(block) > 6 and block[6] == 0) else "image",
                                "bounding_box": list(block[:4]),
                                "token_count": self._count_tokens(text)
                            }
                        })

            doc.close()
            log.info(f"Extracted {len(text_blocks)} text blocks from PDF.")
            return text_blocks

        except Exception as e:
            log.error(f"Failed to extract text blocks from PDF: {e}", exc_info=True)
            raise

    def extract_images(
        self,
        file_source: Union[str, Path, bytes],
        document_id: Optional[str] = None,
        min_width: int = 50,
        min_height: int = 50,
    ) -> Dict[str, Any]:
        """
        Extracts all images from a PDF and saves them to long-term storage.

        Args:
            file_source: The PDF file as a path or bytes.
            document_id: Optional document ID for organizing images. Generated if not provided.
            min_width: Minimum image width to extract (filters out tiny images).
            min_height: Minimum image height to extract.

        Returns:
            Dict containing:
                - document_id: The document identifier.
                - output_dir: Path to the directory containing extracted images.
                - images: List of dicts describing each extracted image.
                - total_images: Total number of images extracted.

        Raises:
            ValueError: If the file_source is not a valid type.
            Exception: For any errors during extraction.
        """
        if document_id is None:
            document_id = self._generate_document_id()

        log.info(f"Extracting images from PDF. Document ID: {document_id}")

        # Create output directory for this document's images
        output_dir = self.image_storage / document_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(file_source, bytes):
                doc = fitz.open(stream=file_source, filetype="pdf")
            elif isinstance(file_source, (str, Path)):
                file_path = Path(file_source)
                if not file_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {file_path}")
                doc = fitz.open(str(file_path))
            else:
                raise ValueError("file_source must be a file path (str/Path) or bytes.")

            image_info_list = []
            total_images = 0

            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images(full=True)
                log.debug(f"Page {page_index + 1}: {len(image_list)} image(s) found")

                for image_index, img in enumerate(image_list, start=1):
                    xref = img[0]  # Unique XREF for the image in the PDF

                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Open image to get dimensions and validate
                        image = Image.open(io.BytesIO(image_bytes))

                        # Filter out tiny images (icons, bullets, etc.)
                        if image.width < min_width or image.height < min_height:
                            log.debug(f"Skipping small image: {image.width}x{image.height}")
                            continue

                        # Save image
                        save_filename = f"page_{page_index + 1}_img_{image_index}.{image_ext}"
                        save_path = output_dir / save_filename
                        image.save(save_path)

                        total_images += 1
                        image_info = {
                            "page_number": page_index + 1,
                            "image_index": image_index,
                            "file_path": str(save_path),
                            "filename": save_filename,
                            "width": image.width,
                            "height": image.height,
                            "extension": image_ext,
                            "xref": xref,
                        }
                        image_info_list.append(image_info)
                        log.debug(f"Saved: {save_path}")

                    except Exception as img_error:
                        log.warning(f"Failed to extract image {image_index} on page {page_index + 1}: {img_error}")
                        continue

            doc.close()
            log.info(f"Extracted {total_images} images from PDF to {output_dir}")

            return {
                "document_id": document_id,
                "output_dir": str(output_dir),
                "images": image_info_list,
                "total_images": total_images,
            }

        except Exception as e:
            log.error(f"Failed to extract images from PDF: {e}", exc_info=True)
            raise

    def parse_full(
        self,
        file_source: Union[str, Path, bytes],
        filename: Optional[str] = None,
        extract_images: bool = True,
        min_image_width: int = 50,
        min_image_height: int = 50,
    ) -> Dict[str, Any]:
        """
        Performs full PDF parsing: saves the PDF, extracts text (markdown + blocks), and extracts images.

        This is the main method for complete PDF processing, suitable for RAG pipelines.

        Args:
            file_source: The PDF file as a path or bytes.
            filename: Optional filename for saving the PDF.
            extract_images: Whether to extract images from the PDF.
            min_image_width: Minimum image width to extract.
            min_image_height: Minimum image height to extract.

        Returns:
            Dict containing:
                - document_id: Unique identifier for this document.
                - pdf_info: Info about the saved PDF file.
                - markdown: Extracted text in markdown format.
                - text_blocks: Structured text blocks with metadata.
                - images: Image extraction results (if extract_images=True).
                - stats: Processing statistics.
        """
        log.info("Starting full PDF parsing...")

        # Generate document ID
        document_id = self._generate_document_id()

        # If file_source is a path, read it to bytes for consistent processing
        if isinstance(file_source, (str, Path)):
            file_path = Path(file_source)
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            if filename is None:
                filename = file_path.name
        elif isinstance(file_source, bytes):
            pdf_bytes = file_source
        else:
            raise ValueError("file_source must be a file path (str/Path) or bytes.")

        result = {
            "document_id": document_id,
            "pdf_info": None,
            "markdown": "",
            "text_blocks": [],
            "images": None,
            "stats": {}
        }

        try:
            # 1. Save PDF to storage
            pdf_info = self.save_pdf(pdf_bytes, filename=filename or f"{document_id}.pdf")
            pdf_info["document_id"] = document_id  # Ensure consistent document_id
            result["pdf_info"] = pdf_info
            log.info(f"PDF saved: {pdf_info['file_path']}")

            # 2. Extract text as markdown
            markdown = self.parse_to_markdown(pdf_bytes)
            result["markdown"] = markdown
            result["stats"]["markdown_length"] = len(markdown)
            result["stats"]["estimated_tokens"] = self._count_tokens(markdown)

            # 3. Extract text blocks
            text_blocks = self.parse_to_blocks(pdf_bytes)
            result["text_blocks"] = text_blocks
            result["stats"]["total_blocks"] = len(text_blocks)

            # 4. Extract images (if enabled)
            if extract_images:
                images_result = self.extract_images(
                    pdf_bytes,
                    document_id=document_id,
                    min_width=min_image_width,
                    min_height=min_image_height,
                )
                result["images"] = images_result
                result["stats"]["total_images"] = images_result["total_images"]

            log.info(f"Full PDF parsing complete. Document ID: {document_id}")
            return result

        except Exception as e:
            log.error(f"Failed to complete full PDF parsing: {e}", exc_info=True)
            raise
