"""PDF Highlighter
=================

A utility for highlighting text in PDF files using PyMuPDF.
Creates temporary copies of PDFs with highlighted text for display purposes.
"""

import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import fitz

log = logging.getLogger(__name__)


class PDFHighlighter:
    """
    A utility for highlighting text in PDF files.

    Creates temporary copies of PDFs and adds highlight annotations
    to specified text on given pages. Useful for showing source
    references in RAG applications.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the PDFHighlighter.

        Args:
            temp_dir: Optional custom directory for temporary files.
                      If not provided, uses system temp directory.
        """
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "pdf_highlights"
            self.temp_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"PDFHighlighter initialized. Temp directory: {self.temp_dir}")

    def _generate_temp_filename(self, original_path: Path) -> str:
        """Generate a unique filename for the temporary highlighted PDF."""
        unique_id = str(uuid.uuid4())[:8]
        return f"{original_path.stem}_highlighted_{unique_id}.pdf"

    def highlight_text(
        self,
        pdf_path: str,
        page_number: int,
        search_text: str,
        highlight_color: Tuple[float, float, float] = (1, 1, 0),
    ) -> Dict[str, Any]:
        """
        Highlights text on a specific page of a PDF.

        Creates a temporary copy of the PDF and adds highlight annotations
        for all instances of the search text on the specified page.

        Args:
            pdf_path: Path to the original PDF file.
            page_number: The page number to highlight on (0-indexed).
            search_text: The text to search for and highlight.
            highlight_color: RGB color tuple for highlight (values 0-1).
                           Default is yellow (1, 1, 0).

        Returns:
            Dict containing:
                - success: Whether highlighting was successful.
                - highlighted_pdf_path: Path to the temporary highlighted PDF.
                - page_number: The page number that was highlighted.
                - instances_found: Number of text instances found and highlighted.
                - error: Error message if unsuccessful.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If page number is out of range.
        """
        log.info(f"Highlighting text '{search_text[:50]}...' on page {page_number} of {pdf_path}")

        original_path = Path(pdf_path)

        # Validate file exists
        if not original_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create temporary copy
        temp_filename = self._generate_temp_filename(original_path)
        temp_path = self.temp_dir / temp_filename

        try:
            # Copy the original PDF to temp location
            shutil.copy2(original_path, temp_path)
            log.debug(f"Created temporary copy at: {temp_path}")

            # Open the temporary copy and add highlights
            doc = fitz.open(str(temp_path))

            # Validate page number
            if page_number < 0 or page_number >= doc.page_count:
                doc.close()
                temp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Page number {page_number} is out of range. "
                    f"PDF has {doc.page_count} pages (0-indexed)."
                )

            page = doc.load_page(page_number)

            # Search for the text and get rectangles where it occurs
            text_instances = page.search_for(search_text)
            instances_found = len(text_instances)

            if instances_found == 0:
                log.warning(f"No instances of '{search_text[:50]}...' found on page {page_number}")
            else:
                # Add highlight annotations for each instance
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors({"stroke": highlight_color})
                    highlight.update()

                log.info(f"Added {instances_found} highlight(s) on page {page_number}")

            # Save the modified document
            doc.save(str(temp_path), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
            doc.close()

            return {
                "success": True,
                "highlighted_pdf_path": str(temp_path),
                "page_number": page_number,
                "instances_found": instances_found,
                "error": None,
            }

        except Exception as e:
            log.error(f"Failed to highlight text in PDF: {e}", exc_info=True)
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return {
                "success": False,
                "highlighted_pdf_path": None,
                "page_number": page_number,
                "instances_found": 0,
                "error": str(e),
            }

    def highlight_multiple_texts(
        self,
        pdf_path: str,
        highlights: list,
        highlight_color: Tuple[float, float, float] = (1, 1, 0),
    ) -> Dict[str, Any]:
        """
        Highlights multiple text instances across different pages.

        Creates a single temporary copy with all highlights applied.

        Args:
            pdf_path: Path to the original PDF file.
            highlights: List of dicts with 'page_number' and 'search_text' keys.
            highlight_color: RGB color tuple for highlight (values 0-1).

        Returns:
            Dict containing:
                - success: Whether highlighting was successful.
                - highlighted_pdf_path: Path to the temporary highlighted PDF.
                - highlights_applied: List of results for each highlight request.
                - total_instances: Total number of text instances highlighted.
                - error: Error message if unsuccessful.
        """
        log.info(f"Highlighting {len(highlights)} text(s) in {pdf_path}")

        original_path = Path(pdf_path)

        if not original_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        temp_filename = self._generate_temp_filename(original_path)
        temp_path = self.temp_dir / temp_filename

        try:
            shutil.copy2(original_path, temp_path)
            doc = fitz.open(str(temp_path))

            highlights_applied = []
            total_instances = 0

            for item in highlights:
                page_number = item.get("page_number", 0)
                search_text = item.get("search_text", "")

                if not search_text:
                    highlights_applied.append({
                        "page_number": page_number,
                        "search_text": search_text,
                        "instances_found": 0,
                        "error": "Empty search text",
                    })
                    continue

                if page_number < 0 or page_number >= doc.page_count:
                    highlights_applied.append({
                        "page_number": page_number,
                        "search_text": search_text,
                        "instances_found": 0,
                        "error": f"Page {page_number} out of range",
                    })
                    continue

                page = doc.load_page(page_number)
                text_instances = page.search_for(search_text)
                instances_found = len(text_instances)

                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors({"stroke": highlight_color})
                    highlight.update()

                total_instances += instances_found
                highlights_applied.append({
                    "page_number": page_number,
                    "search_text": search_text[:50],
                    "instances_found": instances_found,
                    "error": None,
                })

            doc.save(str(temp_path), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
            doc.close()

            log.info(f"Applied {total_instances} highlight(s) across {len(highlights)} request(s)")

            return {
                "success": True,
                "highlighted_pdf_path": str(temp_path),
                "highlights_applied": highlights_applied,
                "total_instances": total_instances,
                "error": None,
            }

        except Exception as e:
            log.error(f"Failed to highlight multiple texts in PDF: {e}", exc_info=True)
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return {
                "success": False,
                "highlighted_pdf_path": None,
                "highlights_applied": [],
                "total_instances": 0,
                "error": str(e),
            }

    def cleanup_temp_file(self, temp_path: str) -> bool:
        """
        Removes a temporary highlighted PDF file.

        Args:
            temp_path: Path to the temporary file to remove.

        Returns:
            True if file was removed, False otherwise.
        """
        try:
            path = Path(temp_path)
            if path.exists() and path.parent == self.temp_dir:
                path.unlink()
                log.debug(f"Cleaned up temporary file: {temp_path}")
                return True
            return False
        except Exception as e:
            log.warning(f"Failed to clean up temp file {temp_path}: {e}")
            return False

    def cleanup_all_temp_files(self) -> int:
        """
        Removes all temporary highlighted PDF files.

        Returns:
            Number of files removed.
        """
        count = 0
        try:
            for file in self.temp_dir.glob("*_highlighted_*.pdf"):
                file.unlink()
                count += 1
            log.info(f"Cleaned up {count} temporary file(s)")
        except Exception as e:
            log.warning(f"Error during cleanup: {e}")
        return count


# Usage example

#   Usage example:

#   from app.core_services.file_parsers import PDFHighlighter

#   highlighter = PDFHighlighter()

#   result = highlighter.highlight_text(
#       pdf_path="/path/to/original.pdf",
#       page_number=0,  # 0-indexed
#       search_text="text to highlight"
#   )

#   if result["success"]:
#       print(f"Highlighted PDF: {result['highlighted_pdf_path']}")
#       print(f"Page: {result['page_number']}")
#       print(f"Instances found: {result['instances_found']}")
