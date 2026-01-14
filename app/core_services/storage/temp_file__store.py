"""
Temporary File Store
====================

This module provides a simple service for temporarily storing uploaded files.
It saves files to a designated temporary directory with unique filenames to prevent
collisions.
"""

import logging
import uuid
from pathlib import Path
from fastapi import UploadFile
from typing import List

log = logging.getLogger(__name__)

class TempFileStore:
    """
    Manages the storage of temporary files.
    """
    def __init__(self, base_dir: str = "storage/tmp"):
        """
        Initializes the TempFileStore.

        Args:
            base_dir: The directory where temporary files will be saved.
                      Defaults to "storage/tmp".
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Temporary file store initialized at: {self.base_dir.resolve()}")

    def save(self, file: UploadFile) -> str:
        """
        Saves a single uploaded file to the temporary storage directory.

        Args:
            file: The `UploadFile` object from a FastAPI request.

        Returns:
            The string path to the newly saved file.

        Raises:
            IOError: If the file cannot be saved.
        """
        try:
            # Create a unique filename to avoid collisions
            file_extension = Path(file.filename).suffix if file.filename else ".tmp"
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = self.base_dir / unique_filename

            with file_path.open("wb") as buffer:
                content = file.file.read()
                buffer.write(content)

            log.info(f"Saved temporary file '{file.filename}' to '{file_path}'")
            return str(file_path)
        except Exception as e:
            log.error(f"Failed to save temporary file: {e}", exc_info=True)
            raise IOError(f"Could not save file: {e}") from e

    def save_multiple(self, files: List[UploadFile]) -> List[str]:
        """
        Saves multiple uploaded files.

        Args:
            files: A list of `UploadFile` objects.

        Returns:
            A list of paths to the saved files.
        """
        saved_paths = []
        for file in files:
            path = self.save(file)
            saved_paths.append(path)
        return saved_paths