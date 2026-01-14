"""
File Utilities
==============

This module provides helper functions for file-related operations, such as
reading and encoding files for multi-modal LLM inputs.
"""

import base64
import mimetypes
import os
import logging

log = logging.getLogger(__name__)

def encode_image_to_data_url(image_path: str) -> str:
    """
    Encodes a local image file into a base64 data URL.

    This function reads an image file, determines its MIME type, and then
    encodes it into the data URL format required by many multi-modal models.

    Args:
        image_path: The local file path to the image.

    Returns:
        A string representing the base64 encoded data URL (e.g., "data:image/jpeg;base64,...").

    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If the MIME type of the file cannot be determined.
    """
    if not os.path.exists(image_path):
        log.error(f"Image file not found at path: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Guess the MIME type of the file
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        log.error(f"Could not determine MIME type for file: {image_path}")
        raise ValueError(f"Could not determine MIME type for: {image_path}")

    try:
        # Read the image file in binary mode
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Format as a data URL
        data_url = f"data:{mime_type};base64,{encoded_string}"
        log.info(f"Successfully encoded image {image_path} to data URL.")
        return data_url

    except Exception as e:
        log.error(f"Failed to encode image file {image_path}: {e}", exc_info=True)
        raise