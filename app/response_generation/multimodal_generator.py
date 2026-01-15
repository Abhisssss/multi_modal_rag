"""Multi-Modal RAG Generator
============================

A generator for multi-modal RAG that combines retrieved text context with
retrieved images to generate answers using vision-capable LLMs.
Images are fetched via vector search, not uploaded by users.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core_services.llm_clients.llm_factory import LLMFactory
from app.schemas.llm import LLMRequest
from app.utils.files import encode_image_to_data_url

log = logging.getLogger(__name__)

# Multi-modal model (only groq_maverick supports images)
MULTIMODAL_MODEL_ID = "groq_maverick"
DEFAULT_TEMPERATURE = 0.3
MAX_IMAGES = 5

# System prompt for multi-modal RAG
# Note: Double curly braces {{ }} are escaped for Python .format()
MULTIMODAL_SYSTEM_PROMPT = """You are an advanced AI assistant that answers questions using both text context and images retrieved from a knowledge base.

RULES:
1. Answer using information from BOTH the retrieved text context AND the provided images.
2. When referencing images in your answer, use the format: "As shown in Image 1, ..." or "Image 2 demonstrates..."
3. If the answer requires visual explanation, describe what's visible in the relevant image.
4. Combine insights from both text and images when applicable.
5. Always respond in valid JSON format:
   {{
     "answer": "your detailed answer with image references like 'As shown in Image 1...'",
     "confidence": "high|medium|low",
     "sources": ["chunk_id_1", "chunk_id_2"],
     "images_referenced": ["Image 1", "Image 2"]
   }}
6. If the answer is not present in the context or images, respond with:
   {{"answer": null, "reason": "Information not found in the provided context or images"}}
7. Be concise, accurate, and always cite your sources (both text chunks and images).

RETRIEVED TEXT CONTEXT:
{context}

RETRIEVED IMAGES:
{image_descriptions}

USER QUESTION: {query}

Respond with valid JSON only:"""


class MultiModalRAGGenerator:
    """
    A generator for multi-modal RAG using retrieved text and images.

    This generator:
    1. Takes retrieved text chunks and retrieved images (from vector search)
    2. Loads images from their stored paths
    3. Constructs a multi-modal prompt with text and image content
    4. Generates a response using groq_maverick (vision-capable LLM)
    5. Returns structured JSON output with image references
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the MultiModalRAGGenerator.

        Args:
            system_prompt: Custom system prompt. If None, uses MULTIMODAL_SYSTEM_PROMPT.
        """
        log.info("Initializing MultiModalRAGGenerator...")

        self.llm_factory = LLMFactory()
        self.system_prompt = system_prompt or MULTIMODAL_SYSTEM_PROMPT
        self.model_id = MULTIMODAL_MODEL_ID

        log.info(f"MultiModalRAGGenerator initialized. Model: {self.model_id}")

    def generate(
        self,
        query: str,
        text_chunks: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Dict[str, Any]:
        """
        Generate a response based on query, retrieved text chunks, and retrieved images.

        Args:
            query: The user's question.
            text_chunks: List of retrieved text chunks with 'id', 'text', and metadata.
            images: List of retrieved images with 'id', 'image_path', and metadata.
            temperature: Generation temperature. Default: 0.3.

        Returns:
            Dict containing:
                - query: Original query
                - answer: Generated answer with image references
                - raw_response: Raw LLM response
                - model_id: Model used (groq_maverick)
                - context_used: Text chunks provided as context
                - images_used: Images sent to LLM
                - usage: Token usage info (if available)
        """
        log.info(
            f"Generating multi-modal response for query: '{query[:50]}...' "
            f"with {len(text_chunks)} text chunks and {len(images)} images"
        )

        # Limit images to MAX_IMAGES
        if len(images) > MAX_IMAGES:
            log.warning(f"Too many images ({len(images)}). Limiting to {MAX_IMAGES}.")
            images = images[:MAX_IMAGES]

        result = {
            "query": query,
            "answer": None,
            "raw_response": "",
            "model_id": self.model_id,
            "context_used": [],
            "images_used": [],
            "usage": None,
        }

        try:
            # Build text context string
            context_str = self._build_context_string(text_chunks)
            result["context_used"] = [
                {"id": chunk.get("id"), "text_preview": chunk.get("text", "")[:100]}
                for chunk in text_chunks
            ]

            # Build image descriptions and load images
            image_descriptions, image_data_urls, valid_images = self._prepare_images(images)

            result["images_used"] = [
                {
                    "id": img.get("id"),
                    "image_path": img.get("image_path"),
                    "page_number": img.get("page_number", 1),
                }
                for img in valid_images
            ]

            # Build the text part of the prompt
            prompt_text = self.system_prompt.format(
                context=context_str,
                image_descriptions=image_descriptions,
                query=query,
            )

            log.debug(f"Prompt text length: {len(prompt_text)} characters")
            log.debug(f"Number of images to send: {len(image_data_urls)}")

            # Call LLM with multi-modal content
            llm_request = LLMRequest(
                user_prompt=prompt_text,
                model_id=self.model_id,
                temperature=temperature,
                images=image_data_urls if image_data_urls else None,
            )

            llm_response = self.llm_factory.generate(llm_request)

            # Extract response text
            raw_response = ""
            if llm_response.response:
                if isinstance(llm_response.response, dict):
                    raw_response = llm_response.response.get("text", str(llm_response.response))
                else:
                    raw_response = str(llm_response.response)

            result["raw_response"] = raw_response
            result["usage"] = llm_response.usage

            # Parse JSON from response
            result["answer"] = self._parse_json_response(raw_response)

            log.info("Multi-modal response generated successfully.")
            return result

        except Exception as e:
            log.error(f"Multi-modal generation failed: {e}", exc_info=True)
            raise

    def _build_context_string(self, chunks: List[Dict[str, Any]]) -> str:
        """Build a formatted context string from retrieved text chunks."""
        if not chunks:
            return "No text context available."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("id", f"chunk_{i}")
            chunk_text = chunk.get("text", "")

            metadata = chunk.get("metadata", {})
            page_num = metadata.get("page_number", "N/A")
            filename = metadata.get("filename", "N/A")

            context_part = f"""--- Chunk {i} (ID: {chunk_id}) ---
Source: {filename}, Page: {page_num}
Content:
{chunk_text}
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _prepare_images(
        self, images: List[Dict[str, Any]]
    ) -> tuple:
        """
        Prepare images for the LLM: build descriptions and load as data URLs.

        Args:
            images: List of retrieved image dicts with 'image_path' and metadata.

        Returns:
            Tuple of (image_descriptions_str, image_data_urls_list, valid_images_list)
        """
        if not images:
            return "No images retrieved.", [], []

        descriptions = []
        data_urls = []
        valid_images = []

        for i, img in enumerate(images, 1):
            image_path = img.get("image_path", "")

            if not image_path:
                log.warning(f"Image {i} has no path, skipping.")
                continue

            # Check if file exists
            if not Path(image_path).exists():
                log.warning(f"Image file not found: {image_path}, skipping.")
                continue

            try:
                # Convert to data URL
                data_url = encode_image_to_data_url(image_path)
                data_urls.append(data_url)
                valid_images.append(img)

                # Build description
                filename = img.get("image_filename", Path(image_path).name)
                page_num = img.get("page_number", "N/A")
                doc_filename = img.get("filename", "N/A")

                description = f"Image {len(valid_images)}: {filename} (from {doc_filename}, page {page_num})"
                descriptions.append(description)

            except Exception as e:
                log.warning(f"Failed to load image {image_path}: {e}")
                continue

        if not descriptions:
            return "No valid images could be loaded.", [], []

        return "\n".join(descriptions), data_urls, valid_images

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling various formats."""
        if not response:
            return {"answer": None, "reason": "Empty response from LLM"}

        response = response.strip()

        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str.strip())
                except (json.JSONDecodeError, IndexError):
                    continue

        # Try parsing Python dict format (single quotes, None, True, False)
        try:
            # Find the JSON-like structure
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                dict_str = match.group(0)
                # Convert Python dict to JSON format
                json_str = dict_str.replace("'", '"')
                json_str = re.sub(r'\bNone\b', 'null', json_str)
                json_str = re.sub(r'\bTrue\b', 'true', json_str)
                json_str = re.sub(r'\bFalse\b', 'false', json_str)
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: return raw response as answer
        log.warning("Could not parse JSON from multi-modal response. Returning raw text.")
        return {
            "answer": response,
            "confidence": "medium",
            "sources": [],
            "images_referenced": [],
        }
