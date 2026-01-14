"""RAG Text Generator
====================

A text generator for RAG that combines retrieved context with user queries
to generate accurate, context-grounded responses using LLMs.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from app.core_services.llm_clients.llm_factory import LLMFactory
from app.schemas.llm import LLMRequest

log = logging.getLogger(__name__)

# Default model for RAG generation
DEFAULT_MODEL_ID = "groq_maverick"
DEFAULT_TEMPERATURE = 0.3

# System prompt optimized for RAG
# Note: Double curly braces {{ }} are escaped for Python .format()
DEFAULT_SYSTEM_PROMPT = """You are an advanced AI assistant specialized in answering questions based solely on the provided context.

RULES:
1. Answer ONLY using information from the retrieved context below.
2. If the answer is not present in the context, respond with: {{"answer": null, "reason": "Information not found in the provided context"}}
3. Always respond in valid JSON format with the structure:
   {{
     "answer": "your detailed answer here",
     "confidence": "high|medium|low",
     "sources": ["chunk_id_1", "chunk_id_2"]
   }}
4. Be concise, accurate, and cite which chunks you used.
5. Do not make up information or use knowledge outside the provided context.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

Respond with valid JSON only:"""


class RAGTextGenerator:
    """
    A text generator for RAG that uses LLMs to answer questions based on retrieved context.

    This generator:
    1. Takes a user query and retrieved context chunks
    2. Constructs a prompt with system instructions and context
    3. Generates a response using the specified LLM
    4. Returns structured JSON output
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the RAGTextGenerator.

        Args:
            system_prompt: Custom system prompt. If None, uses DEFAULT_SYSTEM_PROMPT.
        """
        log.info("Initializing RAGTextGenerator...")

        self.llm_factory = LLMFactory()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        log.info("RAGTextGenerator initialized successfully.")

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved context.

        Args:
            query: The user's question.
            retrieved_chunks: List of retrieved chunks with 'id', 'text', and metadata.
            model_id: The LLM model to use. Default: 'groq_maverick'.
            temperature: Generation temperature. Default: 0.3 (lower for factual).

        Returns:
            Dict containing:
                - query: Original query
                - answer: Generated answer (parsed from JSON if possible)
                - raw_response: Raw LLM response
                - model_id: Model used
                - context_used: Chunks provided as context
                - usage: Token usage info (if available)
        """
        log.info(f"Generating response for query: '{query[:50]}...' using model: {model_id}")

        result = {
            "query": query,
            "answer": None,
            "raw_response": "",
            "model_id": model_id,
            "context_used": [],
            "usage": None,
        }

        try:
            # Build context string from retrieved chunks
            context_str = self._build_context_string(retrieved_chunks)
            result["context_used"] = [
                {"id": chunk.get("id"), "text_preview": chunk.get("text", "")[:100]}
                for chunk in retrieved_chunks
            ]

            # Build the final prompt
            prompt = self.system_prompt.format(
                context=context_str,
                query=query,
            )

            log.debug(f"Prompt length: {len(prompt)} characters")

            # Call LLM
            llm_request = LLMRequest(
                user_prompt=prompt,
                model_id=model_id,
                temperature=temperature,
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

            # Try to parse JSON from response
            result["answer"] = self._parse_json_response(raw_response)

            log.info("Response generated successfully.")
            return result

        except Exception as e:
            log.error(f"Generation failed: {e}", exc_info=True)
            raise

    def _build_context_string(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build a formatted context string from retrieved chunks.

        Args:
            chunks: List of chunk dictionaries.

        Returns:
            Formatted string with all chunks.
        """
        if not chunks:
            return "No context available."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("id", f"chunk_{i}")
            chunk_text = chunk.get("text", "")

            # Include metadata if available
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

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.

        Attempts to extract and parse JSON from the response.
        Falls back to returning the raw text if parsing fails.

        Args:
            response: Raw LLM response string.

        Returns:
            Parsed JSON dict or fallback dict with raw answer.
        """
        if not response:
            return {"answer": None, "reason": "Empty response from LLM"}

        # Try direct JSON parse
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        import re
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
            r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
            r'\{[\s\S]*\}',                   # Raw JSON object
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str.strip())
                except (json.JSONDecodeError, IndexError):
                    continue

        # Fallback: return raw response as answer
        log.warning("Could not parse JSON from response. Returning raw text.")
        return {
            "answer": response.strip(),
            "confidence": "unknown",
            "sources": [],
            "parse_error": "Response was not valid JSON"
        }

    def get_available_models(self) -> List[str]:
        """
        Get list of available LLM models.

        Returns:
            List of model IDs that can be used for generation.
        """
        return [
            "groq_maverick",      # Default, multi-modal
            "groq_llama3_8b",     # Fast, text-only
            "groq_gemma_7b",      # Text-only
            "groq_mixtral_8x7b",  # Text-only
            "cohere_command_r_plus",  # Cohere text
            "cohere_command_a",   # Cohere text
        ]
