"""
Pydantic Schemas for LLM Service
=================================

This module defines the Pydantic models for standardizing the input and output
of the LLM generation services.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict

class LLMRequest(BaseModel):
    """
    Defines the standardized input for an LLM generation request.
    """
    user_prompt: str = Field(
        ...,
        description="The main prompt or query from the user."
    )
    model_id: str = Field(
        ...,
        description="The identifier for the desired LLM model (e.g., 'cohere_command_r_plus')."
    )
    images: Optional[List[str]] = Field(
        default=None,
        description="A list of image paths or base64 encoded data URLs for multi-modal requests."
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="The sampling temperature for generation, controlling randomness."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_prompt": "What are the main ingredients in a margarita?",
                "model_id": "groq_llama3_8b",
                "temperature": 0.5
            },
            "example_multimodal": {
                "user_prompt": "What city is shown in this image?",
                "model_id": "groq_multimodal_model", # Example ID
                "images": ["/path/to/image.jpg"]
            }
        }

class LLMResponse(BaseModel):
    """
    Defines the standardized output from an LLM generation request.
    """
    response: Dict[str, Any] = Field(
        ...,
        description="The parsed JSON response content from the language model."
    )
    model_id: str = Field(
        ...,
        description="The identifier of the model that generated the response."
    )
    # The usage field can be populated with token information from the LLM API response.
    usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Token usage information (e.g., prompt_tokens, completion_tokens)."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": {
                    "ingredients": ["tequila", "lime juice", "triple sec", "salt"]
                },
                "model_id": "groq_llama3_8b",
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 50,
                    "total_tokens": 75
                }
            }
        }
