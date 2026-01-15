from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging

from app.api.routes_chat import router as chat_router
from app.api.routes_core_services import router as core_services_router

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Modal RAG API",
    description="API for multi-modal Retrieval Augmented Generation.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Mount static files for serving stored PDFs and images
storage_path = Path("storage")
if not storage_path.exists():
    storage_path.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage")

# Include API routers
app.include_router(chat_router, prefix="/api/v1", tags=["RAG Chat"])
app.include_router(core_services_router, prefix="/api/v1", tags=["Core Services"])


@app.get("/")
async def root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Multi-Modal RAG API! Visit /docs for API documentation."}


log.info("FastAPI application initialized and routers included.")
