"""
RAG Studio - FastAPI Backend for PDF-based Q&A System
Uses Gemini LLM and FAISS for Retrieval-Augmented Generation
MVC Architecture
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Studio",
    description="PDF-based Q&A using RAG with Gemini and FAISS",
    version="1.0.0"
)

# Enable CORS (allow all origins for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
STORAGE_DIR = BASE_DIR / "storage"

UPLOADS_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)

logger.info(f"📁 Uploads directory: {UPLOADS_DIR}")
logger.info(f"📁 Storage directory: {STORAGE_DIR}")

# Import controllers after app initialization to avoid circular imports
from controllers.rag_controller import RagController
from controllers.cohere_controller import CohereController

# Initialize Gemini controller
rag_controller = RagController(app, UPLOADS_DIR, STORAGE_DIR)
rag_controller.include_router(app)  # Include the router in the main app

# Initialize Cohere controller
cohere_controller = CohereController(app, UPLOADS_DIR, STORAGE_DIR)
cohere_controller.include_router(app)  # Include the router in the main app

# Check for API keys (optional - can be set later)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    rag_controller.rag_service.set_api_key(GEMINI_API_KEY)
    logger.info("✅ GEMINI_API_KEY configured at startup")
else:
    logger.warning("⚠️  GEMINI_API_KEY not set. Set it before uploading documents.")
    logger.warning("   Export: $env:GEMINI_API_KEY='your-key-here' (PowerShell)")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if COHERE_API_KEY:
    cohere_controller.cohere_service.set_api_key(COHERE_API_KEY)
    logger.info("✅ COHERE_API_KEY configured at startup")
else:
    logger.warning("⚠️  COHERE_API_KEY not set. Set it before uploading documents.")
    logger.warning("   Export: $env:COHERE_API_KEY='your-key-here' (PowerShell)")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML file"""
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(html_path))


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting RAG Studio server...")
    # With reload=True, Uvicorn expects an import string target.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
