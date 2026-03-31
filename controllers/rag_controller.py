# RAG Controller - Handles all RAG-related endpoints

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import List
import logging

logger = logging.getLogger(__name__)


class RagController:
    def __init__(self, app, uploads_dir, storage_dir):
        self.router = APIRouter()
        self.uploads_dir = uploads_dir
        self.storage_dir = storage_dir
        
        # Import services
        from services.rag_service import RagService
        from services.file_service import FileService
        
        self.rag_service = RagService(uploads_dir, storage_dir)
        self.file_service = FileService(uploads_dir, storage_dir)
        
        # Register routes
        self.register_routes()
    
    def register_routes(self):
        """Register all routes for this controller"""
        
        @self.router.post("/upload")
        async def upload_pdf(file: UploadFile = File(...)):
            """
            Upload and index a PDF document.
            
            Accepts: multipart/form-data with key="file"
            Rejects: non-PDF files (HTTP 400)
            Clears previous index (overwrite behavior)
            """
            logger.info(f"📤 Upload received: {file.filename}")
            
            try:
                # Validate and save file
                file_path = await self.file_service.save_file(file)
                
                # Index the document
                chunk_count = await self.rag_service.index_document(file_path)
                
                logger.info(f"✅ Indexed successfully: {file.filename} ({chunk_count} chunks)")
                
                return {
                    "message": "Indexed successfully",
                    "filename": file.filename,
                    "chunks": chunk_count
                }
                
            except ValueError as e:
                logger.error(f"❌ Validation error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"❌ Upload failed: {e}")
                raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
        
        @self.router.get("/ask")
        async def ask_question(q: str = Query(...)):
            """
            Ask a question about the indexed document.
            
            Validates query, returns 400 if empty.
            Returns answer and top 3 source chunks.
            """
            logger.info(f"📨 Query received: '{q}'")
            
            # Validate query
            if not q or not q.strip():
                logger.error("❌ Empty query received")
                raise HTTPException(
                    status_code=400,
                    detail="Query cannot be empty. Please provide a valid question."
                )
            
            try:
                # Query the index
                answer, sources = await self.rag_service.query_index(q)
                
                logger.info(f"✅ Response generated for query: '{q}'")
                
                return {
                    "answer": answer,
                    "sources": sources
                }
                
            except FileNotFoundError as e:
                logger.error(f"❌ No index found: {e}")
                raise HTTPException(
                    status_code=400,
                    detail="No document indexed yet. Please upload a PDF first."
                )
            except Exception as e:
                logger.error(f"❌ Query failed: {e}")
                raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
        
        @self.router.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "has_index": self.rag_service.has_index()
            }
    
    def include_router(self, app):
        """Include this controller's router in the main app"""
        app.include_router(self.router)
