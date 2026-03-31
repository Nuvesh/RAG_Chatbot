from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ApiKeyRequest(BaseModel):
    api_key: str


class CohereController:
    def __init__(self, app, uploads_dir, storage_dir):
        self.router = APIRouter(prefix="/cohere", tags=["cohere"])
        self.uploads_dir = uploads_dir
        self.storage_dir = storage_dir

        from services.cohere_service import CohereService
        from services.file_service import FileService

        self.cohere_service = CohereService(uploads_dir, storage_dir)
        self.file_service = FileService(uploads_dir, storage_dir)

        self.register_routes()

    def register_routes(self):
        @self.router.post("/set-api-key")
        async def set_api_key(payload: ApiKeyRequest):
            api_key = payload.api_key.strip()
            if not api_key:
                raise HTTPException(status_code=400, detail="API key cannot be empty.")
            self.cohere_service.set_api_key(api_key)
            return {"message": "Cohere API key configured"}

        @self.router.post("/upload")
        async def upload_pdf(file: UploadFile = File(...)):
            logger.info(f"📤 Cohere upload received: {file.filename}")
            try:
                file_path = await self.file_service.save_file(file)
                chunk_count = await self.cohere_service.index_document(file_path)
                return {
                    "message": "Indexed successfully",
                    "filename": file.filename,
                    "chunks": chunk_count,
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

        @self.router.get("/ask")
        async def ask_question(q: str = Query(...)):
            if not q or not q.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Query cannot be empty. Please provide a valid question.",
                )
            try:
                answer, sources = await self.cohere_service.query_index(q)
                return {"answer": answer, "sources": sources}
            except FileNotFoundError:
                raise HTTPException(
                    status_code=400,
                    detail="No document indexed yet. Please upload a PDF first.",
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

        @self.router.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "has_index": self.cohere_service.has_index(),
                "has_api_key": bool(self.cohere_service.api_key),
            }

    def include_router(self, app):
        app.include_router(self.router)
