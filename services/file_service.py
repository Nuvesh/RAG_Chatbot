# File Service - Handles file operations

import os
import shutil
from pathlib import Path
from typing import Optional
import logging
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)


class FileService:
    def __init__(self, uploads_dir: Path, storage_dir: Path):
        self.uploads_dir = uploads_dir
        self.storage_dir = storage_dir
    
    async def save_file(self, file: UploadFile) -> Path:
        """
        Save uploaded file to uploads directory.
        
        Validates:
        - Only PDF files are accepted
        - Clears previous index before saving
        
        Returns:
        - Path to saved file
        """
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are accepted. Please upload a .pdf file.")
        
        # Save file to ./uploads
        file_path = self.uploads_dir / file.filename
        
        logger.info(f"💾 Saving file to {file_path}...")
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"❌ Error saving file: {e}")
            raise Exception(f"Failed to save file: {str(e)}")
        
        # Clear previous index (remove storage directory contents)
        if self.storage_dir.exists():
            logger.info("🗑️ Clearing previous index...")
            for item in self.storage_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        
        logger.info(f"✅ File saved: {file.filename}")
        
        return file_path
    
    def get_file_info(self, filename: str) -> dict:
        """Get information about a file"""
        file_path = self.uploads_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File {filename} not found")
        
        stat = file_path.stat()
        return {
            "filename": filename,
            "size": stat.st_size,
            "path": str(file_path)
        }
    
    def delete_file(self, filename: str) -> bool:
        """Delete a file from uploads directory"""
        file_path = self.uploads_dir / filename
        
        if file_path.exists():
            file_path.unlink()
            logger.info(f"🗑️ Deleted file: {filename}")
            return True
        
        return False
    
    def clear_storage(self) -> None:
        """Clear all files from storage directory"""
        if self.storage_dir.exists():
            for item in self.storage_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info("🗑️ Storage cleared")
