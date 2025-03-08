import os
import uuid
import json
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from api.schemas import UploadResponse
from services.db_service import get_db_client, get_collection
from services.media_service import process_media_file
from models.model_adapter import ModelAdapter

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_media(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    collection_name: str = Form(...)
):
    """Upload media file (image or video) and store in ChromaDB"""
    try:
        # Check if collection exists, create if not
        collection = get_collection(collection_name)
        
        # Parse metadata
        metadata_dict = json.loads(metadata)
        
        # Generate unique ID for the media
        media_id = str(uuid.uuid4())
        
        # Save file to disk
        file_content = await file.read()
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_path = f"data/uploads/{media_id}{file_extension}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Process media file
        extracted_text = process_media_file(file_content, file_extension)
        
        # Combine metadata with extracted text
        combined_text = f"{metadata_dict.get('description', '')} {extracted_text} {' '.join(metadata_dict.get('tags', []))}"
        
        # Generate embedding
        model_adapter = ModelAdapter(model_type="gemini")
        embedding = model_adapter.get_embedding(combined_text)
        
        # Store in ChromaDB
        collection.add(
            ids=[media_id],
            embeddings=[embedding],
            metadatas=[{
                "filename": file.filename,
                "path": file_path,
                "file_type": file.content_type,
                "description": metadata_dict.get("description", ""),
                "tags": metadata_dict.get("tags", []),
                "extracted_text": extracted_text
            }]
        )
        
        return {
            "message": "Media uploaded and indexed successfully",
            "media_id": media_id,
            "file_path": file_path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")