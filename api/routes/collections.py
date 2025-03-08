from fastapi import APIRouter, Form
from api.schemas import CollectionResponse, MessageResponse
from services.db_service import get_db_client

router = APIRouter()

@router.get("/collections", response_model=CollectionResponse)
async def get_collections():
    """Get all available collections"""
    db_client = get_db_client()
    collections = db_client.list_collections() # # Returns a list of names
    # return {"collections": [coll.name for coll in collections]}
    return {"collections": collections}

@router.post("/create_collection", response_model=MessageResponse)
async def create_collection(collection_name: str = Form(...)):
    """Create a new collection"""
    db_client = get_db_client()
    try:
        db_client.create_collection(name=collection_name)
        return {"message": f"Collection '{collection_name}' created successfully"}
    except ValueError:
        # Collection already exists
        return {"message": f"Collection '{collection_name}' already exists"}