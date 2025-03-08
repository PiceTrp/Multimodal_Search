from typing import List, Optional
from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str
    collection_name: str
    limit: int = 10

class MediaMetadata(BaseModel):
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    collection_name: str

class SearchResult(BaseModel):
    id: str
    metadata: dict
    similarity_score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

class CollectionResponse(BaseModel):
    collections: List[str]

class MessageResponse(BaseModel):
    message: str

class UploadResponse(BaseModel):
    message: str
    media_id: str
    file_path: str