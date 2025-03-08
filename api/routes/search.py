from fastapi import APIRouter, HTTPException
from api.schemas import SearchQuery, SearchResponse
from services.db_service import get_collection
from models.model_adapter import ModelAdapter

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """Search for media based on text query"""
    try:
        # Get collection
        collection = get_collection(query.collection_name)
        
        # Generate embedding for query
        model_adapter = ModelAdapter(model_type="gemini")
        query_embedding = model_adapter.get_embedding(query.query)
        
        # Search collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.limit
        )
        
        # Format and return results
        formatted_results = []
        for i, (id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
            formatted_results.append({
                "id": id,
                "metadata": results["metadatas"][0][i],
                "similarity_score": distance
            })
        
        return {"results": formatted_results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")