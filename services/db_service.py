import chromadb
from utils.config import get_config

# Initialize ChromaDB client
_db_client = None

def get_db_client():
    """Get or initialize the ChromaDB client"""
    global _db_client
    if _db_client is None:
        config = get_config()
        db_path = config.get("CHROMA_DB_PATH", "data/chroma_db")
        _db_client = chromadb.PersistentClient(path=db_path)
    return _db_client

def get_collection(collection_name):
    """Get a collection or create it if it doesn't exist"""
    db_client = get_db_client()
    try:
        collection = db_client.get_collection(name=collection_name)
    except ValueError:
        collection = db_client.create_collection(name=collection_name)
    return collection