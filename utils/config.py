import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_config():
    """Get configuration from environment variables"""
    return {
        # API keys
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        
        # Server settings
        "API_URL": os.getenv("API_URL", "http://localhost:8000"),
        "HOST": os.getenv("HOST", "0.0.0.0"),
        "PORT": int(os.getenv("PORT", "8000")),
        
        # Storage settings
        "CHROMA_DB_PATH": os.getenv("CHROMA_DB_PATH", "data/chroma_db"),
        "UPLOADS_PATH": os.getenv("UPLOADS_PATH", "data/uploads")
    }