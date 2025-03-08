# Multimodal Search App

A web application that allows users to search images and videos using text queries. The app extracts and indexes text content from images and metadata to provide comprehensive search capabilities.

## Features

- Search images and videos by text query
- Extract and index text content inside images using OCR
- Index media metadata (descriptions, tags)
- Collection-based organization
- Swappable embedding models

## Tech Stack

- **Frontend**: Streamlit
- **Backend API**: FastAPI
- **Vector Database**: ChromaDB
- **ML Model**: Google Gemini (with adapter architecture for swapping models)
- **OCR**: Tesseract

## Project Structure

```
multimodal-search-app/
│
├── api/               # FastAPI backend
│   ├── app.py         # Main FastAPI application
│   ├── routes/        # API routes
│   └── schemas.py     # API data models
│
├── models/            # ML model adapters
│   └── model_adapter.py # Swappable model interface
│
├── services/          # Business logic services
│   ├── db_service.py  # ChromaDB interaction
│   └── media_service.py # Media processing
│
├── ui/                # Streamlit frontend
│   ├── streamlit_app.py # Main Streamlit application
│   └── components.py  # UI components
│
├── utils/             # Utility functions
│   └── config.py      # Configuration management
│
├── data/              # Data storage
│   ├── uploads/       # Uploaded media files
│   └── chroma_db/     # ChromaDB persistence
│
├── requirements.txt   # Python dependencies
├── .env               # Environment variables
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
└── entrypoint.sh      # Startup script
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Docker (optional)
- Google API key for Gemini model

### Option 1: Local Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR:
   - On Ubuntu: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
4. Create a `.env` file with your API keys (see `.env` example)
5. Start the backend:
   ```
   python -m api.app
   ```
6. Start the frontend (in a separate terminal):
   ```
   streamlit run ui/streamlit_app.py
   ```

### Option 2: Docker Setup

1. Clone the repository
2. Create a `.env` file with your API keys
3. Build and start with Docker Compose:
   ```
   docker-compose up --build
   ```

## Usage

1. Access the application at `http://localhost:8501`
2. Create a collection in the sidebar
3. Upload images or videos with metadata
4. Use the search bar to find media based on text queries

## Extending with New Models

To add a new embedding model, update the `model_adapter.py` file:

1. Add your model in the `__init__` method
2. Implement the embedding extraction in the `get_embedding` method
3. For multimodal models, implement the `get_multimodal_embedding` method
