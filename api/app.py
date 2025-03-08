from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from api.routes import collections, media, search

# Initialize FastAPI app
app = FastAPI(title="Multimodal Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    print("Starting up the API server...")
    # Ensure the uploads directory exists
    os.makedirs("data/uploads", exist_ok=True)

# Include routers
app.include_router(collections.router, tags=["collections"])
app.include_router(media.router, tags=["media"])
app.include_router(search.router, tags=["search"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)