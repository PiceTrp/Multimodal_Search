import streamlit as st
import requests
import json
import os
from PIL import Image


def render_collection_sidebar(api_url):
    """Render the collection management sidebar"""
    st.header("Collections")
    collections = get_collections(api_url)
    
    # Create new collection
    with st.expander("Create New Collection"):
        new_collection = st.text_input("Collection Name")
        if st.button("Create Collection"):
            if new_collection:
                result = create_collection(api_url, new_collection)
                st.success(result["message"])
                # Refresh collections
                st.experimental_rerun()
            else:
                st.warning("Please enter a collection name")
    
    return collections

def render_upload_section(api_url, collections):
    """Render the media upload section"""
    st.header("Upload Media")
    if collections:
        upload_collection = st.selectbox("Select Collection for Upload", collections)
        uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
        
        if uploaded_file:
            # Display preview
            if uploaded_file.type.startswith("image"):
                st.image(uploaded_file, caption="Preview", width=200)
            elif uploaded_file.type.startswith("video"):
                st.video(uploaded_file)
                # Metadata form
                description = st.text_area("Description")
                tags = st.text_input("Tags (comma-separated)")
                
                if st.button("Upload"):
                    metadata = {
                        "description": description,
                        "tags": [tag.strip() for tag in tags.split(",") if tag.strip()]
                    }
                    
                    result = upload_media(api_url, uploaded_file, metadata, upload_collection)
                    st.success(f"File uploaded successfully! ID: {result.get('media_id', 'Unknown')}")
    else:
        st.warning("Please create a collection first")

def render_search_section(api_url, collections):
    """Render the search section"""
    st.header("Search Media")
    
    # Collection selector for search
    if collections:
        search_collection = st.selectbox("Select Collection to Search", collections)
        query = st.text_input("Enter your search query")
        
        if st.button("Search"):
            if query:
                with st.spinner("Searching..."):
                    results = search_media(api_url, query, search_collection)
                
                # Display results
                if results.get("results"):
                    st.subheader(f"Found {len(results['results'])} results")
                    
                    # Display results in a grid
                    cols = st.columns(3)
                    for i, result in enumerate(results["results"]):
                        col_idx = i % 3
                        with cols[col_idx]:
                            metadata = result["metadata"]
                            file_path = metadata.get("path")
                            
                            # Display image or video
                            if file_path and os.path.exists(file_path):
                                if metadata.get("file_type", "").startswith("image"):
                                    st.image(file_path, caption=metadata.get("filename"), use_column_width=True)
                                elif metadata.get("file_type", "").startswith("video"):
                                    st.video(file_path)
                                
                                # Display metadata
                                st.write(f"**Description:** {metadata.get('description', 'N/A')}")
                                st.write(f"**Tags:** {', '.join(metadata.get('tags', ['N/A']))}")
                                
                                # Display extracted text if available
                                extracted_text = metadata.get("extracted_text", "")
                                if extracted_text:
                                    with st.expander("Extracted Text"):
                                        st.write(extracted_text)
                                
                                st.write(f"**Similarity Score:** {result.get('similarity_score', 'N/A'):.4f}")
                                st.divider()
                else:
                    st.info("No results found. Try a different query.")
            else:
                st.warning("Please enter a search query")
    else:
        st.warning("No collections available. Please create a collection first.")

# API Helper Functions
def get_collections(api_url):
    """Fetch available collections from API"""
    try:
        response = requests.get(f"{api_url}/collections")
        if response.status_code == 200:
            return response.json()["collections"]
        else:
            st.error(f"Error fetching collections: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

def create_collection(api_url, name):
    """Create a new collection"""
    try:
        response = requests.post(
            f"{api_url}/create_collection",
            data={"collection_name": name}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error creating collection: {str(e)}")
        return {"message": f"Error: {str(e)}"}

def upload_media(api_url, file, metadata, collection):
    """Upload media file to API"""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {
            "metadata": json.dumps(metadata),
            "collection_name": collection
        }
        
        response = requests.post(f"{api_url}/upload", files=files, data=data)
        return response.json()
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return {"message": f"Error: {str(e)}"}

def search_media(api_url, query, collection, limit=10):
    """Search for media"""
    try:
        data = {
            "query": query,
            "collection_name": collection,
            "limit": limit
        }
        response = requests.post(f"{api_url}/search", json=data)
        return response.json()
    except Exception as e:
        st.error(f"Error searching: {str(e)}")
        return {"results": []}