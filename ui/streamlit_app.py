import streamlit as st
from ui.components import render_collection_sidebar, render_upload_section, render_search_section
from utils.config import get_config

def main():
    # Set page config
    st.set_page_config(page_title="Multimodal Search App", layout="wide")
    
    st.title("Multimodal Search App")
    
    # Get API URL from config
    config = get_config()
    api_url = config.get("API_URL", "http://localhost:8000")
    
    # Sidebar for collections and uploads
    with st.sidebar:
        collections = render_collection_sidebar(api_url)
        render_upload_section(api_url, collections)
    
    # Main area for search
    render_search_section(api_url, collections)

if __name__ == "__main__":
    main()