import os
import argparse

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from image_data_extractor import extract_image_data_for_retrieval
from chroma_db import create_multi_vector_retriever, save_multi_vector_retriever, load_multi_vector_retriever
from utils import save_images_from_results, inspect_multi_vector_retriever


def get_multi_vector_retriever(gallery_path, collection_name):
    """
    Retrieves a multi-vector retriever, either by loading an existing Chroma database
    or by creating a new one.

    Args:
        gallery_path (str): Path to the image gallery.
        collection_name (str): Name of the Chroma collection.

    Returns:
        langchain.retrievers.MultiVectorRetriever: The multi-vector retriever.
    """
    chroma_db_dirpath = os.path.join(os.getcwd(), "chroma_db")
    if not os.path.exists(chroma_db_dirpath):
        os.makedirs(chroma_db_dirpath, exist_ok=True)
    
    # Path to save/load the retriever
    retriever_save_path = os.path.join(chroma_db_dirpath, collection_name)

    # Create embedding function
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Check if a saved retriever exists
    if os.path.exists(retriever_save_path) and os.path.exists(os.path.join(retriever_save_path, "config.json")):
        print(f"MultiVectorRetriever '{collection_name}' found at: {retriever_save_path}")
        print("Loading...")
        
        # Define a function to load Chroma
        def load_chroma(collection_name, persist_directory, embedding_function):
            return Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=persist_directory
            )
        
        # Load the retriever
        retriever_multi_vector_img = load_multi_vector_retriever(
            retriever_save_path,
            vectorstore_load_func=load_chroma,
            vectorstore_load_kwargs={
                "collection_name": collection_name,
                "persist_directory": retriever_save_path,
                "embedding_function": embeddings,
            }
        )
        
        print("MultiVectorRetriever loaded successfully.")
        return retriever_multi_vector_img

    else:
        print(f"MultiVectorRetriever '{collection_name}' not found at: {retriever_save_path}.")
        print("Generating new retriever...")
        
        # Generate image summaries
        print("Start extracting information from images...")
        img_base64_list, image_summaries, image_texts = extract_image_data_for_retrieval(gallery_path)
        print("Finished extracting information.")

        # Create the vectorstore to use for indexing
        print("Creating vectorstore...")
        persist_directory = os.path.join(chroma_db_dirpath, collection_name)
        vectorstore = Chroma(
            collection_name=collection_name, 
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

        # Create retriever
        print("Creating multi-vector retriever...")
        retriever_multi_vector_img = create_multi_vector_retriever(
            vectorstore,
            img_base64_list,
            image_summaries,
            image_texts,
        )
        print("Multi-vector retriever created successfully.")

        # Save the retriever
        print(f"Saving retriever to {retriever_save_path}...")
        save_multi_vector_retriever(
            retriever_multi_vector_img,  
            retriever_save_path,
            vectorstore_save_method="persist"  # For Chroma, use "persist"
        )
        print("Retriever saved successfully.")
        
        return retriever_multi_vector_img


def main():
    parser = argparse.ArgumentParser(description="Image-based multimodal search system")
    parser.add_argument("--gallery_path", type=str,  default="images", help="Path to the image or directory of images")
    parser.add_argument("--collection_name", type=str,  default="default_collection", help="Chroma collection name for indexing")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    args = parser.parse_args()

    # Get retriever_multi_vector_img
    retriever_multi_vector_img = get_multi_vector_retriever(gallery_path=args.gallery_path,
                                                            collection_name=args.collection_name)
    
    # show information
    inspect_multi_vector_retriever(retriever_multi_vector_img, max_display_items=5)
    
    # performs search
    query = args.query
    results = retriever_multi_vector_img.get_relevant_documents(query)
    # save results image
    save_images_from_results(results)
    

if __name__ == '__main__':
    main()

# python -m multimodal_search/main.py \
    # --gallery_path "./data/collections_test" \
    # --collection_name "collections_test" \
    # --saved_retriever_path "./chroma_db/collections_test" \
    # --query "dogs"