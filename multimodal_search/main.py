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
    
    chroma_db_path = os.path.join(chroma_db_dirpath, collection_name)

    if os.path.exists(chroma_db_path):
        print(f"Chroma database '{collection_name}' found at: {chroma_db_path}")
        print("Loading...")
        retriever_multi_vector_img = load_multi_vector_retriever(chroma_db_path)
        print("Chroma database loaded.")
        return retriever_multi_vector_img

    else:
        print(f"Chroma database '{collection_name}' not found at: {chroma_db_path}.")
        print("Generating...")
        
        # Generate image summaries
        print("Start extracting information...")
        img_base64_list, image_summaries, image_texts = extract_image_data_for_retrieval(gallery_path)
        print("End extracting information...")

        # The vectorstore to use to index the summaries
        print("Start creating retriever_multi_vector...")
        vectorstore = Chroma(
            collection_name=collection_name, 
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        )

        # Create retriever
        retriever_multi_vector_img = create_multi_vector_retriever(
            vectorstore,
            img_base64_list,
            image_summaries,
            image_texts,
        )
        print("End creating retriever_multi_vector...")

        # Save the retriever - Default to chroma_db/...
        save_multi_vector_retriever(retriever_multi_vector_img,  chroma_db_path)
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