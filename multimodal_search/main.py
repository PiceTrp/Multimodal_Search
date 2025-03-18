import os
import argparse

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from image_data_extractor import extract_image_data_for_retrieval
from chroma_db import get_multi_vector_retriever, create_multi_vector_retriever, save_multi_vector_retriever, load_multi_vector_retriever
from utils import save_images_from_results, display_multi_vector_retriever_df, print_retriever_contents


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
    print_retriever_contents(retriever_multi_vector_img)

    # performs search
    query = args.query
    results = retriever_multi_vector_img.get_relevant_documents(query)
    # save results image
    save_images_from_results(results)


if __name__ == '__main__':
    main()

# python multimodal_search/main.py \
#     --gallery_path "./data/collections_test" \
#     --collection_name "collections_test" \
#     --query "dogs"
