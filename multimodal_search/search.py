import os
import shutil
from typing import List, Any
from multimodal_search.chroma_db import get_multi_vector_retriever
from multimodal_search.utils import save_images_from_results, print_retriever_contents


def search(
    query: str,
    gallery_path: str = "./data/default_collection",
    collection_name: str = "default_collection"
) -> None:
    """
    Image-based multimodal search system

    Args:
        query: Search query
        gallery_path: Path to the image or directory of images
        collection_name: Chroma collection name for indexing
    """
    # Get retriever_multi_vector_img
    retriever_multi_vector_img = get_multi_vector_retriever(
        gallery_path=gallery_path,
        collection_name=collection_name
    )

    # show information
    print_retriever_contents(retriever_multi_vector_img)

    # performs search
    results = retriever_multi_vector_img.get_relevant_documents(query)

    # save results
    output_dir = os.path.join(os.getcwd(), "outputs")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    save_images_from_results(results)


# if __name__ == '__main__':
#     search()

# python multimodal_search/search.py \
#     --gallery_path "./data/collections_test" \
#     --collection_name "collections_test" \
#     --query "dogs"
