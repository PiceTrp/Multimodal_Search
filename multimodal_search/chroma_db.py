import pickle
import os
import json
import uuid
from typing import Dict, List, Any, Optional, Union, Type
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# project
from image_data_extractor import extract_image_data_for_retrieval


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


def create_multi_vector_retriever(
    vectorstore, images, image_summaries, image_texts,
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Make sure we have the same number of images, summaries, and texts
    assert len(images) == len(image_summaries) == len(image_texts), "Number of images, summaries, and texts must match"

    # Generate unique IDs for each image
    doc_ids = [str(uuid.uuid4()) for _ in images]
    # Add all documents to the docstore first
    retriever.docstore.mset(list(zip(doc_ids, images)))

    # Create and add documents for summaries
    summary_docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)

    # Create and add documents for texts
    text_docs = [
        Document(page_content=text, metadata={id_key: doc_ids[i]})
        for i, text in enumerate(image_texts)
    ]
    retriever.vectorstore.add_documents(text_docs)

    return retriever


def save_multi_vector_retriever(
    retriever: MultiVectorRetriever,
    save_dir: str,
    vectorstore_save_method: Optional[str] = None,
) -> Dict[str, str]:
    """
    Save a MultiVectorRetriever to disk.

    Parameters:
    - retriever: The MultiVectorRetriever instance to save
    - save_dir: Directory where to save the retriever components
    - vectorstore_save_method: Optional method name to call on vectorstore for saving
                               (e.g., "save_local" for FAISS, "persist" for Chroma)

    Returns:
    - A dictionary with paths to the saved components
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    saved_paths = {}

    # 1. Save docstore contents
    docstore_path = os.path.join(save_dir, "docstore.pkl")

    # Get all keys and documents from docstore
    all_keys = list(retriever.docstore.yield_keys())
    all_docs = retriever.docstore.mget(all_keys)
    docstore_data = dict(zip(all_keys, all_docs))

    with open(docstore_path, 'wb') as f:
        pickle.dump(docstore_data, f)

    saved_paths['docstore'] = docstore_path

    # 2. Save vectorstore if a save method is provided
    if vectorstore_save_method is not None and hasattr(retriever.vectorstore, vectorstore_save_method):
        vectorstore_path = os.path.join(save_dir, "vectorstore")
        os.makedirs(vectorstore_path, exist_ok=True)

        save_method = getattr(retriever.vectorstore, vectorstore_save_method)

        # Handle different vectorstore saving methods
        if vectorstore_save_method == "persist":
            # For Chroma, persist method typically saves to the current collection path
            # We need to save the collection_name and persist_directory
            if hasattr(retriever.vectorstore, "_collection"):
                collection_info = {
                    "collection_name": retriever.vectorstore._collection.name,
                    "persist_directory": retriever.vectorstore._persist_directory
                }
                collection_info_path = os.path.join(save_dir, "vectorstore_info.json")
                with open(collection_info_path, 'w') as f:
                    json.dump(collection_info, f)
                saved_paths['vectorstore_info'] = collection_info_path

            # Call persist method
            save_method()
            saved_paths['vectorstore'] = retriever.vectorstore._persist_directory
        else:
            # For other vectorstores, call the save method with the path
            save_method(vectorstore_path)
            saved_paths['vectorstore'] = vectorstore_path

    # 3. Save configuration
    config = {
        'id_key': retriever.id_key,
        'search_kwargs': retriever.search_kwargs or {},
        'vectorstore_type': retriever.vectorstore.__class__.__module__ + "." + retriever.vectorstore.__class__.__name__,
        'docstore_type': retriever.docstore.__class__.__module__ + "." + retriever.docstore.__class__.__name__,
    }

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)

    saved_paths['config'] = config_path

    print(f"MultiVectorRetriever saved to {save_dir}")
    return saved_paths


def load_multi_vector_retriever(
    save_dir: str,
    vectorstore_load_func: Optional[callable] = None,
    vectorstore_load_kwargs: Optional[Dict[str, Any]] = None,
) -> MultiVectorRetriever:
    """
    Load a MultiVectorRetriever from disk.

    Parameters:
    - save_dir: Directory where the retriever components are saved
    - vectorstore_load_func: A function that loads the vectorstore (e.g., Chroma.from_existing_collection)
    - vectorstore_load_kwargs: Additional kwargs for vectorstore loading

    Returns:
    - A reconstructed MultiVectorRetriever instance
    """
    # 1. Load configuration
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. Load docstore contents
    docstore_path = os.path.join(save_dir, "docstore.pkl")
    with open(docstore_path, 'rb') as f:
        docstore_data = pickle.load(f)

    # Create new docstore and populate it
    docstore = InMemoryStore()
    for key, doc in docstore_data.items():
        docstore.mset([(key, doc)])

    # 3. Load vectorstore
    if vectorstore_load_func is None:
        raise ValueError("vectorstore_load_func must be provided to load the vectorstore")

    vectorstore_load_kwargs = vectorstore_load_kwargs or {}

    # Check if there's Chroma-specific information
    vectorstore_info_path = os.path.join(save_dir, "vectorstore_info.json")
    if os.path.exists(vectorstore_info_path):
        with open(vectorstore_info_path, 'r') as f:
            vectorstore_info = json.load(f)

        # Update kwargs with Chroma-specific info
        vectorstore_load_kwargs.update(vectorstore_info)

    # Load the vectorstore
    vectorstore = vectorstore_load_func(**vectorstore_load_kwargs)

    # 4. Reconstruct the MultiVectorRetriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=config['id_key'],
        search_kwargs=config.get('search_kwargs', {})
    )

    print(f"MultiVectorRetriever loaded from {save_dir}")
    return retriever
