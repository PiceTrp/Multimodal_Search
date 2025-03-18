import pickle
import os
import json
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


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

    # # Helper function to add documents to the vectorstore and docstore
    # def add_documents(retriever, doc_summaries, doc_contents):
    #     doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    #     summary_docs = [
    #         Document(page_content=s, metadata={id_key: doc_ids[i]})
    #         for i, s in enumerate(doc_summaries)
    #     ]
    #     retriever.vectorstore.add_documents(summary_docs)
    #     retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # # Add texts and images
    # # Check that image_summaries is not empty before adding
    # if image_summaries:
    #     add_documents(retriever, image_summaries, images)
    # # Check that image_texts is not empty before adding
    # if image_texts:
    #     add_documents(retriever, image_texts, images)

    return retriever


def save_multi_vector_retriever(retriever, base_path):
    """
    Save a MultiVectorRetriever to disk
    
    Args:
        retriever: The MultiVectorRetriever to save
        base_path: Directory to save the retriever components
    """
    # Create the directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Save Chroma collection name and settings
    chroma_settings = {
        "collection_name": retriever.vectorstore._collection.name,
        # Add other relevant settings if needed
    }
    with open(os.path.join(base_path, "chroma_settings.json"), "w") as f:
        json.dump(chroma_settings, f)
    
    # Save the document store using pickle
    with open(os.path.join(base_path, "docstore.pkl"), "wb") as f:
        pickle.dump(retriever.docstore, f)
    
    # Save any important parameters
    params = {
        "id_key": retriever.id_key,
        "search_kwargs": retriever.search_kwargs if hasattr(retriever, "search_kwargs") else {},
    }
    with open(os.path.join(base_path, "params.pkl"), "wb") as f:
        pickle.dump(params, f)


def load_multi_vector_retriever(base_path):
    """
    Load a MultiVectorRetriever from disk
    
    Args:
        base_path: Directory where the retriever components are saved
        
    Returns:
        A MultiVectorRetriever instance
    """
    # Load Chroma settings
    with open(os.path.join(base_path, "chroma_settings.json"), "r") as f:
        chroma_settings = json.load(f)
    
    # Recreate the vectorstore
    # Note: This assumes Chroma's data is stored in its default location
    vectorstore = Chroma(
        collection_name=chroma_settings["collection_name"],
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
    
    # Load the document store
    with open(os.path.join(base_path, "docstore.pkl"), "rb") as f:
        docstore = pickle.load(f)
    
    # Load parameters
    with open(os.path.join(base_path, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    
    # Recreate the retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=params["id_key"],
        search_kwargs=params.get("search_kwargs", {})
    )
    
    return retriever