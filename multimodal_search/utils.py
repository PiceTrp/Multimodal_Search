import os
import base64
from io import BytesIO
from PIL import Image

def save_base64_image(base64_string, output_path):
    """Decodes a base64 string and saves it as an image file."""
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)
    
    # Open the image from the decoded bytes
    image = Image.open(BytesIO(image_data))
    
    # Save the image to the specified output path
    image.save(output_path, 'JPEG')

def save_images_from_results(results):
    """Save the base64 images from the results list to the outputs folder."""
    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)

    for index, base64_string in enumerate(results):
        # Create the output file path with the correct name
        output_path = os.path.join('outputs', f'result_image_{index}.jpg')
        
        # Save the image
        save_base64_image(base64_string, output_path)
        print(f"Saved image {output_path}")

def inspect_multi_vector_retriever(retriever, max_display_items=5):
    """
    Display detailed information about a MultiVectorRetriever to understand its structure.
    
    Parameters:
    - retriever: The MultiVectorRetriever instance to inspect
    - max_display_items: Maximum number of items to display in each section
    
    Returns:
    - A dictionary containing the inspection results
    """
    import pprint
    from typing import Dict, List, Any
    
    results = {}
    
    # Inspect docstore
    print("=" * 50)
    print("DOCSTORE INSPECTION")
    print("=" * 50)
    
    # Get all keys from the docstore
    all_keys = list(retriever.docstore.yield_keys())
    doc_count = len(all_keys)
    print(f"Total documents in docstore: {doc_count}")
    
    # Display sample keys
    print("\nSample document IDs:")
    sample_keys = all_keys[:min(max_display_items, len(all_keys))]
    for i, key in enumerate(sample_keys):
        print(f"  {i+1}. {key}")
    
    # Sample document contents
    print("\nSample document contents:")
    sample_docs = []
    for key in sample_keys:
        doc = retriever.docstore.mget([key])[0]
        sample_docs.append(doc)
        doc_summary = str(doc)
        if len(doc_summary) > 100:
            doc_summary = doc_summary[:100] + "..."
        print(f"  {key}: {doc_summary} (Type: {type(doc).__name__})")
    
    results["docstore"] = {
        "total_docs": doc_count,
        "sample_keys": sample_keys,
        "sample_docs": sample_docs
    }
    
    # Inspect vectorstore
    print("\n" + "=" * 50)
    print("VECTORSTORE INSPECTION")
    print("=" * 50)
    
    # Get vectorstore type
    vectorstore_type = type(retriever.vectorstore).__name__
    print(f"Vectorstore type: {vectorstore_type}")
    
    # Attempt to get document count from vectorstore (implementation may vary)
    try:
        if hasattr(retriever.vectorstore, "index"):
            vector_count = len(retriever.vectorstore.index)
            print(f"Total vectors: {vector_count}")
        elif hasattr(retriever.vectorstore, "docstore"):
            vector_count = len(retriever.vectorstore.docstore.docs)
            print(f"Total vectors: {vector_count}")
        else:
            print("Vector count: Unknown (couldn't determine from vectorstore)")
            vector_count = "Unknown"
    except Exception as e:
        print(f"Error getting vector count: {str(e)}")
        vector_count = "Error"
    
    # Try to get a sample of vectors
    print("\nSample vectors:")
    try:
        # Most common vectorstore implementations
        if hasattr(retriever.vectorstore, "docstore") and hasattr(retriever.vectorstore.docstore, "docs"):
            sample_vector_docs = list(retriever.vectorstore.docstore.docs.values())[:max_display_items]
            for i, doc in enumerate(sample_vector_docs):
                doc_id = doc.metadata.get(retriever.id_key, "Unknown")
                print(f"  {i+1}. Document ID: {doc_id}")
                print(f"     Content: {doc.page_content[:50]}..." if len(doc.page_content) > 50 else f"     Content: {doc.page_content}")
                print(f"     Metadata: {pprint.pformat(doc.metadata, indent=10)}")
        else:
            print("  (Cannot retrieve sample vectors from this vectorstore implementation)")
            sample_vector_docs = []
    except Exception as e:
        print(f"  Error retrieving sample vectors: {str(e)}")
        sample_vector_docs = []
    
    results["vectorstore"] = {
        "type": vectorstore_type,
        "vector_count": vector_count,
        "sample_vectors": sample_vector_docs
    }
    
    # Perform a test retrieval
    print("\n" + "=" * 50)
    print("TEST RETRIEVAL")
    print("=" * 50)
    
    try:
        if hasattr(retriever.vectorstore, "docstore") and hasattr(retriever.vectorstore.docstore, "docs"):
            # Get a sample query from one of the vectors
            if sample_vector_docs:
                test_query = sample_vector_docs[0].page_content
                shortened_query = test_query[:50] + "..." if len(test_query) > 50 else test_query
                print(f"Test query: \"{shortened_query}\"")
                
                retrieved_docs = retriever.get_relevant_documents(test_query, k=2)
                print(f"Retrieved {len(retrieved_docs)} documents")
                
                for i, doc in enumerate(retrieved_docs):
                    print(f"\nRetrieved document {i+1}:")
                    doc_summary = str(doc)
                    if len(doc_summary) > 100:
                        doc_summary = doc_summary[:100] + "..."
                    print(f"  Content: {doc_summary}")
                    print(f"  Type: {type(doc).__name__}")
            else:
                print("No sample vectors available for test retrieval")
                retrieved_docs = []
        else:
            print("Cannot perform test retrieval with this vectorstore implementation")
            retrieved_docs = []
    except Exception as e:
        print(f"Error during test retrieval: {str(e)}")
        retrieved_docs = []
    
    results["test_retrieval"] = {
        "retrieved_docs": retrieved_docs
    }
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Documents in docstore: {doc_count}")
    print(f"Vectors in vectorstore: {vector_count}")
    print(f"ID key used: {retriever.id_key}")
    
    if isinstance(vector_count, int) and isinstance(doc_count, int):
        vectors_per_doc = vector_count / doc_count if doc_count > 0 else 0
        print(f"Average vectors per document: {vectors_per_doc:.2f}")
    
    return results