import os
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from tabulate import tabulate


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


def print_retriever_contents(retriever):
    """
    Prints the image summaries and text documents stored in the retriever's vectorstore
    in a DataFrame format.

    Args:
        retriever (MultiVectorRetriever): The multi-vector retriever.
    """
    # Access vectorstore
    vectorstore = retriever.vectorstore

    # Retrieve all documents (metadata and embeddings are usually stored as documents)
    documents = vectorstore.get(include=["documents", "metadatas"])

    # Extract information
    image_summaries = [doc for doc in documents["documents"]]  # Text embeddings or image summaries
    text_docs = [meta.get("text", "N/A") for meta in documents["metadatas"]]  # Metadata text

    # Create DataFrame
    df = pd.DataFrame({"Image Summaries": image_summaries, "Text Documents": text_docs})

    # Print DataFrame
    print(df)


def display_multi_vector_retriever_df(retriever):
    """
    Prints the image summaries and text documents stored in the retriever's vectorstore
    in a well-formatted table.
    """
    # Access vectorstore
    vectorstore = retriever.vectorstore

    # Retrieve all documents (metadata and embeddings are usually stored as documents)
    documents = vectorstore.get(include=["documents", "metadatas"])

    # Extract information
    image_summaries = [doc for doc in documents["documents"]]
    text_docs = [meta.get("text", "N/A") for meta in documents["metadatas"]]

    # Create DataFrame
    df = pd.DataFrame({"Image Summaries": image_summaries, "Text Documents": text_docs})

    # Print nicely formatted table
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=True))
