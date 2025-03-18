from flask import Flask, request, jsonify
import os
import shutil
from typing import List, Any
from multimodal_search.chroma_db import get_multi_vector_retriever
from multimodal_search.utils import save_images_from_results, print_retriever_contents
import io
from contextlib import redirect_stdout

app = Flask(__name__)

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

@app.route('/search', methods=['POST'])
def search():
    """
    Flask route for image-based multimodal search

    Request JSON format:
    {
        "query": "search query text",
        "gallery_path": "optional path to images (default: ./data/default_collection)",
        "collection_name": "optional collection name (default: default_collection)"
    }

    Returns:
        JSON response with search results and output messages
    """
    # Get JSON data from request
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Extract parameters from request
    query = data.get('query')
    gallery_path = data.get('gallery_path', './data/default_collection')
    collection_name = data.get('collection_name', 'default_collection')

    # Validate required parameters
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    # Capture stdout to include in response
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        try:
            # Get retriever_multi_vector_img
            retriever_multi_vector_img = get_multi_vector_retriever(
                gallery_path=gallery_path,
                collection_name=collection_name
            )

            # Show information
            print_retriever_contents(retriever_multi_vector_img)

            # Perform search
            results = retriever_multi_vector_img.get_relevant_documents(query)

            # Save results
            output_dir = os.path.join(os.getcwd(), "outputs")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)
            save_images_from_results(results)

            # Get list of saved image files
            image_files = []
            if os.path.exists(output_dir):
                image_files = [
                    f for f in os.listdir(output_dir)
                    if os.path.isfile(os.path.join(output_dir, f)) and
                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
                ]

            # Return success response
            return jsonify({
                "status": "success",
                "message": "Search completed successfully",
                "output": output_buffer.getvalue(),
                "result_count": len(results),
                "image_files": image_files
            })

        except Exception as e:
            # Return error response
            return jsonify({
                "status": "error",
                "message": f"Search failed: {str(e)}",
                "output": output_buffer.getvalue()
            }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
