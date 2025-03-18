import streamlit as st
import requests
import os
from PIL import Image
import glob

# Set page title
st.title("Image Search App")

# Define collection name selection
collections = ["default_collection", "collections_test", "collections_100_pics"]  # Add your collections here
collection_name = st.selectbox("Select Collection", collections)

# Define search bar
query = st.text_input("Enter search query")

# Search button
if st.button("Search"):
    if query:
        # Make API request to backend
        response = requests.post(
            "http://localhost:5001/search",
            json={"query": query, "collection_name": collection_name}
        )

        if response.status_code == 200:
            st.success("Search completed successfully!")

            # Display results
            st.subheader("Search Results")

            # Get all images from the output directory
            output_path = "../outputs"
            image_files = glob.glob(os.path.join(output_path, "*.jpg")) + \
                         glob.glob(os.path.join(output_path, "*.jpeg")) + \
                         glob.glob(os.path.join(output_path, "*.png"))

            if image_files:
                # Create columns for displaying images
                cols = st.columns(3)  # Adjust number of columns as needed

                for i, image_file in enumerate(image_files):
                    with cols[i % 3]:
                        img = Image.open(image_file)
                        st.image(img, caption=os.path.basename(image_file), use_container_width=True)
            else:
                st.info("No images found in the output directory")
        else:
            st.error(f"Error Jaa: {response.text}")
    else:
        st.warning("Please enter a search query")
