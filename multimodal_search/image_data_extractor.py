import base64
import os
from typing import List, Tuple
from tqdm import tqdm

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def encode_image(image_path: str) -> str:
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def prompt_query_with_image(
    img_base64: str, 
    prompt: str, 
    model_name: str = "gemini-2.0-flash", 
    max_tokens: int = 1024
) -> str:
    """Queries Google Gemini API with an image and a text prompt."""
    chat = ChatGoogleGenerativeAI(model=model_name, max_output_tokens=max_tokens)
    # Create the message using HumanMessage with text and image
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def extract_image_data_for_retrieval(image_directory: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Generate base64 encoded strings, information summaries, and extracted texts for images in a directory (gallery).
    
    Args:
        image_directory: Path to the directory containing .jpg files.

    Returns:
        Tuple containing:
            - List of base64 encoded image strings.
            - List of image summaries.
            - List of extracted image texts.
    """

    # Store base64 encoded images
    img_base64_list = []
    # Store image summaries
    image_summaries = []
    # Store text extraction information
    image_texts = []

    # Prompt
    summary_prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    text_extraction_prompt = """You are an assistant tasked with extracting all the texts yoe see in an image for retrieval. \
    These texts will be embedded and used to retrieve the raw image. \
    Give all texts that you see in the image that is well optimized for retrieval."""

    # Apply to images
    for img_file in tqdm(sorted(os.listdir(image_directory))):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(image_directory, img_file)
            base64_image = encode_image(img_path)
            # img_base64 list append
            img_base64_list.append(base64_image)
            # image_summaries list append
            image_summaries.append(prompt_query_with_image(base64_image, summary_prompt))
            # image_texts list append
            image_texts.append(prompt_query_with_image(base64_image, text_extraction_prompt))

    return img_base64_list, image_summaries, image_texts