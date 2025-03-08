import io
import pytesseract
from PIL import Image

def process_media_file(file_content, file_extension):
    """Process media file and extract text if applicable"""
    extracted_text = ""
    
    if file_extension in [".jpg", ".jpeg", ".png"]:
        # Process image
        image = Image.open(io.BytesIO(file_content))
        # Extract text from image using OCR
        extracted_text = pytesseract.image_to_string(image)
    elif file_extension in [".mp4", ".avi", ".mov"]:
        # For videos, we could extract frames and process them
        # For simplicity, we'll just return an empty string
        extracted_text = ""
    
    return extracted_text