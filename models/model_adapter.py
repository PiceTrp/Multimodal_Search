import os
import google.generativeai as genai
import numpy as np
from utils.config import get_config

class ModelAdapter:
    """Adapter for different embedding models"""
    
    def __init__(self, model_type="gemini", api_key=None):
        self.model_type = model_type.lower()
        config = get_config()
        
        if self.model_type == "gemini":
            # Use the API key from environment or provided
            api_key = api_key or config.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key is required for Gemini model")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.embedding_model = genai.GenerativeModel('embedding-001')
        
        elif self.model_type == "ollama":
            # Example setup for Ollama API
            from ollama import Client
            self.client = Client(host=config.get("OLLAMA_HOST", "http://localhost:11434"))
            self.model_name = "llama3"  # or any other model available in Ollama
        
        elif self.model_type == "openai":
            # Example setup for OpenAI API
            import openai
            openai.api_key = api_key or config.get("OPENAI_API_KEY")
            self.client = openai.Client()
        
        elif self.model_type == "huggingface":
            # Example setup for Hugging Face model
            from transformers import AutoTokenizer, AutoModel
            model_name = "Qwen/Qwen-VL-Chat"  # Example model name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_embedding(self, text):
        """Get embedding for text based on the selected model"""
        if self.model_type == "gemini":
            result = self.embedding_model.embed_content(text)
            return result.embedding
        
        elif self.model_type == "ollama":
            response = self.client.embeddings(model=self.model_name, prompt=text)
            return response["embedding"]
        
        elif self.model_type == "openai":
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        
        elif self.model_type == "huggingface":
            # Process with HuggingFace model
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            # Use the [CLS] token embedding as the text embedding
            embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            return embedding.flatten().tolist()
        
        else:
            raise ValueError(f"Embedding not implemented for model type: {self.model_type}")

    def get_multimodal_embedding(self, image_path, text=None):
        """Get embedding for an image, optionally combined with text"""
        if self.model_type == "gemini":
            # For Gemini, we'd need to use the multimodal capabilities
            # This is a simplified example
            import PIL.Image
            image = PIL.Image.open(image_path)
            
            if text:
                response = self.model.generate_content([text, image])
            else:
                response = self.model.generate_content(image)
                
            # For actual embeddings we'd need to use the embedding model
            # This is just a placeholder - in reality we'd need to use
            # the proper embedding extraction method
            text_representation = response.text
            result = self.embedding_model.embed_content(text_representation)
            return result.embedding
        
        # Implement other model types as needed
        else:
            raise NotImplementedError(f"Multimodal embedding not implemented for {self.model_type}")