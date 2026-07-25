"""
BERT Embedder Module
Generates semantic embeddings using Sentence-BERT for resumes and job descriptions
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class BERTEmbedder:
    """
    Generate semantic embeddings using Sentence-BERT
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize BERT embedder with specified model
        
        Args:
            model_name: HuggingFace model name (default: all-mpnet-base-v2)
        """
        logging.info(f"Loading BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logging.info(f"BERT model loaded successfully (dim={self.embedding_dim})")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim)
        
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts (efficient batch processing)
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
    
    def embed_resume(self, resume_text: str) -> np.ndarray:
        """
        Generate embedding for resume
        
        Args:
            resume_text: Full resume text
            
        Returns:
            numpy array embedding
        """
        return self.embed_text(resume_text)
    
    def embed_jd(self, jd_text: str) -> np.ndarray:
        """
        Generate embedding for job description
        
        Args:
            jd_text: Job description text
            
        Returns:
            numpy array embedding
        """
        return self.embed_text(jd_text)


if __name__ == "__main__":
    # Test the embedder
    embedder = BERTEmbedder(model_name="all-MiniLM-L6-v2")
    
    test_text = "Senior Python Developer with 5 years experience in Flask and AWS"
    embedding = embedder.embed_text(test_text)
    
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Embedding sample: {embedding[:5]}")
