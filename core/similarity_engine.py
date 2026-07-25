"""
Similarity Engine Module
Calculate cosine similarity and other distance metrics between embeddings
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class SimilarityEngine:
    """
    Calculate similarity scores between embeddings
    """
    
    @staticmethod
    def cosine_similarity_score(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0-100
        """
        # Handle zero vectors
        if np.allclose(embedding1, 0) or np.allclose(embedding2, 0):
            return 50.0  # Neutral score for zero vectors
        
        # Reshape for sklearn
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        
        # Calculate cosine similarity (returns value in [-1, 1])
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        # Convert from [-1, 1] to [0, 100]
        # -1 -> 0, 0 -> 50, 1 -> 100
        score = (similarity + 1) * 50
        
        return float(score)
    
    @staticmethod
    def batch_similarity(resume_embeddings: np.ndarray, 
                        jd_embedding: np.ndarray) -> List[float]:
        """
        Calculate similarity for multiple resumes against one JD
        Efficient batch processing
        
        Args:
            resume_embeddings: Array of shape (num_resumes, embedding_dim)
            jd_embedding: Single embedding vector
            
        Returns:
            List of similarity scores (0-100)
        """
        if len(resume_embeddings) == 0:
            return []
        
        # Reshape JD embedding for comparison
        jd_emb = jd_embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(resume_embeddings, jd_emb)
        
        # Convert to 0-100 scale
        scores = [(sim[0] + 1) * 50 for sim in similarities]
        
        return scores
    
    @staticmethod
    def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(embedding1 - embedding2))
    
    @staticmethod
    def dot_product_score(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate dot product similarity
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Dot product score
        """
        return float(np.dot(embedding1, embedding2))


if __name__ == "__main__":
    # Test the similarity engine
    engine = SimilarityEngine()
    
    # Test with identical vectors
    vec1 = np.random.rand(384)
    score_identical = engine.cosine_similarity_score(vec1, vec1)
    print(f"Identical vectors score: {score_identical:.2f} (should be ~100)")
    
    # Test with orthogonal vectors
    vec2 = np.array([1.0] + [0.0] * 383)
    vec3 = np.array([0.0, 1.0] + [0.0] * 382)
    score_orthogonal = engine.cosine_similarity_score(vec2, vec3)
    print(f"Orthogonal vectors score: {score_orthogonal:.2f} (should be ~50)")
    
    # Test batch similarity
    resume_vecs = np.random.rand(5, 384)
    jd_vec = np.random.rand(384)
    batch_scores = engine.batch_similarity(resume_vecs, jd_vec)
    print(f"Batch scores: {[f'{s:.2f}' for s in batch_scores]}")
