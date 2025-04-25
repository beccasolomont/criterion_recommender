from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os
import json
import gc
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model at module level
logger.info("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L3-v2', device='cpu')  # Even smaller model
model.max_seq_length = 64  # Further reduce sequence length
logger.info("Model loaded successfully")

# Global variables for caching
_movies = None
_embeddings = None

def load_movies() -> List[Dict]:
    """Load movies from the database."""
    try:
        with open('data/criterion_movies.json', 'r') as f:
            movies = json.load(f)
        logger.info(f"Loaded {len(movies)} movies from database")
        return movies
    except Exception as e:
        logger.error(f"Error loading movies: {str(e)}")
        raise

def load_or_create_embeddings():
    """Load cached embeddings or create new ones if not available."""
    global _movies, _embeddings
    
    try:
        # Try to load cached embeddings
        if os.path.exists('movie_embeddings.npy'):
            logger.info("Loading cached embeddings...")
            _movies = load_movies()
            _embeddings = np.load('movie_embeddings.npy')
            logger.info(f"Loaded {len(_movies)} movies and embeddings from cache")
            return _movies, _embeddings
        
        # If no cache, create new embeddings
        logger.info("Creating new embeddings...")
        _movies = load_movies()
        if not _movies:
            raise ValueError("No movies found in the database")
        
        # Generate embeddings one at a time to minimize memory usage
        embeddings = []
        for movie in _movies:
            # Clear memory before each encoding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Generate embedding with minimal memory usage
            embedding = model.encode(
                [movie['description']],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=1,
                device='cpu'
            )[0]
            embeddings.append(embedding)
            
            # Force garbage collection after each embedding
            del embedding
            gc.collect()
        
        _embeddings = np.array(embeddings, dtype=np.float32)  # Use float32 to save memory
        logger.info("Embeddings generated successfully")
        
        # Save embeddings in numpy format for faster loading
        np.save('movie_embeddings.npy', _embeddings)
        logger.info("Cached embeddings saved")
        
        return _movies, _embeddings
        
    except Exception as e:
        logger.error(f"Error in load_or_create_embeddings: {str(e)}")
        raise

def get_recommendations(preferences: str, num_recommendations: int = 5) -> List[Dict]:
    """
    Get movie recommendations based on user preferences.
    
    Args:
        preferences (str): User's movie preferences/description
        num_recommendations (int): Number of recommendations to return
        
    Returns:
        List[Dict]: List of recommended movies with their details
    """
    global _movies, _embeddings
    
    try:
        # Ensure embeddings are loaded
        if _movies is None or _embeddings is None:
            _movies, _embeddings = load_or_create_embeddings()
        
        logger.info("Creating embedding for user preferences...")
        # Create embedding for the user preferences with minimal memory usage
        user_embedding = model.encode(
            [preferences],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=1,
            device='cpu'
        )[0]
        
        # Clear memory after encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Calculating similarities...")
        # Calculate similarities using cosine similarity
        similarities = np.dot(_embeddings, user_embedding)
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[-num_recommendations:][::-1]
        
        # Return recommended movies
        recommendations = [_movies[i] for i in top_indices]
        
        # Clear memory before returning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Successfully generated {len(recommendations)} recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        raise Exception(f"Error generating recommendations: {str(e)}") 