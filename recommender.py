from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from data_collection import get_criterion_movies
import logging
import os
import json
import pickle
import gc
from functools import lru_cache
import traceback
from tqdm import tqdm
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model at module level
logger.info("Loading sentence transformer model...")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
model.max_seq_length = 128  # Reduce sequence length to save memory
logger.info("Model loaded successfully")

# Global variables for caching
_movies = None
_embeddings = None

def load_or_create_embeddings():
    """Load cached embeddings or create new ones if not available."""
    global _movies, _embeddings
    
    try:
        # Try to load cached embeddings
        if os.path.exists('movie_embeddings.pkl'):
            logger.info("Loading cached embeddings...")
            with open('movie_embeddings.pkl', 'rb') as f:
                _movies, _embeddings = pickle.load(f)
            logger.info(f"Loaded {len(_movies)} movies from cache")
            return _movies, _embeddings
        
        # If no cache, create new embeddings
        logger.info("Creating new embeddings...")
        _movies = get_criterion_movies()
        if not _movies:
            raise ValueError("No movies found in the database")
        
        logger.info(f"Loaded {len(_movies)} movies from database")
        
        # Generate embeddings in batches
        batch_size = 1  # Process one at a time to minimize memory usage
        descriptions = [movie['description'] for movie in _movies]
        embeddings = []
        
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
            
            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        _embeddings = np.array(embeddings)
        logger.info("Embeddings generated successfully")
        
        # Cache the embeddings
        with open('movie_embeddings.pkl', 'wb') as f:
            pickle.dump((_movies, _embeddings), f)
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
        # Create embedding for the user preferences
        user_embedding = model.encode([preferences], show_progress_bar=False, convert_to_numpy=True)[0]
        
        # Clear memory after encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Calculating similarities...")
        # Calculate similarities
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
        logger.error(traceback.format_exc())
        raise Exception(f"Error generating recommendations: {str(e)}") 