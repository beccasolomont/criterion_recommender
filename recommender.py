from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from data_collection import get_criterion_movies
import logging
import os
import json
import pickle
from functools import lru_cache
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = 'cache'
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, 'movie_embeddings.pkl')

@lru_cache(maxsize=1)
def get_model():
    """Get the cached sentence transformer model."""
    try:
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_or_create_embeddings():
    """Load cached embeddings or create new ones if not available."""
    try:
        # Create cache directory if it doesn't exist
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            logger.info(f"Created cache directory at {CACHE_DIR}")
        
        # Try to load cached embeddings
        if os.path.exists(EMBEDDINGS_CACHE):
            try:
                with open(EMBEDDINGS_CACHE, 'rb') as f:
                    logger.info("Loading cached movie embeddings...")
                    movies, embeddings = pickle.load(f)
                    logger.info(f"Loaded {len(movies)} movies from cache")
                    return movies, embeddings
            except Exception as e:
                logger.warning(f"Error loading cached embeddings: {str(e)}")
        
        # If no cache or error, create new embeddings
        logger.info("Creating new movie embeddings...")
        criterion_movies = get_criterion_movies()
        if not criterion_movies:
            raise ValueError("No movies found in the database")
        
        logger.info(f"Loaded {len(criterion_movies)} movies from database")
        model = get_model()
        
        logger.info("Generating embeddings for movie descriptions...")
        descriptions = [movie['description'] for movie in criterion_movies]
        embeddings = model.encode(descriptions)
        logger.info("Embeddings generated successfully")
        
        # Cache the embeddings
        try:
            with open(EMBEDDINGS_CACHE, 'wb') as f:
                pickle.dump((criterion_movies, embeddings), f)
            logger.info("Cached movie embeddings")
        except Exception as e:
            logger.warning(f"Error caching embeddings: {str(e)}")
        
        return criterion_movies, embeddings
        
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
    # Input validation
    if not preferences or not isinstance(preferences, str):
        raise ValueError("Invalid preferences provided")
    
    try:
        logger.info("Getting cached movies and embeddings...")
        # Get cached movies and embeddings
        criterion_movies, movie_embeddings = load_or_create_embeddings()
        
        logger.info("Creating embedding for user preferences...")
        # Create embedding for the user preferences
        model = get_model()
        user_embedding = model.encode([preferences])[0]
        
        logger.info("Calculating similarities...")
        # Calculate similarities
        similarities = np.dot(movie_embeddings, user_embedding)
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[-num_recommendations:][::-1]
        
        # Return recommended movies
        recommendations = [criterion_movies[i] for i in top_indices]
        
        logger.info(f"Successfully generated {len(recommendations)} recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Error generating recommendations: {str(e)}") 