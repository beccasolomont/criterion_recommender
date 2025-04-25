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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = 'cache'
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, 'movie_embeddings.pkl')

# Initialize model at module level with memory optimization
try:
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    # Clear memory after model loading
    gc.collect()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def load_or_create_embeddings():
    """Load cached embeddings or create new ones if not available."""
    try:
        # Create cache directory if it doesn't exist
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory at {cache_dir}")
        
        cache_file = os.path.join(cache_dir, 'movie_embeddings.pkl')
        
        # Try to load cached embeddings
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    logger.info("Loading cached movie embeddings...")
                    movies, embeddings = pickle.load(f)
                    logger.info(f"Loaded {len(movies)} movies from cache")
                    return movies, embeddings
            except Exception as e:
                logger.warning(f"Error loading cached embeddings: {str(e)}")
                logger.warning(traceback.format_exc())
        
        # If no cache or error, create new embeddings
        logger.info("Creating new movie embeddings...")
        movies = get_criterion_movies()
        if not movies:
            raise ValueError("No movies found in the database")
        
        logger.info(f"Loaded {len(movies)} movies from database")
        
        # Generate embeddings in smaller batches with memory cleanup
        batch_size = 2  # Reduced batch size further
        descriptions = [movie['description'] for movie in movies]
        embeddings = []
        
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Generating embeddings"):
            try:
                batch = descriptions[i:i + batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                # Clear memory after each batch
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        embeddings = np.array(embeddings)
        logger.info("Embeddings generated successfully")
        
        # Cache the embeddings
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((movies, embeddings), f)
            logger.info("Cached movie embeddings")
        except Exception as e:
            logger.warning(f"Error caching embeddings: {str(e)}")
            logger.warning(traceback.format_exc())
        
        return movies, embeddings
        
    except Exception as e:
        logger.error(f"Error in load_or_create_embeddings: {str(e)}")
        logger.error(traceback.format_exc())
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
        movies, embeddings = load_or_create_embeddings()
        
        logger.info("Creating embedding for user preferences...")
        # Create embedding for the user preferences
        user_embedding = model.encode([preferences], show_progress_bar=False)[0]
        
        logger.info("Calculating similarities...")
        # Calculate similarities
        similarities = np.dot(embeddings, user_embedding)
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[-num_recommendations:][::-1]
        
        # Return recommended movies
        recommendations = [movies[i] for i in top_indices]
        
        # Clear memory
        gc.collect()
        
        logger.info(f"Successfully generated {len(recommendations)} recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Error generating recommendations: {str(e)}") 