from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import logging
import gc
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model at module level
try:
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    logger.info("Successfully loaded sentence transformer model")
except Exception as e:
    logger.error(f"Failed to load sentence transformer model: {str(e)}")
    raise

def load_or_create_embeddings() -> List[np.ndarray]:
    """Load embeddings from cache or create new ones."""
    cache_file = os.path.join(os.path.dirname(__file__), 'data/movie_embeddings.pkl')
    
    try:
        # Try to load cached embeddings
        if os.path.exists(cache_file):
            logger.info("Loading cached embeddings...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Load movies from JSON
        json_file = os.path.join(os.path.dirname(__file__), 'data/criterion_movies.json')
        with open(json_file, 'r') as f:
            movies = json.load(f)
        
        if not movies:
            raise ValueError("No movies found in the database")
        
        logger.info(f"Loaded {len(movies)} movies from database")
        
        # Generate embeddings in batches
        batch_size = 8
        embeddings = []
        descriptions = [movie['description'] for movie in movies]
        
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(descriptions) + batch_size - 1)//batch_size}")
            
            # Generate embeddings for the batch
            batch_embeddings = model.encode(batch, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
            
            # Clear memory
            gc.collect()
        
        # Cache the embeddings
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info("Embeddings generated and cached successfully")
        return embeddings
    
    except Exception as e:
        logger.error(f"Error in load_or_create_embeddings: {str(e)}")
        raise

def get_recommendations(user_preferences: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
    """Get movie recommendations based on user preferences."""
    try:
        # Load movies and embeddings
        json_file = os.path.join(os.path.dirname(__file__), 'data/criterion_movies.json')
        with open(json_file, 'r') as f:
            movies = json.load(f)
        
        embeddings = load_or_create_embeddings()
        
        # Encode user preferences
        user_embedding = model.encode(user_preferences, convert_to_numpy=True)
        
        # Calculate similarities
        similarities = np.dot(embeddings, user_embedding)
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[-num_recommendations:][::-1]
        recommendations = [movies[i] for i in top_indices]
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        raise 