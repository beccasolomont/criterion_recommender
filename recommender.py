from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from data_collection import get_criterion_movies
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info("Loading sentence transformer model...")
        # Load the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Getting Criterion movies...")
        # Get Criterion movies
        criterion_movies = get_criterion_movies()
        
        if not criterion_movies:
            raise ValueError("No movies found in the database. Please check if the data file exists and contains valid movie data.")
        
        logger.info(f"Loaded {len(criterion_movies)} movies from database")
        
        logger.info("Creating embeddings for user preferences...")
        # Create embeddings for the user preferences
        user_embedding = model.encode([preferences])[0]
        
        logger.info("Creating embeddings for movie descriptions...")
        # Create embeddings for movie descriptions
        movie_embeddings = model.encode([movie['description'] for movie in criterion_movies])
        
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
        logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
        raise Exception(f"Error generating recommendations: {str(e)}") 