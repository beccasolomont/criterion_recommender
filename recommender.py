from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from data_collection import get_criterion_movies

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
        # Load the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get Criterion movies
        criterion_movies = get_criterion_movies()
        
        if not criterion_movies:
            raise ValueError("No movies found in the database")
        
        # Create embeddings for the user preferences
        user_embedding = model.encode([preferences])[0]
        
        # Create embeddings for movie descriptions
        movie_embeddings = model.encode([movie['description'] for movie in criterion_movies])
        
        # Calculate similarities
        similarities = np.dot(movie_embeddings, user_embedding)
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[-num_recommendations:][::-1]
        
        # Return recommended movies
        recommendations = [criterion_movies[i] for i in top_indices]
        
        return recommendations
        
    except Exception as e:
        raise Exception(f"Error generating recommendations: {str(e)}") 