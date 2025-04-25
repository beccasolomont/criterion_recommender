from typing import List, Dict
import json
import os

def get_criterion_movies() -> List[Dict]:
    """
    Get a list of Criterion Collection movies from our dataset.
    
    Returns:
        List[Dict]: List of movies with their details
    """
    # Check if we have cached data
    cache_file = 'criterion_movies.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # If no cached data, return empty list
    print("No cached movie data found. Please ensure criterion_movies.json exists.")
    return []

def save_movies_to_json(movies, filename='criterion_movies.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(movies, f, indent=2)
    except Exception as e:
        print(f"Error saving movies to JSON: {str(e)}")

if __name__ == '__main__':
    movies = get_criterion_movies()
    if movies:
        print(f"Successfully loaded {len(movies)} movies from cache")
    else:
        print("No movies were found in the cache") 