from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import json
import os

def get_criterion_movies() -> List[Dict]:
    """
    Get a list of Criterion Collection movies.
    
    Returns:
        List[Dict]: List of movies with their details
    """
    # Check if we have cached data
    cache_file = 'criterion_movies.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    try:
        # Criterion Collection website URL
        url = 'https://www.criterion.com/shop/browse/list'
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        movies = []
        
        # Extract movie information
        for movie in soup.select('.gridFilm'):
            title = movie.select_one('.title').text.strip()
            director = movie.select_one('.director').text.strip()
            description = movie.select_one('.description').text.strip()
            
            movies.append({
                'title': title,
                'director': director,
                'description': description
            })
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(movies, f)
        
        return movies
        
    except Exception as e:
        raise Exception(f"Error collecting movie data: {str(e)}") 