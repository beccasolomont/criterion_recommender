from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import json
import os
import time
import random

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
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Add a random delay between 1-3 seconds
        time.sleep(random.uniform(1, 3))
        
        response = requests.get(
            'https://www.criterion.com/shop/browse/list',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 403:
            raise Exception("Access denied. The website may be blocking automated requests.")
        
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        movies = []
        
        for movie_div in soup.find_all('div', class_='product'):
            try:
                title = movie_div.find('h3').text.strip()
                year = movie_div.find('span', class_='year').text.strip()
                director = movie_div.find('span', class_='director').text.strip()
                
                movies.append({
                    'title': title,
                    'year': year,
                    'director': director
                })
                
                # Add a small delay between processing each movie
                time.sleep(random.uniform(0.5, 1))
                
            except Exception as e:
                print(f"Error processing movie: {str(e)}")
                continue
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(movies, f, indent=2)
        
        return movies
        
    except requests.exceptions.RequestException as e:
        print(f"Error collecting movie data: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return []

def save_movies_to_json(movies, filename='data/criterion_movies.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(movies, f, indent=2)
    except Exception as e:
        print(f"Error saving movies to JSON: {str(e)}")

if __name__ == '__main__':
    movies = get_criterion_movies()
    if movies:
        save_movies_to_json(movies)
        print(f"Successfully collected {len(movies)} movies")
    else:
        print("No movies were collected") 