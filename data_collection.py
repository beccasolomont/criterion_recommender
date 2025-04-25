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
    data_dir = 'data'
    cache_file = os.path.join(data_dir, 'criterion_movies.json')
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at {data_dir}")
    
    # If the cache file doesn't exist, create it with sample data
    if not os.path.exists(cache_file):
        print(f"Creating sample data file at {cache_file}")
        sample_movies = [
            {
                "title": "Seven Samurai",
                "director": "Akira Kurosawa",
                "year": 1954,
                "description": "A veteran samurai, who has fallen on hard times, answers a village's request for protection from bandits. He gathers six other samurai to help him teach the people how to defend themselves, and they wage a desperate battle against the bandits.",
                "spine_number": 2,
                "country": "Japan",
                "genres": ["Action", "Drama", "Adventure"],
                "runtime": 207
            },
            {
                "title": "The 400 Blows",
                "director": "François Truffaut",
                "year": 1959,
                "description": "A young boy, left without attention, delves into a life of petty crime.",
                "spine_number": 5,
                "country": "France",
                "genres": ["Drama"],
                "runtime": 99
            },
            {
                "title": "8½",
                "director": "Federico Fellini",
                "year": 1963,
                "description": "A harried movie director retreats into his memories and fantasies.",
                "spine_number": 140,
                "country": "Italy",
                "genres": ["Drama", "Fantasy"],
                "runtime": 138
            },
            {
                "title": "Persona",
                "director": "Ingmar Bergman",
                "year": 1966,
                "description": "A nurse is put in charge of a mute actress and finds that their personae are melding together.",
                "spine_number": 701,
                "country": "Sweden",
                "genres": ["Drama", "Thriller"],
                "runtime": 85
            },
            {
                "title": "The Seventh Seal",
                "director": "Ingmar Bergman",
                "year": 1957,
                "description": "A man seeks answers about life, death, and the existence of God as he plays chess against the Grim Reaper during the Black Plague.",
                "spine_number": 11,
                "country": "Sweden",
                "genres": ["Drama", "Fantasy"],
                "runtime": 96
            }
        ]
        save_movies_to_json(sample_movies, cache_file)
    
    try:
        with open(cache_file, 'r') as f:
            movies = json.load(f)
            print(f"Successfully loaded {len(movies)} movies from {cache_file}")
            return movies
    except Exception as e:
        print(f"Error loading movies from {cache_file}: {str(e)}")
        return []

def save_movies_to_json(movies, filename='data/criterion_movies.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(movies, f, indent=2)
        print(f"Successfully saved movies to {filename}")
    except Exception as e:
        print(f"Error saving movies to JSON: {str(e)}")

if __name__ == '__main__':
    movies = get_criterion_movies()
    if movies:
        print(f"Successfully loaded {len(movies)} movies from cache")
    else:
        print("No movies were found in the cache") 