"""
Cache management utilities for the movie recommendation system.
"""

import os
import pickle
from pathlib import Path
from typing import Tuple, Dict


def get_cache_path(ratings_file: str, movies_file: str, method: str, cache_type: str = 'similarity') -> str:
    """
    Get the cache file path for the similarity matrix or full model.
    
    Parameters:
    -----------
    ratings_file : str
        Path to the ratings CSV file
    movies_file : str
        Path to the movies CSV file
    method : str
        'user' or 'item' for collaborative filtering method
    cache_type : str
        Type of cache: 'similarity' for similarity matrix only, 'model' for full model
    
    Returns:
    --------
    str
        Path to the cache file
    """
    # Create cache directory if it doesn't exist
    cache_dir = Path('.cache')
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache filename based on ratings file, method, and cache type
    ratings_basename = Path(ratings_file).stem
    movies_basename = Path(movies_file).stem
    cache_filename = f"{cache_type}_{ratings_basename}_{movies_basename}_{method}.pkl"
    return str(cache_dir / cache_filename)


def is_cache_valid(cache_path: str, ratings_file: str, movies_file: str = None, check_movies: bool = False) -> bool:
    """
    Check if the cache file exists and is newer than the data files.
    
    Parameters:
    -----------
    cache_path : str
        Path to the cache file
    ratings_file : str
        Path to the ratings CSV file
    movies_file : str, optional
        Path to the movies CSV file
    check_movies : bool
        If True, also check movies file timestamp (for full model cache)
        
    Returns:
    --------
    bool
        True if cache is valid, False otherwise
    """
    if not os.path.exists(cache_path):
        return False
    
    if not os.path.exists(ratings_file):
        return False
    
    # Check if cache is newer than ratings file
    cache_time = os.path.getmtime(cache_path)
    ratings_time = os.path.getmtime(ratings_file)
    
    if cache_time < ratings_time:
        return False
    
    # If checking movies file (for full model cache)
    if check_movies:
        if movies_file is None:
            return False
        if not os.path.exists(movies_file):
            return False
        movies_time = os.path.getmtime(movies_file)
        if cache_time < movies_time:
            return False
    
    return True


def load_cache(cache_path: str) -> Tuple[bool, Dict]:
    """
    Load data from cache file.
    
    Parameters:
    -----------
    cache_path : str
        Path to the cache file
        
    Returns:
    --------
    Tuple[bool, Dict]
        (success, cached_data) - True if successfully loaded, False otherwise, and cached data
    """
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        return True, cached_data
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return False, {}


def save_cache(cache_path: str, data: Dict, cache_type: str = 'similarity', method: str = 'user'):
    """
    Save data to cache file.
    
    Parameters:
    -----------
    cache_path : str
        Path to the cache file
    data : Dict
        Data to save
    cache_type : str
        Type of cache: 'similarity' or 'model'
    method : str
        Method name for logging
    """
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        if cache_type == 'similarity':
            print(f"Saved {method}-based similarity matrix to cache")
        else:
            print(f"Saved full {method}-based model to cache")
    except Exception as e:
        print(f"Failed to save cache: {e}")


def clear_cache(ratings_file: str = 'ratings_full.csv', movies_file: str = 'movies_clean.csv', 
                method: str = None):
    """
    Clear all cache files for the recommender system.
    
    Parameters:
    -----------
    ratings_file : str
        Ratings file name (default: 'ratings_full.csv')
    movies_file : str
        Movies file name (default: 'movies_clean.csv')
    method : str, optional
        'user' or 'item' to clear cache for specific method, or None to clear all
    """
    import shutil
    
    cache_dir = Path('.cache')
    
    if not cache_dir.exists():
        print("No cache directory found. Nothing to clear.")
        return
    
    if method:
        # Clear cache for specific method
        ratings_basename = Path(ratings_file).stem
        movies_basename = Path(movies_file).stem
        patterns = [
            f"model_{ratings_basename}_{movies_basename}_{method}.pkl",
            f"similarity_{ratings_basename}_{movies_basename}_{method}.pkl"
        ]
        deleted = 0
        for pattern in patterns:
            cache_file = cache_dir / pattern
            if cache_file.exists():
                cache_file.unlink()
                deleted += 1
                print(f"Deleted: {cache_file.name}")
        
        if deleted == 0:
            print(f"No cache files found for {method}-based method.")
        else:
            print(f"Cleared {deleted} cache file(s) for {method}-based method.")
    else:
        # Clear all cache files
        cache_files = list(cache_dir.glob('*.pkl'))
        if not cache_files:
            print("No cache files found. Nothing to clear.")
            return
        
        for cache_file in cache_files:
            cache_file.unlink()
            print(f"Deleted: {cache_file.name}")
        
        print(f"Cleared {len(cache_files)} cache file(s).")

