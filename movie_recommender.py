"""
Movie Recommendation System using Collaborative Filtering
Supports both User-Based and Item-Based Collaborative Filtering
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Set
import warnings
import ast
import re
import pickle
import os
from pathlib import Path
# Import content-based recommendation functions
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("content_based", "content-based.py")
    content_based_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(content_based_module)
    user_recommendation_ann = content_based_module.user_recommendation_ann
    load_content_based_system = content_based_module.load_content_based_system
except Exception as e:
    # Fallback if import fails
    user_recommendation_ann = None
    load_content_based_system = None
warnings.filterwarnings('ignore')


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering based movie recommendation system.
    Supports both user-based and item-based filtering approaches.
    """
    
    def __init__(self, ratings_file: str, movies_file: str, method: str = 'user', similarity_threshold: float = 0.3):
        """
        Initialize the recommender system.
        
        Parameters:
        -----------
        ratings_file : str
            Path to the ratings CSV file
        movies_file : str
            Path to the movies CSV file
        method : str
            'user' for user-based CF or 'item' for item-based CF
        similarity_threshold : float
            Minimum similarity value to consider (default: 0.3). Similarities below this threshold are ignored.
        """
        self.method = method
        self.ratings_file = ratings_file
        self.movies_file = movies_file
        self.similarity_threshold = similarity_threshold
        self.ratings_df = None
        self.movies_df = None
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.user_item_matrix_sparse = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self.similarity_matrix_type = None  # 'dict' or 'dataframe'
        
        # Content-based filtering components
        self.movies_full_df = None  # Full movies dataframe with genres and tags
        self.movie_embeddings = None  # Movie embeddings from content-based system
        self.content_nn = None  # NearestNeighbors model for content-based recommendations
        self.content_id_to_idx = None  # Mapping from movieId to embedding index
        
    def _get_cache_path(self, cache_type: str = 'similarity') -> str:
        """
        Get the cache file path for the similarity matrix or full model.
        
        Parameters:
        -----------
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
        ratings_basename = Path(self.ratings_file).stem
        movies_basename = Path(self.movies_file).stem
        cache_filename = f"{cache_type}_{ratings_basename}_{movies_basename}_{self.method}.pkl"
        return str(cache_dir / cache_filename)
    
    def _is_cache_valid(self, cache_path: str, check_movies: bool = False) -> bool:
        """
        Check if the cache file exists and is newer than the data files.
        
        Parameters:
        -----------
        cache_path : str
            Path to the cache file
        check_movies : bool
            If True, also check movies file timestamp (for full model cache)
            
        Returns:
        --------
        bool
            True if cache is valid, False otherwise
        """
        if not os.path.exists(cache_path):
            return False
        
        if not os.path.exists(self.ratings_file):
            return False
        
        # Check if cache is newer than ratings file
        cache_time = os.path.getmtime(cache_path)
        ratings_time = os.path.getmtime(self.ratings_file)
        
        if cache_time < ratings_time:
            return False
        
        # If checking movies file (for full model cache)
        if check_movies:
            if not os.path.exists(self.movies_file):
                return False
            movies_time = os.path.getmtime(self.movies_file)
            if cache_time < movies_time:
                return False
        
        return True
    
    def _load_similarity_cache(self, cache_path: str) -> Tuple[bool, Dict]:
        """
        Load similarity matrix from cache.
        
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
    
    def _save_similarity_cache(self, cache_path: str):
        """
        Save similarity matrix to cache.
        
        Parameters:
        -----------
        cache_path : str
            Path to the cache file
        """
        try:
            cache_data = {
                'similarity_matrix': self.similarity_matrix,
                'user_ids': self.user_ids,
                'movie_ids': self.movie_ids
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved {self.method}-based similarity matrix to cache")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _save_model_cache(self, cache_path: str):
        """
        Save full model state to cache (user-item matrix, mappings, similarity, etc.).
        
        Parameters:
        -----------
        cache_path : str
            Path to the cache file
        """
        try:
            cache_data = {
                'user_item_matrix_sparse': self.user_item_matrix_sparse,
                'user_to_idx': self.user_to_idx,
                'movie_to_idx': self.movie_to_idx,
                'user_ids': self.user_ids,
                'movie_ids': self.movie_ids,
                'similarity_matrix': self.similarity_matrix,
                'similarity_matrix_type': self.similarity_matrix_type,
                'movies_df': self.movies_df  # Save movies_df for recommendations
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved full {self.method}-based model to cache")
        except Exception as e:
            print(f"Failed to save model cache: {e}")
    
    def _load_model_cache(self, cache_path: str) -> Tuple[bool, Dict]:
        """
        Load full model state from cache.
        
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
            print(f"Failed to load model cache: {e}")
            return False, {}
    
    def load_model(self, min_year: int = 2015, use_cache: bool = True) -> bool:
        """
        Load data and build model, using cache if available to skip retraining.
        
        Parameters:
        -----------
        min_year : int
            Minimum year to include movies (default: 2015)
        use_cache : bool
            Whether to use cached model if available (default: True)
            
        Returns:
        --------
        bool
            True if model was loaded from cache, False if it was rebuilt
        """
        # Check for full model cache first
        if use_cache:
            model_cache_path = self._get_cache_path('model')
            if self._is_cache_valid(model_cache_path, check_movies=True):
                success, cached_data = self._load_model_cache(model_cache_path)
                if success:
                    # Restore all model state
                    self.user_item_matrix_sparse = cached_data.get('user_item_matrix_sparse')
                    self.user_to_idx = cached_data.get('user_to_idx')
                    self.movie_to_idx = cached_data.get('movie_to_idx')
                    self.user_ids = cached_data.get('user_ids')
                    self.movie_ids = cached_data.get('movie_ids')
                    self.similarity_matrix = cached_data.get('similarity_matrix')
                    self.similarity_matrix_type = cached_data.get('similarity_matrix_type')
                    self.movies_df = cached_data.get('movies_df')
                    
                    # Verify critical components exist
                    if (self.user_item_matrix_sparse is not None and 
                        self.user_to_idx is not None and 
                        self.movie_to_idx is not None and
                        self.similarity_matrix is not None):
                        # Always load ratings_df for methods that need it (like get_movies_by_genres)
                        # This is fast since we're just reading the CSV
                        if self.ratings_df is None:
                            print("Loading ratings data for additional features...")
                            self.ratings_df = pd.read_csv(self.ratings_file)
                            # Filter to match cached movies
                            self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(self.movie_ids)]
                        
                        print(f"Loaded full {self.method}-based model from cache (skipping retraining)")
                        # Content-based system will be loaded lazily when needed
                        # (not loaded here to avoid blocking during model load)
                        return True
                    else:
                        print("Cached model is incomplete, rebuilding...")
        
        # If cache not available or invalid, build model from scratch
        print("Building model from scratch...")
        self.load_data(min_year=min_year)
        self.create_user_item_matrix()
        self.compute_similarity()
        
        # Content-based system will be loaded lazily when needed
        # (not loaded here to avoid blocking during model load)
        
        # Save full model to cache
        if use_cache:
            model_cache_path = self._get_cache_path('model')
            self._save_model_cache(model_cache_path)
        
        return False
    
    def load_data(self, min_year: int = 2015):
        """
        Load ratings and movies data (full dataset).
        
        Parameters:
        -----------
        min_year : int
            Minimum year to include movies (default: 2015). Only movies released after this year will be included.
        """
        print("Loading ratings data...")
        self.ratings_df = pd.read_csv(self.ratings_file)
        print(f"Loaded {len(self.ratings_df):,} ratings")
        
        print("Loading movies data...")
        self.movies_df = pd.read_csv(self.movies_file)
        print(f"Loaded {len(self.movies_df):,} movies")
        
        # Use year_filled instead of year (if year_filled exists)
        if 'year_filled' in self.movies_df.columns:
            if 'year' in self.movies_df.columns:
                # Drop the original year column and rename year_filled to year
                self.movies_df = self.movies_df.drop(columns=['year'])
            self.movies_df = self.movies_df.rename(columns={'year_filled': 'year'})
        
        # Filter movies by year (only movies released after min_year)
        if 'year' in self.movies_df.columns:
            initial_count = len(self.movies_df)
            # Convert year to numeric, handling any non-numeric values
            self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce')
            # Filter movies released after min_year
            self.movies_df = self.movies_df[self.movies_df['year'] > min_year]
            print(f"Filtered to {len(self.movies_df):,} movies released after {min_year} (from {initial_count:,} total)")
        else:
            print("Warning: 'year' column not found, skipping year-based filtering")
        
        # Filter ratings to only include movies that exist in movies_df
        initial_ratings_count = len(self.ratings_df)
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(self.movies_df['movieId'])]
        print(f"Filtered to {len(self.ratings_df):,} ratings after movie matching (from {initial_ratings_count:,} total)")
        
    def create_user_item_matrix(self):
        """Create a user-item rating matrix using sparse matrix directly."""
        print("Creating user-item matrix...")
        
        # Aggregate duplicate ratings (same user-movie pair) by taking mean
        print("Aggregating duplicate ratings...")
        ratings_agg = self.ratings_df.groupby(['userId', 'movieId'])['rating'].mean().reset_index()
        
        # Get unique user and movie IDs
        unique_users = sorted(ratings_agg['userId'].unique())
        unique_movies = sorted(ratings_agg['movieId'].unique())
        
        # Create mapping from IDs to indices
        user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        
        # Store IDs for reference (in sorted order to match matrix indices)
        self.user_ids = np.array(unique_users)
        self.movie_ids = np.array(unique_movies)
        
        print(f"Matrix shape: {len(self.user_ids):,} users × {len(self.movie_ids):,} movies")
        
        # Build sparse matrix directly from aggregated ratings (more memory efficient)
        print("Building sparse matrix from ratings...")
        rows = []
        cols = []
        data = []
        
        # Process ratings in chunks to avoid memory issues
        chunk_size = 1000000
        total_ratings = len(ratings_agg)
        
        for start_idx in range(0, total_ratings, chunk_size):
            end_idx = min(start_idx + chunk_size, total_ratings)
            chunk = ratings_agg.iloc[start_idx:end_idx]
            
            for _, row in chunk.iterrows():
                user_id = row['userId']
                movie_id = row['movieId']
                rating = row['rating']
                
                rows.append(user_to_idx[user_id])
                cols.append(movie_to_idx[movie_id])
                data.append(rating)
            
            if (start_idx + chunk_size) % 5000000 == 0 or end_idx == total_ratings:
                print(f"  Processed {end_idx:,}/{total_ratings:,} ratings ({100*end_idx/total_ratings:.1f}%)", end='\r')
        
        print()  # New line after progress
        
        # Create sparse matrix directly
        self.user_item_matrix_sparse = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.movie_ids))
        )
        
        # Store mappings for later use
        self.user_to_idx = user_to_idx
        self.movie_to_idx = movie_to_idx
        
        print("Sparse matrix created successfully")
    
    def _get_user_rating(self, user_id: int, movie_id: int) -> float:
        """Get rating for a specific user-movie pair from sparse matrix."""
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return 0.0
        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]
        return float(self.user_item_matrix_sparse[user_idx, movie_idx])
    
    def _get_user_ratings_vector(self, user_id: int) -> np.ndarray:
        """Get all ratings for a user as a numpy array."""
        if user_id not in self.user_to_idx:
            return np.zeros(len(self.movie_ids))
        user_idx = self.user_to_idx[user_id]
        return self.user_item_matrix_sparse[user_idx, :].toarray().flatten()
    
    def _get_similarities(self, entity_id: int) -> pd.Series:
        """
        Get similarities for a user or movie from the similarity matrix.
        
        Parameters:
        -----------
        entity_id : int
            User ID (for user-based) or Movie ID (for item-based)
            
        Returns:
        --------
        pd.Series
            Series with entity_id as index and similarity scores as values
        """
        if self.similarity_matrix is None:
            return pd.Series(dtype=float)
        
        if self.similarity_matrix_type == 'dict':
            # Similarity matrix is stored as dict of dicts
            if entity_id in self.similarity_matrix:
                # Return similarities as a Series
                similarities_dict = self.similarity_matrix[entity_id]
                if isinstance(similarities_dict, dict):
                    return pd.Series(similarities_dict)
                else:
                    return pd.Series(dtype=float)
            else:
                # Entity not in similarity matrix (wasn't sampled or is new)
                return pd.Series(dtype=float)
        else:
            # Similarity matrix is a DataFrame (legacy format)
            if entity_id in self.similarity_matrix.index:
                return self.similarity_matrix.loc[entity_id]
            else:
                return pd.Series(dtype=float)
    
    def compute_similarity(self, n_neighbors: int = 10, use_cache: bool = True, 
                         use_mean_centering: bool = False):
        """
        Compute similarity matrix based on the chosen method.
        Uses cache if available and valid to avoid recomputation.
        
        Parameters:
        -----------
        n_neighbors : int
            Number of nearest neighbors to consider (for efficiency)
        use_cache : bool
            Whether to use cached similarity matrix if available (default: True)
        use_mean_centering : bool
            Whether to mean-center ratings before computing similarity (default: False).
            Improves accuracy but significantly slower for large datasets.
        """
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path()
            if self._is_cache_valid(cache_path):
                # Store current IDs for comparison
                current_user_ids = self.user_ids.copy() if self.user_ids is not None else None
                current_movie_ids = self.movie_ids.copy() if self.movie_ids is not None else None
                
                success, cached_data = self._load_similarity_cache(cache_path)
                if success:
                    # Verify that cached IDs match current IDs
                    cached_user_ids = cached_data.get('user_ids')
                    cached_movie_ids = cached_data.get('movie_ids')
                    
                    if (current_user_ids is not None and cached_user_ids is not None and
                        len(current_user_ids) == len(cached_user_ids) and
                        np.array_equal(current_user_ids, cached_user_ids) and
                        current_movie_ids is not None and cached_movie_ids is not None and
                        len(current_movie_ids) == len(cached_movie_ids) and
                        np.array_equal(current_movie_ids, cached_movie_ids)):
                        # IDs match, use cached similarity matrix
                        self.similarity_matrix = cached_data['similarity_matrix']
                        # Check if cached data is dict or DataFrame
                        if isinstance(self.similarity_matrix, dict):
                            self.similarity_matrix_type = 'dict'
                        else:
                            self.similarity_matrix_type = 'dataframe'
                        print(f"Loaded {self.method}-based similarity matrix from cache")
                        return
                    # If IDs don't match, recompute
                    print("Cache IDs don't match current data, recomputing...")
        
        # Compute similarity matrix (sparse, top-k only)
        print(f"Computing {self.method}-based similarity matrix (top-{n_neighbors} neighbors)...")
        
        if self.method == 'user':
            # User-based: compute similarity between users
            n_users = len(self.user_ids)
            
            # Process all users (no sampling)
            sampled_indices = np.arange(n_users)
            sampled_user_ids = self.user_ids
            sampled_matrix = self.user_item_matrix_sparse
            print(f"  Computing similarities for {n_users:,} users...")
            
            # Initialize sparse similarity storage (dictionary of dictionaries)
            similarity_dict = {}
            
            # Process in smaller batches to avoid memory issues
            batch_size = 500  # Reduced batch size
            n_sampled = len(sampled_user_ids)
            total_batches = (n_sampled + batch_size - 1) // batch_size
            
            # Pre-compute user means if mean-centering is enabled
            if use_mean_centering:
                print("  Pre-computing user means for mean-centering...")
                # Compute means directly from sparse matrix (much faster than converting to dense)
                user_means = np.zeros(n_sampled)
                for i in range(n_sampled):
                    user_row = sampled_matrix[i, :]
                    if user_row.nnz > 0:  # If user has ratings
                        user_means[i] = user_row.data.mean()
                print("  Mean-centering enabled (slower but more accurate)")
            else:
                user_means = None
                print("  Mean-centering disabled (faster computation)")
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_sampled)
                batch_local_indices = np.arange(start_idx, end_idx)
                batch_global_indices = sampled_indices[start_idx:end_idx]
                batch_users = sampled_user_ids[start_idx:end_idx]
                
                # Extract batch users (convert to dense only for this batch)
                batch_matrix = sampled_matrix[batch_local_indices, :].toarray()
                
                # Mean-center batch if enabled
                if use_mean_centering and user_means is not None:
                    for i, local_idx in enumerate(batch_local_indices):
                        rated_mask = batch_matrix[i, :] > 0
                        if rated_mask.sum() > 0:
                            batch_matrix[i, rated_mask] -= user_means[local_idx]
                
                # For comparison matrix, use sparse when possible
                if use_mean_centering and user_means is not None:
                    # Need to mean-center the full comparison matrix
                    comparison_matrix = sampled_matrix.toarray()
                    for i in range(len(comparison_matrix)):
                        rated_mask = comparison_matrix[i, :] > 0
                        if rated_mask.sum() > 0:
                            comparison_matrix[i, rated_mask] -= user_means[i]
                    # Compute similarity on mean-centered matrices
                    batch_similarity = cosine_similarity(batch_matrix, comparison_matrix)
                else:
                    # Use sparse matrix directly (much faster - no mean-centering)
                    batch_similarity = cosine_similarity(batch_matrix, sampled_matrix)
                
                # For each user in batch, keep only top-k similarities
                for i, user_id in enumerate(batch_users):
                    user_similarities = batch_similarity[i, :]
                    # Get top-k (excluding self)
                    top_k_local_indices = np.argsort(user_similarities)[::-1]
                    # Exclude self and get top-k
                    top_k_local_indices = [idx for idx in top_k_local_indices if idx != start_idx + i][:n_neighbors]
                    top_k_similarities = user_similarities[top_k_local_indices]
                    
                    # Map back to global user IDs
                    top_k_global_indices = sampled_indices[top_k_local_indices]
                    
                    # Store only similarities above threshold
                    similarity_dict[user_id] = {
                        self.user_ids[global_idx]: float(sim) 
                        for global_idx, sim in zip(top_k_global_indices, top_k_similarities) 
                        if sim >= self.similarity_threshold
                    }
                
                if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                    print(f"  Progress: {batch_idx + 1}/{total_batches} batches ({100*(batch_idx+1)/total_batches:.1f}%)", end='\r')
            
            print()  # New line after progress
            
            # Store as dictionary for memory efficiency (only top-k per user)
            # We'll create a wrapper DataFrame-like object for compatibility
            self.similarity_matrix = similarity_dict
            self.similarity_matrix_type = 'dict'
            print("User similarity matrix computed (sparse, top-k, dict storage)")
            
        elif self.method == 'item':
            # Item-based: compute similarity between movies
            # Transpose the sparse matrix: movies as rows, users as columns
            item_user_matrix_sparse = self.user_item_matrix_sparse.T
            
            n_movies = len(self.movie_ids)
            
            # Process all movies (no sampling)
            sampled_indices = np.arange(n_movies)
            sampled_movie_ids = self.movie_ids
            sampled_matrix = item_user_matrix_sparse
            print(f"  Computing similarities for {n_movies:,} movies...")
            
            # Initialize sparse similarity storage
            similarity_dict = {}
            
            # Process in smaller batches to avoid memory issues
            batch_size = 500  # Reduced batch size
            n_sampled = len(sampled_movie_ids)
            total_batches = (n_sampled + batch_size - 1) // batch_size
            
            # Pre-compute movie means if mean-centering is enabled
            if use_mean_centering:
                print("  Pre-computing movie means for mean-centering...")
                # Compute means directly from sparse matrix (much faster than converting to dense)
                movie_means = np.zeros(n_sampled)
                for i in range(n_sampled):
                    movie_row = sampled_matrix[i, :]
                    if movie_row.nnz > 0:  # If movie has ratings
                        movie_means[i] = movie_row.data.mean()
                print("  Mean-centering enabled (slower but more accurate)")
            else:
                movie_means = None
                print("  Mean-centering disabled (faster computation)")
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_sampled)
                batch_local_indices = np.arange(start_idx, end_idx)
                batch_global_indices = sampled_indices[start_idx:end_idx]
                batch_movies = sampled_movie_ids[start_idx:end_idx]
                
                # Extract batch movies (convert to dense only for this batch)
                batch_matrix = sampled_matrix[batch_local_indices, :].toarray()
                
                # Mean-center batch if enabled
                if use_mean_centering and movie_means is not None:
                    for i, local_idx in enumerate(batch_local_indices):
                        rated_mask = batch_matrix[i, :] > 0
                        if rated_mask.sum() > 0:
                            batch_matrix[i, rated_mask] -= movie_means[local_idx]
                
                # For comparison matrix, use sparse when possible
                if use_mean_centering and movie_means is not None:
                    # Need to mean-center the full comparison matrix
                    comparison_matrix = sampled_matrix.toarray()
                    for i in range(len(comparison_matrix)):
                        rated_mask = comparison_matrix[i, :] > 0
                        if rated_mask.sum() > 0:
                            comparison_matrix[i, rated_mask] -= movie_means[i]
                    # Compute similarity on mean-centered matrices
                    batch_similarity = cosine_similarity(batch_matrix, comparison_matrix)
                else:
                    # Use sparse matrix directly (much faster - no mean-centering)
                    batch_similarity = cosine_similarity(batch_matrix, sampled_matrix)
                
                # For each movie in batch, keep only top-k similarities
                for i, movie_id in enumerate(batch_movies):
                    movie_similarities = batch_similarity[i, :]
                    # Get top-k (excluding self)
                    top_k_local_indices = np.argsort(movie_similarities)[::-1]
                    # Exclude self and get top-k
                    top_k_local_indices = [idx for idx in top_k_local_indices if idx != start_idx + i][:n_neighbors]
                    top_k_similarities = movie_similarities[top_k_local_indices]
                    
                    # Map back to global movie IDs
                    top_k_global_indices = sampled_indices[top_k_local_indices]
                    
                    # Store only similarities above threshold
                    similarity_dict[movie_id] = {
                        self.movie_ids[global_idx]: float(sim) 
                        for global_idx, sim in zip(top_k_global_indices, top_k_similarities) 
                        if sim >= self.similarity_threshold
                    }
                
                if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                    print(f"  Progress: {batch_idx + 1}/{total_batches} batches ({100*(batch_idx+1)/total_batches:.1f}%)", end='\r')
            
            print()  # New line after progress
            
            # Store as dictionary for memory efficiency (only top-k per movie)
            self.similarity_matrix = similarity_dict
            self.similarity_matrix_type = 'dict'
            print("Item similarity matrix computed (sparse, top-k, dict storage)")
        else:
            raise ValueError("Method must be 'user' or 'item'")
        
        # Save to cache
        if use_cache:
            cache_path = self._get_cache_path()
            self._save_similarity_cache(cache_path)
    
    def get_user_ratings(self, user_id: int) -> pd.Series:
        """Get all ratings for a specific user."""
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found in the dataset")
        
        ratings_vector = self._get_user_ratings_vector(user_id)
        ratings_series = pd.Series(ratings_vector, index=self.movie_ids)
        return ratings_series[ratings_series > 0]  # Return only rated movies
    
    def predict_rating_user_based(self, user_id: int, movie_id: int, k: int = 10) -> float:
        """
        Predict rating for a user-movie pair using user-based CF.
        
        Parameters:
        -----------
        user_id : int
            User ID
        movie_id : int
            Movie ID
        k : int
            Number of similar users to consider
            
        Returns:
        --------
        float
            Predicted rating
        """
        if user_id not in self.user_ids:
            return 0.0
        
        if movie_id not in self.movie_ids:
            return 0.0
        
        # Get user's average rating
        user_ratings_vector = self._get_user_ratings_vector(user_id)
        user_mean = user_ratings_vector[user_ratings_vector > 0].mean()
        
        # Get similar users
        # Note: If user_id is not in similarity matrix (wasn't sampled), 
        # we compute similarity on-the-fly
        if self.similarity_matrix_type == 'dict' and user_id not in self.similarity_matrix:
            # Compute similarity for this user on-the-fly against sampled users
            user_vector = self._get_user_ratings_vector(user_id).copy()
            
            # Mean-center user vector
            rated_mask = user_vector > 0
            if rated_mask.sum() > 0:
                user_vector_mean = user_vector[rated_mask].mean()
                user_vector[rated_mask] -= user_vector_mean
            
            user_vector = user_vector.reshape(1, -1)
            
            # Get all users in similarity matrix (sampled users)
            sampled_user_ids = list(self.similarity_matrix.keys())
            sampled_indices = [self.user_to_idx[uid] for uid in sampled_user_ids if uid in self.user_to_idx]
            if len(sampled_indices) == 0:
                return user_mean  # No similar users found
            
            sampled_matrix = self.user_item_matrix_sparse[sampled_indices, :].toarray()
            
            # Mean-center sampled users
            for i in range(len(sampled_matrix)):
                sampled_ratings = sampled_matrix[i, :]
                sampled_rated_mask = sampled_ratings > 0
                if sampled_rated_mask.sum() > 0:
                    sampled_user_mean = sampled_ratings[sampled_rated_mask].mean()
                    sampled_matrix[i, sampled_rated_mask] -= sampled_user_mean
            
            similarities = cosine_similarity(user_vector, sampled_matrix).flatten()
            
            # Get top-k
            top_k_local_indices = np.argsort(similarities)[::-1][:k]
            top_k_similarities = similarities[top_k_local_indices]
            top_k_user_ids = [sampled_user_ids[idx] for idx in top_k_local_indices]
            
            # Create temporary similarity dict for this user
            user_similarities_dict = {
                uid: float(sim) for uid, sim in zip(top_k_user_ids, top_k_similarities) if sim >= self.similarity_threshold
            }
        else:
            user_similarities = self._get_similarities(user_id).sort_values(ascending=False)
            user_similarities = user_similarities[user_similarities.index != user_id]  # Exclude self
            top_k_users = user_similarities.head(k)
            user_similarities_dict = {uid: sim for uid, sim in top_k_users.items() if sim >= self.similarity_threshold}
        
        # Calculate weighted average using similarities
        top_k_users = user_similarities_dict
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for similar_user_id, similarity in top_k_users.items():
            if similarity < self.similarity_threshold:
                continue
                
            similar_user_rating = self._get_user_rating(similar_user_id, movie_id)
            
            if similar_user_rating > 0:
                similar_user_ratings_vector = self._get_user_ratings_vector(similar_user_id)
                similar_user_mean = similar_user_ratings_vector[similar_user_ratings_vector > 0].mean()
                
                numerator += similarity * (similar_user_rating - similar_user_mean)
                denominator += abs(similarity)
        
        if denominator == 0:
            return user_mean
        
        predicted_rating = user_mean + (numerator / denominator)
        # Clamp rating between 0.5 and 5.0
        predicted_rating = max(0.5, min(5.0, predicted_rating))
        
        return predicted_rating
    
    def predict_rating_item_based(self, user_id: int, movie_id: int, k: int = 10) -> float:
        """
        Predict rating for a user-movie pair using item-based CF.
        
        Parameters:
        -----------
        user_id : int
            User ID
        movie_id : int
            Movie ID
        k : int
            Number of similar movies to consider
            
        Returns:
        --------
        float
            Predicted rating
        """
        if user_id not in self.user_ids:
            return 0.0
        
        if movie_id not in self.movie_ids:
            return 0.0
        
        # Get user's ratings
        user_ratings_vector = self._get_user_ratings_vector(user_id)
        user_rated_movie_indices = np.where(user_ratings_vector > 0)[0]
        user_rated_movie_ids = self.movie_ids[user_rated_movie_indices]
        user_rated_movie_ratings = user_ratings_vector[user_rated_movie_indices]
        
        if len(user_rated_movie_ids) == 0:
            return 0.0
        
        # Get similar movies
        # Note: If movie_id is not in similarity matrix (wasn't sampled),
        # we compute similarity on-the-fly
        if self.similarity_matrix_type == 'dict' and movie_id not in self.similarity_matrix:
            # Compute similarity for this movie on-the-fly against sampled movies
            if movie_id not in self.movie_to_idx:
                return 0.0
            
            movie_idx = self.movie_to_idx[movie_id]
            movie_vector = self.user_item_matrix_sparse[:, movie_idx].toarray().flatten()
            
            # Mean-center movie vector
            rated_mask = movie_vector > 0
            if rated_mask.sum() > 0:
                movie_vector_mean = movie_vector[rated_mask].mean()
                movie_vector[rated_mask] -= movie_vector_mean
            
            movie_vector = movie_vector.reshape(1, -1)
            
            # Get all movies in similarity matrix (sampled movies)
            sampled_movie_ids = list(self.similarity_matrix.keys())
            sampled_indices = [self.movie_to_idx[mid] for mid in sampled_movie_ids if mid in self.movie_to_idx]
            if len(sampled_indices) == 0:
                return user_rated_movie_ratings.mean() if len(user_rated_movie_ratings) > 0 else 0.0
            
            sampled_matrix = self.user_item_matrix_sparse[:, sampled_indices].toarray().T
            
            # Mean-center sampled movies
            for i in range(len(sampled_matrix)):
                sampled_ratings = sampled_matrix[i, :]
                sampled_rated_mask = sampled_ratings > 0
                if sampled_rated_mask.sum() > 0:
                    sampled_movie_mean = sampled_ratings[sampled_rated_mask].mean()
                    sampled_matrix[i, sampled_rated_mask] -= sampled_movie_mean
            
            similarities = cosine_similarity(movie_vector, sampled_matrix).flatten()
            
            # Get top-k
            top_k_local_indices = np.argsort(similarities)[::-1][:k]
            top_k_similarities = similarities[top_k_local_indices]
            top_k_movie_ids = [sampled_movie_ids[idx] for idx in top_k_local_indices]
            
            # Create temporary similarity dict for this movie
            movie_similarities_dict = {
                mid: float(sim) for mid, sim in zip(top_k_movie_ids, top_k_similarities) if sim >= self.similarity_threshold
            }
        else:
            if self.similarity_matrix_type == 'dict':
                if movie_id not in self.similarity_matrix:
                    return 0.0
            else:
                if movie_id not in self.similarity_matrix.index:
                    return 0.0
            
            movie_similarities = self._get_similarities(movie_id).sort_values(ascending=False)
            movie_similarities = movie_similarities[movie_similarities.index != movie_id]  # Exclude self
            movie_similarities_dict = {mid: sim for mid, sim in movie_similarities.head(k).items() if sim >= self.similarity_threshold}
        
        # Only consider movies the user has rated
        rated_movie_similarities = {
            mid: sim for mid, sim in movie_similarities_dict.items() 
            if mid in user_rated_movie_ids
        }
        # Sort by similarity and take top-k
        top_k_movies = dict(sorted(rated_movie_similarities.items(), key=lambda x: x[1], reverse=True)[:k])
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for similar_movie_id, similarity in top_k_movies.items():
            if similarity < self.similarity_threshold:
                continue
            
            # Find the rating for this movie
            movie_idx = np.where(user_rated_movie_ids == similar_movie_id)[0]
            if len(movie_idx) > 0:
                user_rating = user_rated_movie_ratings[movie_idx[0]]
                numerator += similarity * user_rating
                denominator += abs(similarity)
        
        if denominator == 0:
            # Return mean of only rated movies (consistent with user-based CF)
            return user_rated_movie_ratings.mean() if len(user_rated_movie_ratings) > 0 else 0.0
        
        predicted_rating = numerator / denominator
        # Clamp rating between 0.5 and 5.0
        predicted_rating = max(0.5, min(5.0, predicted_rating))
        
        return predicted_rating
    
    def recommend_movies(self, user_id: int, n_recommendations: int = 10, k: int = 10) -> List[Tuple[int, str, float]]:
        """
        Generate movie recommendations for a user.
        
        Parameters:
        -----------
        user_id : int
            User ID
        n_recommendations : int
            Number of recommendations to return
        k : int
            Number of neighbors to consider for prediction
            
        Returns:
        --------
        List[Tuple[int, str, float]]
            List of (movie_id, title, predicted_rating) tuples
        """
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found in the dataset")
        
        # Get movies the user has already rated
        user_ratings = self.get_user_ratings(user_id)
        rated_movie_ids = set(user_ratings.index)
        
        # Get candidate movies (popular/well-rated movies that user hasn't rated)
        # This is much more efficient than predicting for all 20k+ movies
        candidate_movies = self.get_popular_movies(n=500)  # Get top 500 popular movies
        candidate_movie_ids = {movie_id for movie_id, _, _ in candidate_movies}
        
        # Filter to only unrated movies
        unrated_candidates = candidate_movie_ids - rated_movie_ids
        
        if len(unrated_candidates) == 0:
            # Fallback: use all unrated movies if no popular candidates
            all_movie_ids = set(self.movie_ids)
            unrated_candidates = all_movie_ids - rated_movie_ids
            unrated_candidates = list(unrated_candidates)[:1000]  # Limit to 1000
        else:
            unrated_candidates = list(unrated_candidates)
        
        # Predict ratings for candidate movies
        predictions = []
        
        for movie_id in unrated_candidates:
            if self.method == 'user':
                pred_rating = self.predict_rating_user_based(user_id, movie_id, k)
            else:
                pred_rating = self.predict_rating_item_based(user_id, movie_id, k)
            
            if pred_rating > 0:
                predictions.append((movie_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recommendations = predictions[:n_recommendations]
        
        # Add movie titles
        recommendations = []
        for movie_id, pred_rating in top_recommendations:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_info) > 0:
                title = movie_info.iloc[0]['title']
                recommendations.append((movie_id, title, pred_rating))
        
        return recommendations
    
    def get_similar_users(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get users similar to the given user.
        
        Parameters:
        -----------
        user_id : int
            User ID
        n : int
            Number of similar users to return
            
        Returns:
        --------
        List[Tuple[int, float]]
            List of (user_id, similarity_score) tuples
        """
        if self.method != 'user':
            raise ValueError("This method only works with user-based CF")
        
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found")
        
        user_similarities = self._get_similarities(user_id).sort_values(ascending=False)
        user_similarities = user_similarities[user_similarities.index != user_id]
        
        return [(uid, float(sim)) for uid, sim in user_similarities.head(n).items() if sim >= self.similarity_threshold]
    
    def get_similar_movies(self, movie_id: int, n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get movies similar to the given movie.
        
        Parameters:
        -----------
        movie_id : int
            Movie ID
        n : int
            Number of similar movies to return
            
        Returns:
        --------
        List[Tuple[int, str, float]]
            List of (movie_id, title, similarity_score) tuples
        """
        if self.method != 'item':
            raise ValueError("This method only works with item-based CF")
        
        if movie_id not in self.movie_ids:
            raise ValueError(f"Movie {movie_id} not found")
        
        movie_similarities = self._get_similarities(movie_id).sort_values(ascending=False)
        movie_similarities = movie_similarities[movie_similarities.index != movie_id]
        
        similar_movies = []
        for similar_movie_id, similarity in movie_similarities.head(n).items():
            if similarity >= self.similarity_threshold:
                movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie_id]
                if len(movie_info) > 0:
                    title = movie_info.iloc[0]['title']
                    similar_movies.append((similar_movie_id, title, float(similarity)))
        
        return similar_movies
    
    def _load_content_based_system(self, movies_full_file: str = 'movies_full.csv', 
                                   embeddings_file: str = 'movie_embeddings.npy'):
        """
        Load content-based recommendation system components using content-based.py.
        Automatically generates embeddings if they don't exist.
        
        Parameters:
        -----------
        movies_full_file : str
            Path to movies_full.csv file with genres and tags
        embeddings_file : str
            Path to movie_embeddings.npy file
        """
        if self.movie_embeddings is not None and self.content_nn is not None:
            return  # Already loaded
        
        if load_content_based_system is None:
            raise ImportError("Could not import content-based module. Please ensure content-based.py exists.")
        
        print("Loading content-based system...")
        # Use the function from content-based.py (auto-generates embeddings if needed)
        self.movies_full_df, self.movie_embeddings, self.content_nn = load_content_based_system(
            movies_full_file, embeddings_file, auto_generate=True
        )
        
        # Create mapping from movieId to embedding index
        self.content_id_to_idx = pd.Series(
            self.movies_full_df.index.values, 
            index=self.movies_full_df["movieId"]
        ).to_dict()
        
        print("Content-based system ready.")
    
    def recommend_movies_content_based(self, user_id: int, n_recommendations: int = 10, 
                                       candidates: int = 500) -> List[Tuple[int, str, float]]:
        """
        Generate content-based movie recommendations for a user.
        
        Parameters:
        -----------
        user_id : int
            User ID
        n_recommendations : int
            Number of recommendations to return
        candidates : int
            Number of candidate movies to consider
            
        Returns:
        --------
        List[Tuple[int, str, float]]
            List of (movie_id, title, similarity_score) tuples
        """
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found in the dataset")
        
        # Load content-based system if not already loaded
        if self.movie_embeddings is None:
            self._load_content_based_system()
        
        # Get user's ratings
        user_ratings = self.get_user_ratings(user_id)
        rated_movie_ids = set(user_ratings.index)
        
        # Use all rated movies (not just "liked" ones) to build user profile
        # This way we use all the rating information, not just high ratings
        all_rated_movie_ids = list(user_ratings.index)
        
        if len(all_rated_movie_ids) == 0:
            return []
        
        # Create rating dictionary for weighted averaging
        movie_ratings_dict = {mid: rating for mid, rating in user_ratings.items()}
        
        if user_recommendation_ann is None:
            raise ImportError("Could not import user_recommendation_ann from content-based.py")
        
        # Use the function from content-based.py with all ratings and rating weights
        content_recs = user_recommendation_ann(
            self.movies_full_df,
            self.movie_embeddings,
            self.content_nn,
            all_rated_movie_ids,
            k=n_recommendations,
            candidates=candidates,
            movie_ratings=movie_ratings_dict  # Pass ratings for weighted average
        )
        
        # Filter to only movies in current dataset (after year filtering) and get titles
        recommendations = []
        for movie_id, clean_title, sim_score in content_recs:
            # Filter to only movies in current dataset
            if movie_id in self.movie_ids:
                # Get title from movies_df (current dataset)
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
                if len(movie_info) > 0:
                    title = movie_info.iloc[0]['title']
                    recommendations.append((movie_id, title, float(sim_score)))
        
        return recommendations
    
    def add_new_user_ratings(self, user_ratings: Dict[int, float]) -> int:
        """
        Add a new user with their ratings to the system.
        
        Parameters:
        -----------
        user_ratings : Dict[int, float]
            Dictionary mapping movie_id to rating (0.5-5.0)
            
        Returns:
        --------
        int
            New user ID assigned
        """
        # Generate new user ID (max existing + 1)
        new_user_id = int(self.user_ids.max()) + 1 if len(self.user_ids) > 0 else 1
        
        # Create new row data for sparse matrix
        new_row_data = []
        new_row_cols = []
        
        # Add ratings
        for movie_id, rating in user_ratings.items():
            if movie_id in self.movie_to_idx:
                new_row_cols.append(self.movie_to_idx[movie_id])
                new_row_data.append(rating)
        
        # Create new row as sparse matrix
        new_row = csr_matrix((new_row_data, ([0] * len(new_row_data), new_row_cols)), 
                     shape=(1, len(self.movie_ids)))
        
        # Append new row to sparse matrix
        self.user_item_matrix_sparse = vstack([self.user_item_matrix_sparse, new_row])
        
        # Update user IDs and mapping
        self.user_ids = np.append(self.user_ids, new_user_id)
        self.user_to_idx[new_user_id] = len(self.user_ids) - 1
        
        # For user-based CF, compute similarities only for the new user (not all users)
        # This is much faster than recomputing the entire similarity matrix
        if self.method == 'user':
            if self.similarity_matrix is None:
                # If no similarity matrix exists, we need to compute it
                print("Computing similarity matrix (first time)...")
                self.compute_similarity(use_cache=False)
            else:
                # Just compute similarities for the new user against existing users
                print("Computing similarities for new user (fast incremental update)...")
                new_user_idx = self.user_to_idx[new_user_id]
                new_user_vector = self.user_item_matrix_sparse[new_user_idx, :].toarray().flatten()
                
                # Get all existing users (excluding the new one)
                existing_user_indices = np.arange(len(self.user_ids) - 1)
                existing_user_matrix = self.user_item_matrix_sparse[existing_user_indices, :]
                
                # Compute similarity for new user against all existing users
                new_user_vector_2d = new_user_vector.reshape(1, -1)
                similarities = cosine_similarity(new_user_vector_2d, existing_user_matrix).flatten()
                
                # Get top-k similar users
                top_k_indices = np.argsort(similarities)[::-1][:10]  # top 10
                top_k_similarities = similarities[top_k_indices]
                top_k_user_ids = self.user_ids[existing_user_indices[top_k_indices]]
                
                # Add new user's similarities to the similarity matrix
                if self.similarity_matrix_type == 'dict':
                    # Store only similarities above threshold
                    self.similarity_matrix[new_user_id] = {
                        uid: float(sim) for uid, sim in zip(top_k_user_ids, top_k_similarities)
                        if sim >= self.similarity_threshold
                    }
                    print(f"✓ Added new user {new_user_id} with {len(self.similarity_matrix[new_user_id])} similar users")
                else:
                    # If using DataFrame, convert to dict format
                    self.similarity_matrix = {}
                    self.similarity_matrix_type = 'dict'
                    self.similarity_matrix[new_user_id] = {
                        uid: float(sim) for uid, sim in zip(top_k_user_ids, top_k_similarities)
                        if sim >= self.similarity_threshold
                    }
        
        return new_user_id
    
    def get_popular_movies(self, n: int = 50) -> List[Tuple[int, str, float]]:
        """
        Get popular movies (by number of ratings and average rating).
        
        Parameters:
        -----------
        n : int
            Number of popular movies to return
            
        Returns:
        --------
        List[Tuple[int, str, float]]
            List of (movie_id, title, average_rating) tuples
        """
        # Calculate average rating and count for each movie
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
        
        # Filter movies with at least 10 ratings
        movie_stats = movie_stats[movie_stats['num_ratings'] >= 10]
        
        # Sort by average rating and number of ratings
        movie_stats = movie_stats.sort_values(
            ['avg_rating', 'num_ratings'],
            ascending=[False, False]
        )
        
        # Get top N
        top_movies = movie_stats.head(n)
        
        # Add movie titles
        popular_movies = []
        for _, row in top_movies.iterrows():
            movie_id = int(row['movieId'])
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_info) > 0:
                title = movie_info.iloc[0]['title']
                avg_rating = row['avg_rating']
                popular_movies.append((movie_id, title, avg_rating))
        
        return popular_movies
    
    def get_all_genres(self) -> List[str]:
        """
        Get all unique genres from the movies dataset.
        
        Returns:
        --------
        List[str]
            List of unique genre names
        """
        all_genres = set()
        
        for genres_str in self.movies_df['genres']:
            if pd.isna(genres_str):
                continue
            
            # Parse genres string (can be list string or regular string)
            try:
                # Try to parse as list
                if isinstance(genres_str, str) and genres_str.startswith('['):
                    genres_list = ast.literal_eval(genres_str)
                else:
                    # Handle pipe-separated or other formats
                    genres_list = [g.strip() for g in str(genres_str).split('|')]
                
                all_genres.update(genres_list)
            except:
                # Fallback: split by common delimiters
                genres_list = re.split(r'[|,\[\]]+', str(genres_str))
                all_genres.update([g.strip().strip("'\"") for g in genres_list if g.strip()])
        
        # Remove empty strings and sort
        all_genres = sorted([g for g in all_genres if g and g != '(no genres listed)'])
        return all_genres
    
    def get_movies_by_genres(self, genres: List[str], n: int = 50) -> List[Tuple[int, str, float, str]]:
        """
        Get popular movies filtered by genres.
        
        Parameters:
        -----------
        genres : List[str]
            List of genre names to filter by
        n : int
            Number of movies to return
            
        Returns:
        --------
        List[Tuple[int, str, float, str]]
            List of (movie_id, title, average_rating, genres_str) tuples
        """
        # Calculate average rating and count for each movie
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
        
        # Filter movies with at least 10 ratings
        movie_stats = movie_stats[movie_stats['num_ratings'] >= 10]
        
        # Filter by genres
        genre_movies = []
        for _, row in self.movies_df.iterrows():
            movie_id = row['movieId']
            genres_str = row['genres']
            
            if pd.isna(genres_str):
                continue
            
            # Parse genres
            try:
                if isinstance(genres_str, str) and genres_str.startswith('['):
                    movie_genres = ast.literal_eval(genres_str)
                else:
                    movie_genres = [g.strip() for g in str(genres_str).split('|')]
                
                # Check if movie has any of the requested genres
                if any(genre in movie_genres for genre in genres):
                    genre_movies.append(movie_id)
            except:
                continue
        
        # Filter movie_stats to only include genre movies
        movie_stats = movie_stats[movie_stats['movieId'].isin(genre_movies)]
        
        # Remove duplicates (in case a movie appears multiple times)
        movie_stats = movie_stats.drop_duplicates(subset=['movieId'])
        
        # Sort by average rating and number of ratings
        movie_stats = movie_stats.sort_values(
            ['avg_rating', 'num_ratings'],
            ascending=[False, False]
        )
        
        # Get top N
        top_movies = movie_stats.head(n)
        
        # Add movie titles and genres
        genre_movies_list = []
        for _, row in top_movies.iterrows():
            movie_id = int(row['movieId'])
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_info) > 0:
                title = movie_info.iloc[0]['title']
                avg_rating = row['avg_rating']
                genres_str = movie_info.iloc[0]['genres']
                genre_movies_list.append((movie_id, title, avg_rating, genres_str))
        
        return genre_movies_list
    
    def search_movies(self, query: str, n: int = 20) -> List[Tuple[int, str, str]]:
        """
        Search for movies by title.
        
        Parameters:
        -----------
        query : str
            Search query (case-insensitive)
        n : int
            Maximum number of results to return
            
        Returns:
        --------
        List[Tuple[int, str, str]]
            List of (movie_id, title, genres) tuples
        """
        query_lower = query.lower()
        
        # Search in titles
        matches = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(query_lower, na=False)
        ]
        
        # Limit results
        matches = matches.head(n)
        
        results = []
        for _, row in matches.iterrows():
            movie_id = int(row['movieId'])
            title = row['title']
            genres = row['genres'] if not pd.isna(row['genres']) else 'Unknown'
            results.append((movie_id, title, genres))
        
        return results
    
    def evaluate_rating_prediction(self, test_ratings: pd.DataFrame, k: int = 10, 
                                   max_samples: int = 10000) -> Dict[str, float]:
        """
        Evaluate rating prediction accuracy using RMSE and MAE.
        
        Parameters:
        -----------
        test_ratings : pd.DataFrame
            DataFrame with columns: userId, movieId, rating
        k : int
            Number of neighbors to consider for prediction
        max_samples : int
            Maximum number of test ratings to evaluate (default: 10000)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with 'rmse' and 'mae' metrics
        """
        predictions = []
        actuals = []
        
        # Sample if too many test ratings
        if len(test_ratings) > max_samples:
            test_ratings = test_ratings.sample(n=max_samples, random_state=42).reset_index(drop=True)
            print(f"  Sampling {max_samples:,} test ratings for evaluation...")
        
        total = len(test_ratings)
        print(f"  Evaluating {total:,} rating predictions...")
        
        for idx, (_, row) in enumerate(test_ratings.iterrows(), 1):
            if idx % 1000 == 0 or idx == total:
                print(f"  Progress: {idx:,}/{total:,} ({100*idx/total:.1f}%)", end='\r')
            
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            if user_id not in self.user_ids or movie_id not in self.movie_ids:
                continue
            
            # Predict rating
            if self.method == 'user':
                pred_rating = self.predict_rating_user_based(user_id, movie_id, k)
            else:
                pred_rating = self.predict_rating_item_based(user_id, movie_id, k)
            
            if pred_rating > 0:
                predictions.append(pred_rating)
                actuals.append(actual_rating)
        
        print()  # New line after progress
        
        if len(predictions) == 0:
            return {'rmse': float('inf'), 'mae': float('inf'), 'n_samples': 0}
        
        # Calculate RMSE and MAE
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(predictions)
        }
    
    def evaluate_ranking(self, test_ratings: pd.DataFrame, k: int = 10, 
                        top_k: int = 1, threshold: float = 4.0, max_users: int = 100) -> Dict[str, float]:
        """
        Evaluate ranking quality using Precision@K, Recall@K, F1@K, and NDCG@K.
        
        Parameters:
        -----------
        test_ratings : pd.DataFrame
            DataFrame with columns: userId, movieId, rating
        k : int
            Number of neighbors to consider for prediction
        top_k : int
            Number of top recommendations to consider
        threshold : float
            Rating threshold to consider a movie as "relevant" (default: 4.0)
        max_users : int
            Maximum number of users to evaluate (default: 100)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with precision, recall, f1, and ndcg metrics
        """
        user_precisions = []
        user_recalls = []
        user_ndcgs = []
        
        # Group test ratings by user
        user_groups = list(test_ratings.groupby('userId'))
        
        # Limit number of users to evaluate
        if len(user_groups) > max_users:
            import random
            random.seed(42)
            user_groups = random.sample(user_groups, max_users)
            print(f"  Sampling {max_users} users for ranking evaluation...")
        
        total = len(user_groups)
        print(f"  Evaluating ranking for {total} users...")
        
        for idx, (user_id, user_test) in enumerate(user_groups, 1):
            if idx % 10 == 0 or idx == total:
                print(f"  Progress: {idx}/{total} ({100*idx/total:.1f}%)", end='\r')
            
            if user_id not in self.user_ids:
                continue
            
            # Get relevant movies (highly rated in test set)
            relevant_movies = set(
                user_test[user_test['rating'] >= threshold]['movieId'].values
            )
            
            if len(relevant_movies) == 0:
                continue
            
            # Get recommendations
            try:
                recommendations = self.recommend_movies(user_id, n_recommendations=top_k, k=k)
                recommended_movies = set([rec[0] for rec in recommendations])
            except:
                continue
            
            # Calculate metrics
            if len(recommended_movies) > 0:
                # Precision@K: relevant items in recommendations / total recommendations
                precision = len(relevant_movies & recommended_movies) / len(recommended_movies)
                user_precisions.append(precision)
                
                # Recall@K: relevant items in recommendations / total relevant items
                recall = len(relevant_movies & recommended_movies) / len(relevant_movies)
                user_recalls.append(recall)
                
                # NDCG@K
                ndcg = self._calculate_ndcg(recommendations, relevant_movies, top_k)
                user_ndcgs.append(ndcg)
        
        print()  # New line after progress
        
        if len(user_precisions) == 0:
            return {
                'precision@k': 0.0,
                'recall@k': 0.0,
                'f1@k': 0.0,
                'ndcg@k': 0.0,
                'n_users': 0
            }
        
        precision_avg = np.mean(user_precisions)
        recall_avg = np.mean(user_recalls)
        f1 = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg) if (precision_avg + recall_avg) > 0 else 0.0
        ndcg_avg = np.mean(user_ndcgs)
        
        return {
            'precision@k': precision_avg,
            'recall@k': recall_avg,
            'f1@k': f1,
            'ndcg@k': ndcg_avg,
            'n_users': len(user_precisions)
        }
    
    def _calculate_ndcg(self, recommendations: List[Tuple[int, str, float]], 
                       relevant_movies: Set[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        Parameters:
        -----------
        recommendations : List[Tuple[int, str, float]]
            List of (movie_id, title, predicted_rating) tuples
        relevant_movies : Set[int]
            Set of relevant movie IDs
        k : int
            Number of recommendations to consider
            
        Returns:
        --------
        float
            NDCG@K score
        """
        # DCG: sum of (relevance / log2(position + 1))
        dcg = 0.0
        for i, (movie_id, _, _) in enumerate(recommendations[:k], 1):
            if movie_id in relevant_movies:
                relevance = 1.0  # Binary relevance
                dcg += relevance / np.log2(i + 1)
        
        # IDCG: ideal DCG (all relevant items at top)
        n_relevant = min(len(relevant_movies), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_coverage(self, n_recommendations: int = 10, k: int = 10, 
                          sample_users: int = 100) -> Dict[str, float]:
        """
        Calculate recommendation coverage (fraction of movies that can be recommended).
        
        Parameters:
        -----------
        n_recommendations : int
            Number of recommendations per user
        k : int
            Number of neighbors to consider
        sample_users : int
            Number of users to sample for coverage calculation
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with coverage metrics
        """
        recommended_movies = set()
        total_users = min(sample_users, len(self.user_ids))
        
        # Sample users
        np.random.seed(42)
        sampled_user_ids = np.random.choice(self.user_ids, size=total_users, replace=False)
        
        for idx, user_id in enumerate(sampled_user_ids, 1):
            if idx % 10 == 0 or idx == total_users:
                print(f"  Progress: {idx}/{total_users} ({100*idx/total_users:.1f}%)", end='\r')
            try:
                recommendations = self.recommend_movies(user_id, n_recommendations=n_recommendations, k=k)
                recommended_movies.update([rec[0] for rec in recommendations])
            except:
                continue
        
        if total_users > 0:
            print()  # New line after progress
        
        coverage = len(recommended_movies) / len(self.movie_ids) if len(self.movie_ids) > 0 else 0.0
        
        return {
            'coverage': coverage,
            'unique_movies_recommended': len(recommended_movies),
            'total_movies': len(self.movie_ids),
            'n_users_sampled': total_users
        }
    
    def calculate_diversity(self, recommendations: List[Tuple[int, str, float]]) -> float:
        """
        Calculate diversity of recommendations using cosine similarity between movies.
        Lower similarity = higher diversity.
        
        Parameters:
        -----------
        recommendations : List[Tuple[int, str, float]]
            List of (movie_id, title, predicted_rating) tuples
            
        Returns:
        --------
        float
            Diversity score (1 - average similarity)
        """
        if len(recommendations) < 2:
            return 0.0
        
        movie_ids = [rec[0] for rec in recommendations]
        
        # Get movie vectors from user-item matrix (item-based) or similarity matrix
        if self.method == 'item' and hasattr(self, 'similarity_matrix'):
            # Use item similarity matrix
            similarities = []
            for i, movie_id1 in enumerate(movie_ids):
                for movie_id2 in movie_ids[i+1:]:
                    sim = self._get_similarity(movie_id1, movie_id2)
                    if sim >= self.similarity_threshold:
                        similarities.append(sim)
            
            if len(similarities) > 0:
                avg_similarity = np.mean(similarities)
                return 1.0 - avg_similarity  # Diversity = 1 - similarity
        
        return 0.5  # Default if can't calculate
    
    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split ratings into training and test sets.
        
        Parameters:
        -----------
        test_size : float
            Proportion of ratings to use for testing (default: 0.2)
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (train_ratings, test_ratings) DataFrames
        """
        if self.ratings_df is None:
            raise ValueError("Data must be loaded first. Call load_data() before train_test_split().")
        
        # Shuffle ratings
        ratings_shuffled = self.ratings_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split by user to ensure users appear in both sets
        n_test = int(len(ratings_shuffled) * test_size)
        test_ratings = ratings_shuffled.head(n_test)
        train_ratings = ratings_shuffled.tail(len(ratings_shuffled) - n_test)
        
        return train_ratings, test_ratings
    
    def evaluate_model(self, test_ratings: pd.DataFrame, k: int = 10, 
                      top_k: int = 1, threshold: float = 4.0,
                      max_samples: int = 10000, max_users: int = 100) -> Dict[str, float]:
        """
        Comprehensive model evaluation combining rating prediction and ranking metrics.
        
        Parameters:
        -----------
        test_ratings : pd.DataFrame
            DataFrame with columns: userId, movieId, rating
        k : int
            Number of neighbors to consider for prediction
        top_k : int
            Number of top recommendations to consider for ranking metrics
        threshold : float
            Rating threshold to consider a movie as "relevant" (default: 4.0)
        max_samples : int
            Maximum number of test ratings to evaluate (default: 10000)
        max_users : int
            Maximum number of users to evaluate for ranking (default: 100)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with all evaluation metrics
        """
        print("Evaluating model...")
        print("=" * 60)
        
        # Rating prediction metrics
        print("\n[1/2] Calculating rating prediction metrics (RMSE, MAE)...")
        rating_metrics = self.evaluate_rating_prediction(test_ratings, k=k, max_samples=max_samples)
        
        # Ranking metrics
        print("\n[2/2] Calculating ranking metrics (Precision@K, Recall@K, NDCG@K)...")
        ranking_metrics = self.evaluate_ranking(test_ratings, k=k, top_k=top_k, threshold=threshold, max_users=max_users)
        
        # Combine all metrics
        all_metrics = {
            **rating_metrics,
            **ranking_metrics
        }
        
        return all_metrics
    
    def evaluate_content_based(self, train_ratings: pd.DataFrame, test_ratings: pd.DataFrame,
                              top_k: int = 1, threshold: float = 4.0, max_users: int = 100) -> Dict[str, float]:
        """
        Evaluate content-based recommendation quality using ranking metrics.
        
        Parameters:
        -----------
        train_ratings : pd.DataFrame
            Training ratings DataFrame (used to build user profiles)
        test_ratings : pd.DataFrame
            Test ratings DataFrame (used to determine relevant movies)
        top_k : int
            Number of top recommendations to consider (default: 10)
        threshold : float
            Rating threshold to consider a movie as "relevant" (default: 4.0)
        max_users : int
            Maximum number of users to evaluate (default: 100)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with precision, recall, f1, and ndcg metrics
        """
        # Load content-based system if not already loaded
        if self.movie_embeddings is None:
            try:
                self._load_content_based_system()
            except Exception as e:
                print(f"  Error loading content-based system: {e}")
                return {
                    'precision@k': 0.0,
                    'recall@k': 0.0,
                    'f1@k': 0.0,
                    'ndcg@k': 0.0,
                    'n_users': 0
                }
        
        user_precisions = []
        user_recalls = []
        user_ndcgs = []
        
        # Group test ratings by user
        user_groups = list(test_ratings.groupby('userId'))
        
        # Limit number of users to evaluate
        if len(user_groups) > max_users:
            import random
            random.seed(42)
            user_groups = random.sample(user_groups, max_users)
            print(f"  Sampling {max_users} users for content-based evaluation...")
        
        total = len(user_groups)
        print(f"  Evaluating content-based recommendations for {total} users...")
        
        for idx, (user_id, user_test) in enumerate(user_groups, 1):
            if idx % 10 == 0 or idx == total:
                print(f"  Progress: {idx}/{total} ({100*idx/total:.1f}%)", end='\r')
            
            # Get relevant movies (highly rated in test set)
            relevant_movies = set(
                user_test[user_test['rating'] >= threshold]['movieId'].values
            )
            
            if len(relevant_movies) == 0:
                continue
            
            # Get user's training ratings to build profile
            user_train = train_ratings[train_ratings['userId'] == user_id]
            if len(user_train) == 0:
                continue  # User has no training ratings, can't build profile
            
            # Check if user exists in current model (needed for recommend_movies_content_based)
            if user_id not in self.user_to_idx:
                continue
            
            try:
                # Get content-based recommendations
                # The method will use get_user_ratings which gets ratings from the model
                # Since we built the model from training data, it will use training ratings
                recommendations = self.recommend_movies_content_based(
                    user_id, 
                    n_recommendations=top_k,
                    candidates=500
                )
                recommended_movies = set([rec[0] for rec in recommendations])
                
            except Exception as e:
                # Skip if recommendation fails
                continue
            
            # Calculate metrics
            if len(recommended_movies) > 0:
                # Precision@K: relevant items in recommendations / total recommendations
                precision = len(relevant_movies & recommended_movies) / len(recommended_movies)
                user_precisions.append(precision)
                
                # Recall@K: relevant items in recommendations / total relevant items
                recall = len(relevant_movies & recommended_movies) / len(relevant_movies)
                user_recalls.append(recall)
                
                # NDCG@K
                ndcg = self._calculate_ndcg(recommendations, relevant_movies, top_k)
                user_ndcgs.append(ndcg)
        
        print()  # New line after progress
        
        if len(user_precisions) == 0:
            return {
                'precision@k': 0.0,
                'recall@k': 0.0,
                'f1@k': 0.0,
                'ndcg@k': 0.0,
                'n_users': 0
            }
        
        precision_avg = np.mean(user_precisions)
        recall_avg = np.mean(user_recalls)
        f1 = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg) if (precision_avg + recall_avg) > 0 else 0.0
        ndcg_avg = np.mean(user_ndcgs)
        
        return {
            'precision@k': precision_avg,
            'recall@k': recall_avg,
            'f1@k': f1,
            'ndcg@k': ndcg_avg,
            'n_users': len(user_precisions)
        }
    
    def evaluate_thresholds(self, test_ratings: pd.DataFrame, k: int = 10, 
                           top_k: int = 1, max_users: int = 100,
                           threshold_min: float = 0.0, threshold_max: float = 5.0, 
                           threshold_step: float = 0.5) -> Dict[float, Dict[str, float]]:
        """
        Evaluate ranking metrics across multiple relevance thresholds.
        
        Parameters:
        -----------
        test_ratings : pd.DataFrame
            DataFrame with columns: userId, movieId, rating
        k : int
            Number of neighbors to consider for prediction
        top_k : int
            Number of top recommendations to consider
        max_users : int
            Maximum number of users to evaluate (default: 100)
        threshold_min : float
            Minimum threshold value (default: 0.0)
        threshold_max : float
            Maximum threshold value (default: 5.0)
        threshold_step : float
            Step size for threshold values (default: 0.5)
            
        Returns:
        --------
        Dict[float, Dict[str, float]]
            Dictionary mapping threshold values to their metrics (precision, recall, f1)
        """
        thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
        results = {}
        
        print(f"\nEvaluating across {len(thresholds)} thresholds: {threshold_min} to {threshold_max} (step {threshold_step})")
        print("=" * 60)
        
        for threshold in thresholds:
            threshold = round(threshold, 1)  # Round to avoid floating point issues
            print(f"\nEvaluating with threshold >= {threshold:.1f}...")
            
            # Evaluate ranking metrics for this threshold
            ranking_metrics = self.evaluate_ranking(
                test_ratings, 
                k=k, 
                top_k=top_k, 
                threshold=threshold, 
                max_users=max_users
            )
            
            results[threshold] = {
                'precision@k': ranking_metrics['precision@k'],
                'recall@k': ranking_metrics['recall@k'],
                'f1@k': ranking_metrics['f1@k'],
                'n_users': ranking_metrics['n_users']
            }
        
        return results
    
    def evaluate_content_based_thresholds(self, train_ratings: pd.DataFrame, 
                                          test_ratings: pd.DataFrame,
                                          top_k: int = 1, max_users: int = 100,
                                          threshold_min: float = 0.0, threshold_max: float = 5.0, 
                                          threshold_step: float = 0.5) -> Dict[float, Dict[str, float]]:
        """
        Evaluate content-based recommendations across multiple relevance thresholds.
        
        Parameters:
        -----------
        train_ratings : pd.DataFrame
            Training ratings DataFrame (used to build user profiles)
        test_ratings : pd.DataFrame
            Test ratings DataFrame (used to determine relevant movies)
        top_k : int
            Number of top recommendations to consider (default: 10)
        max_users : int
            Maximum number of users to evaluate (default: 100)
        threshold_min : float
            Minimum threshold value (default: 0.0)
        threshold_max : float
            Maximum threshold value (default: 5.0)
        threshold_step : float
            Step size for threshold values (default: 0.5)
            
        Returns:
        --------
        Dict[float, Dict[str, float]]
            Dictionary mapping threshold values to their metrics (precision, recall, f1)
        """
        thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
        results = {}
        
        print(f"\nEvaluating across {len(thresholds)} thresholds: {threshold_min} to {threshold_max} (step {threshold_step})")
        print("=" * 60)
        
        for threshold in thresholds:
            threshold = round(threshold, 1)  # Round to avoid floating point issues
            print(f"\nEvaluating with threshold >= {threshold:.1f}...")
            
            # Evaluate content-based metrics for this threshold
            metrics = self.evaluate_content_based(
                train_ratings=train_ratings,
                test_ratings=test_ratings,
                top_k=top_k,
                threshold=threshold,
                max_users=max_users
            )
            
            results[threshold] = {
                'precision@k': metrics['precision@k'],
                'recall@k': metrics['recall@k'],
                'f1@k': metrics['f1@k'],
                'n_users': metrics['n_users']
            }
        
        return results


def interactive_new_user_mode(recommender, show_both_cf=True):
    """
    Interactive mode for new users to rate movies and get recommendations.
    
    Parameters:
    -----------
    recommender : CollaborativeFilteringRecommender
        Initialized recommender object (primary method)
    show_both_cf : bool
        If True, shows both user-based and item-based CF recommendations (default: True)
    """
    print("\n" + "=" * 60)
    print("Welcome! New User Movie Recommendation System")
    print("=" * 60)
    print("\nTo get personalized recommendations, please rate at least 5 movies.")
    print("You can rate movies from 0.5 to 5.0 (in increments of 0.5)")
    print("Rating scale:")
    print("  0.5 - 1.5: Didn't like it")
    print("  2.0 - 2.5: It was okay")
    print("  3.0 - 3.5: Liked it")
    print("  4.0 - 4.5: Really liked it")
    print("  5.0: Loved it")
    print("\nType 'skip' to skip a movie, 'done' when you've rated enough movies.")
    print("-" * 60)
    
    user_ratings = {}
    min_ratings = 5
    
    # Step 1: Ask about genre preferences
    print("\n" + "=" * 60)
    print("Step 1: Select Your Favorite Genres (Optional)")
    print("=" * 60)
    print("\nThis will help us show you movies from your preferred genres in Step 3.")
    print("If you skip this, we'll show you popular movies from all genres.")
    
    all_genres = recommender.get_all_genres()
    print(f"\nAvailable genres ({len(all_genres)} total):")
    
    # Display genres in columns for better readability
    for i, genre in enumerate(all_genres, 1):
        print(f"  {i:2d}. {genre:<20}", end="")
        if i % 3 == 0:
            print()
    if len(all_genres) % 3 != 0:
        print()
    
    print("\nEnter the numbers of genres you like (comma-separated, e.g., 1,5,12)")
    print("Or press Enter to skip and see all popular movies:")
    
    selected_genres = []
    while True:
        try:
            genre_input = input("Your genre selections: ").strip()
            
            if not genre_input:
                print("Skipping genre selection. Showing all popular movies.")
                break
            
            genre_indices = [int(x.strip()) for x in genre_input.split(',')]
            selected_genres = [all_genres[i-1] for i in genre_indices if 1 <= i <= len(all_genres)]
            
            if selected_genres:
                print(f"\n✓ Selected genres: {', '.join(selected_genres)}")
                break
            else:
                print("Invalid selection. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter numbers separated by commas.")
    
    # Step 2: Option to search for specific movies
    print("\n" + "=" * 60)
    print("Step 2: Search for Movies You Know (Optional)")
    print("=" * 60)
    print("\nYou can search for specific movies you've watched to rate them.")
    print("Type 'skip' to proceed to genre-based recommendations.")
    
    search_ratings = {}
    while True:
        search_query = input("\nSearch for a movie (or 'skip'/'done'): ").strip()
        
        if search_query.lower() in ['skip', 'done', '']:
            break
        
        search_results = recommender.search_movies(search_query, n=10)
        
        if not search_results:
            print("No movies found. Try a different search term.")
            continue
        
        print(f"\nFound {len(search_results)} movies:")
        for i, (movie_id, title, genres) in enumerate(search_results, 1):
            print(f"  {i}. {title}")
        
        try:
            choice = input("\nSelect a movie number to rate (or 'skip'): ").strip().lower()
            
            if choice == 'skip':
                continue
            
            movie_idx = int(choice) - 1
            if 0 <= movie_idx < len(search_results):
                movie_id, title, genres = search_results[movie_idx]
                
                while True:
                    try:
                        rating = float(input(f"Rate '{title}' (0.5-5.0): "))
                        if 0.5 <= rating <= 5.0:
                            rating = round(rating * 2) / 2
                            search_ratings[movie_id] = rating
                            user_ratings[movie_id] = rating
                            print(f"✓ Rated {title} with {rating:.1f}/5.0")
                            break
                        else:
                            print("Rating must be between 0.5 and 5.0")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
            
            if len(user_ratings) >= min_ratings:
                more = input(f"\nYou've rated {len(user_ratings)} movies. Rate more? (y/n): ").strip().lower()
                if more != 'y':
                    break
        except (ValueError, IndexError):
            print("Invalid selection.")
    
    # Step 3: Show movies from selected genres (or popular movies)
    if len(user_ratings) < min_ratings:
        print("\n" + "=" * 60)
        print("Step 3: Rate Movies to Build Your Profile")
        print("=" * 60)
        
        if selected_genres:
            movies_to_rate = recommender.get_movies_by_genres(selected_genres, n=30)
            print(f"\nHere are popular movies from your selected genres ({', '.join(selected_genres)}):")
            print("(Based on your genre preferences from Step 1)")
        else:
            movies_to_rate = [(m[0], m[1], m[2], '') for m in recommender.get_popular_movies(n=30)]
            print(f"\nHere are some popular movies from all genres:")
            print("(You can skip movies you haven't seen)")
        
        # Deduplicate movies by movie_id (in case of duplicates)
        seen_movie_ids = set()
        deduplicated_movies = []
        for movie_data in movies_to_rate:
            movie_id = movie_data[0]
            if movie_id not in seen_movie_ids:
                seen_movie_ids.add(movie_id)
                deduplicated_movies.append(movie_data)
        movies_to_rate = deduplicated_movies
        
        print(f"(You've already rated {len(user_ratings)} movies. Need at least {min_ratings} total.)\n")
        
        for i, movie_data in enumerate(movies_to_rate, 1):
            if len(user_ratings) >= 20:  # Limit to 20 ratings max
                break
            
            if len(movie_data) == 4:
                movie_id, title, avg_rating, genres_str = movie_data
            else:
                # Handle 3-tuple from get_popular_movies
                movie_id, title, avg_rating = movie_data
                genres_str = ''
            
            # Skip if already rated
            if movie_id in user_ratings:
                continue
            
            print(f"\n[{i}] {title}")
            print(f"    Average rating: {avg_rating:.2f}/5.0")
            if genres_str:
                try:
                    if isinstance(genres_str, str) and genres_str.startswith('['):
                        genres_list = ast.literal_eval(genres_str)
                        print(f"    Genres: {', '.join(genres_list)}")
                except:
                    pass
            
            rating_input = None
            while True:
                try:
                    rating_input = input(f"Your rating (0.5-5.0, or 'skip'/'done'): ").strip().lower()
                    
                    if rating_input == 'done':
                        if len(user_ratings) >= min_ratings:
                            break
                        else:
                            print(f"Please rate at least {min_ratings} movies before finishing.")
                            continue
                    
                    if rating_input == 'skip':
                        break
                    
                    rating = float(rating_input)
                    
                    # Validate rating
                    if rating < 0.5 or rating > 5.0:
                        print("Rating must be between 0.5 and 5.0")
                        continue
                    
                    # Round to nearest 0.5
                    rating = round(rating * 2) / 2
                    
                    user_ratings[movie_id] = rating
                    print(f"✓ Rated {title} with {rating:.1f}/5.0")
                    break
                    
                except ValueError:
                    print("Invalid input. Please enter a number between 0.5 and 5.0, or 'skip'/'done'")
                    continue
            
            if rating_input == 'done' and len(user_ratings) >= min_ratings:
                break
    
    if len(user_ratings) < min_ratings:
        print(f"\n⚠ You've only rated {len(user_ratings)} movies. Need at least {min_ratings} for recommendations.")
        print("Exiting interactive mode.")
        return
    
    print(f"\n✓ Thank you! You've rated {len(user_ratings)} movies.")
    print("\nAdding your ratings to the system...")
    
    # Add new user to the system
    new_user_id = recommender.add_new_user_ratings(user_ratings)
    
    print(f"✓ Your user ID is: {new_user_id}")
    print("\nGenerating personalized recommendations...")
    print("-" * 60)
    
    # Get collaborative filtering recommendations from primary recommender
    cf_recommendations_primary = recommender.recommend_movies(new_user_id, n_recommendations=10, k=10)
    
    # Get recommendations from the other CF method if show_both_cf is True
    cf_recommendations_secondary = []
    if show_both_cf:
        other_method = 'item' if recommender.method == 'user' else 'user'
        print(f"\nLoading {other_method}-based model for additional recommendations...")
        try:
            # Create the other recommender
            other_recommender = CollaborativeFilteringRecommender(
                ratings_file=recommender.ratings_file,
                movies_file=recommender.movies_file,
                method=other_method,
                similarity_threshold=recommender.similarity_threshold
            )
            # Load model (should use cache)
            other_recommender.load_model(use_cache=True)
            # Add the new user to the other recommender as well
            other_new_user_id = other_recommender.add_new_user_ratings(user_ratings)
            # Get recommendations
            cf_recommendations_secondary = other_recommender.recommend_movies(other_new_user_id, n_recommendations=10, k=10)
        except Exception as e:
            print(f"Note: Could not load {other_method}-based recommendations: {e}")
    
    # Get content-based recommendations
    try:
        cb_recommendations = recommender.recommend_movies_content_based(new_user_id, n_recommendations=10)
    except Exception as e:
        print(f"Note: Content-based recommendations unavailable: {e}")
        cb_recommendations = []
    
    # Display Collaborative Filtering recommendations (primary method)
    if cf_recommendations_primary:
        method_name = "User-Based" if recommender.method == 'user' else "Item-Based"
        print(f"\n Collaborative Filtering Recommendations ({method_name}):")
        print("=" * 60)
        for i, (movie_id, title, pred_rating) in enumerate(cf_recommendations_primary, 1):
            print(f"{i:2d}. {title}")
            print(f"    Predicted Rating: {pred_rating:.2f}/5.0")
    else:
        print("\n No collaborative filtering recommendations available.")
    
    # Display Collaborative Filtering recommendations (secondary method)
    if show_both_cf and cf_recommendations_secondary:
        other_method_name = "Item-Based" if recommender.method == 'user' else "User-Based"
        print(f"\n Collaborative Filtering Recommendations ({other_method_name}):")
        print("=" * 60)
        for i, (movie_id, title, pred_rating) in enumerate(cf_recommendations_secondary, 1):
            print(f"{i:2d}. {title}")
            print(f"    Predicted Rating: {pred_rating:.2f}/5.0")
    
    # Display Content-Based recommendations
    if cb_recommendations:
        print(f"\n Content-Based Recommendations:")
        print("=" * 60)
        for i, (movie_id, title, sim_score) in enumerate(cb_recommendations, 1):
            print(f"{i:2d}. {title}")
            print(f"    Similarity Score: {sim_score:.3f}")
    else:
        print("\n No content-based recommendations available.")
    
    print("\n" + "=" * 60)
    print("Thank you for using our recommendation system!")
    print("=" * 60)


def main(user_id: int = 1, method: str = None):
    """
    Example usage of the Collaborative Filtering Recommender.
    
    Parameters:
    -----------
    user_id : int
        User ID to get recommendations for (default: 1)
    method : str, optional
        'user' for user-based CF, 'item' for item-based CF, or None for both (default: None)
    """
    # If no method specified, show both (for backward compatibility)
    if method is None:
        methods_to_run = ['user', 'item']
    else:
        methods_to_run = [method]
    
    for cf_method in methods_to_run:
        method_name = "User-Based" if cf_method == 'user' else "Item-Based"
        print("=" * 60)
        print(f"Movie Recommendation System - {method_name} Collaborative Filtering")
        print("=" * 60)
        
        recommender = CollaborativeFilteringRecommender(
            ratings_file='ratings_full.csv',
            movies_file='movies_clean.csv',
            method=cf_method
        )
        
        # Load model (uses cache if available to skip retraining)
        print("Loading model...")
        recommender.load_model(use_cache=True)
        print("Model loaded successfully!")
        
        # Check if user exists (use user_to_idx mapping which is more reliable)
        if recommender.user_to_idx is None or user_id not in recommender.user_to_idx:
            print(f"\n⚠ Error: User {user_id} not found in the dataset.")
            if recommender.user_ids is not None and len(recommender.user_ids) > 0:
                print(f"Available user IDs range from {recommender.user_ids.min()} to {recommender.user_ids.max()}")
                print(f"Note: Only users who have rated movies released after 2015 are included.")
                print(f"      User {user_id} may not have any ratings for movies after 2015.")
                # Show some example user IDs that do exist
                if len(recommender.user_ids) > 0:
                    sample_ids = recommender.user_ids[:5] if len(recommender.user_ids) >= 5 else recommender.user_ids
                    print(f"Example user IDs that exist: {', '.join(map(str, sample_ids))}")
            continue
        
        # Get recommendations for specified user
        print(f"\nGenerating recommendations for User {user_id}...")
        
        # Show user's current ratings (only for first method to avoid repetition)
        if cf_method == methods_to_run[0]:
            user_ratings = recommender.get_user_ratings(user_id)
            print(f"\nUser {user_id} has rated {len(user_ratings)} movies:")
            for movie_id, rating in user_ratings.head(10).items():
                movie_info = recommender.movies_df[recommender.movies_df['movieId'] == movie_id]
                if len(movie_info) > 0:
                    print(f"  - {movie_info.iloc[0]['title']}: {rating:.1f}")
        
        # Get collaborative filtering recommendations (primary method)
        cf_recommendations = recommender.recommend_movies(user_id, n_recommendations=10)
        
        print(f"\nTop 10 Collaborative Filtering Recommendations for User {user_id} ({method_name}):")
        print("-" * 60)
        for i, (movie_id, title, pred_rating) in enumerate(cf_recommendations, 1):
            print(f"{i:2d}. {title} (Predicted Rating: {pred_rating:.2f})")
        
        # Get recommendations from the other CF method if showing both
        if method is None or len(methods_to_run) > 1:
            # Already showing both, skip
            pass
        else:
            # Show the other method as well
            other_method = 'item' if cf_method == 'user' else 'user'
            other_method_name = "Item-Based" if other_method == 'item' else "User-Based"
            try:
                other_recommender = CollaborativeFilteringRecommender(
                    ratings_file='ratings_full.csv',
                    movies_file='movies_clean.csv',
                    method=other_method
                )
                other_recommender.load_model(use_cache=True)
                other_cf_recommendations = other_recommender.recommend_movies(user_id, n_recommendations=10)
                
                if other_cf_recommendations:
                    print(f"\nTop 10 Collaborative Filtering Recommendations for User {user_id} ({other_method_name}):")
                    print("-" * 60)
                    for i, (movie_id, title, pred_rating) in enumerate(other_cf_recommendations, 1):
                        print(f"{i:2d}. {title} (Predicted Rating: {pred_rating:.2f})")
            except Exception as e:
                print(f"\nNote: {other_method_name} recommendations unavailable: {e}")
        
        # Get content-based recommendations (only show once, not per method)
        if cf_method == methods_to_run[0]:  # Only show content-based once
            try:
                cb_recommendations = recommender.recommend_movies_content_based(user_id, n_recommendations=10)
                if cb_recommendations:
                    print(f"\nTop 10 Content-Based Recommendations for User {user_id}:")
                    print("-" * 60)
                    for i, (movie_id, title, sim_score) in enumerate(cb_recommendations, 1):
                        print(f"{i:2d}. {title} (Similarity: {sim_score:.3f})")
            except Exception as e:
                print(f"\nNote: Content-based recommendations unavailable: {e}")
        
        # Show similar movies (only for item-based CF)
        if cf_method == 'item':
            print("\n" + "=" * 60)
            print("Finding Similar Movies")
            print("=" * 60)
            
            # Try to find a movie from user's top recommendations or their rated movies
            sample_movie_id = None
            
            # First, try to use the top recommendation
            if cf_recommendations and len(cf_recommendations) > 0:
                sample_movie_id = cf_recommendations[0][0]  # Use top recommended movie
            else:
                # Fall back to a movie the user has rated highly
                user_rated = recommender.get_user_ratings(user_id)
                if len(user_rated) > 0:
                    # Get the highest rated movie
                    sample_movie_id = user_rated.idxmax()
            
            # If still no movie found, use a default popular movie
            if sample_movie_id is None or sample_movie_id not in recommender.movie_ids:
                sample_movie_id = 1  # Default to Toy Story as fallback
            
            similar_movies = recommender.get_similar_movies(sample_movie_id, n=10)
            
            movie_info = recommender.movies_df[recommender.movies_df['movieId'] == sample_movie_id]
            if len(movie_info) > 0:
                movie_title = movie_info.iloc[0]['title']
                print(f"\nMovies similar to '{movie_title}':")
                print("-" * 60)
                for i, (movie_id, title, similarity) in enumerate(similar_movies, 1):
                    print(f"{i:2d}. {title} (Similarity: {similarity:.3f})")
        
        # Add spacing between methods if showing both
        if len(methods_to_run) > 1 and cf_method != methods_to_run[-1]:
            print("\n")


def main_evaluate(method: str = 'user', test_size: float = 0.2, threshold: float = 4.0):
    """
    Evaluate the recommendation model using train/test split.
    
    Parameters:
    -----------
    method : str
        'user' for user-based CF, 'item' for item-based CF, or 'content' for content-based
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    threshold : float
        Rating threshold to consider a movie as "relevant" (default: 4.0)
    """
    if method == 'content':
        # Content-based evaluation
        print("=" * 60)
        print("Model Evaluation - Content-Based Filtering")
        print("=" * 60)
        
        # Initialize recommender (method doesn't matter for content-based, but we need it for structure)
        recommender = CollaborativeFilteringRecommender(
            ratings_file='ratings_full.csv',
            movies_file='movies_clean.csv',
            method='user'  # Method doesn't affect content-based, but required for initialization
        )
        
        # Load data
        print("\nLoading data...")
        recommender.load_data()
        
        # Split into train and test
        print(f"\nSplitting data into train/test sets (test_size={test_size})...")
        train_ratings, test_ratings = recommender.train_test_split(test_size=test_size)
        
        print(f"Training set: {len(train_ratings):,} ratings")
        print(f"Test set: {len(test_ratings):,} ratings")
        
        # Build user-item matrix from training data (needed for get_user_ratings)
        print("\nBuilding user-item matrix from training data...")
        recommender.ratings_df = train_ratings
        recommender.create_user_item_matrix()
        
        # Load content-based system
        print("\nLoading content-based system...")
        try:
            recommender._load_content_based_system()
        except Exception as e:
            print(f"Error loading content-based system: {e}")
            print("Please ensure movies_full.csv and movie_embeddings.npy exist.")
            return
        
        # Evaluate content-based recommendations
        print("\n" + "=" * 60)
        print(f"Using relevance threshold: rating >= {threshold}")
        print("\nEvaluating content-based recommendations...")
        print("=" * 60)
        
        metrics = recommender.evaluate_content_based(
            train_ratings=train_ratings,
            test_ratings=test_ratings,
            top_k=1,
            threshold=threshold,
            max_users=100
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("Content-Based Evaluation Results")
        print("=" * 60)
        print("\nRanking Metrics (Top-1 Recommendations):")
        print(f"  Precision@1: {metrics['precision@k']:.4f}")
        print(f"  Recall@1:    {metrics['recall@k']:.4f}")
        print(f"  F1@1:        {metrics['f1@k']:.4f}")
        print(f"  NDCG@1:      {metrics['ndcg@k']:.4f}")
        print(f"  Users evaluated: {metrics['n_users']:,}")
        print("\nNote: Content-based filtering does not predict ratings,")
        print("      so only ranking metrics are available.")
        print("=" * 60)
        
        # Evaluate across multiple thresholds
        print("\n" + "=" * 60)
        print("Threshold Analysis (0.0 to 5.0, step 0.5)")
        print("=" * 60)
        threshold_results = recommender.evaluate_content_based_thresholds(
            train_ratings=train_ratings,
            test_ratings=test_ratings,
            top_k=1,
            max_users=100,
            threshold_min=0.0,
            threshold_max=5.0,
            threshold_step=0.5
        )
        
        # Display threshold results
        print("\n" + "=" * 60)
        print("Threshold Analysis Results")
        print("=" * 60)
        print(f"\n{'Threshold':<12} {'Precision@1':<15} {'Recall@1':<15} {'F1@1':<15} {'Users':<10}")
        print("-" * 70)
        for threshold in sorted(threshold_results.keys()):
            metrics = threshold_results[threshold]
            print(f"{threshold:<12.1f} {metrics['precision@k']:<15.4f} {metrics['recall@k']:<15.4f} "
                  f"{metrics['f1@k']:<15.4f} {metrics['n_users']:<10}")
        print("=" * 60)
        
    else:
        # Collaborative filtering evaluation (user-based or item-based)
        method_name = "User-Based" if method == 'user' else "Item-Based"
        print("=" * 60)
        print(f"Model Evaluation - {method_name} Collaborative Filtering")
        print("=" * 60)
        
        # Initialize recommender
        recommender = CollaborativeFilteringRecommender(
            ratings_file='ratings_full.csv',
            movies_file='movies_clean.csv',
            method=method
        )
        
        # Load data
        print("\nLoading data...")
        recommender.load_data()
        
        # Split into train and test
        print(f"\nSplitting data into train/test sets (test_size={test_size})...")
        train_ratings, test_ratings = recommender.train_test_split(test_size=test_size)
        
        print(f"Training set: {len(train_ratings):,} ratings")
        print(f"Test set: {len(test_ratings):,} ratings")
        
        # Use training data to build the model
        print("\nBuilding model on training data...")
        # Save original ratings
        original_ratings = recommender.ratings_df.copy()
        # Replace with training data
        recommender.ratings_df = train_ratings
        # Rebuild matrices with training data
        recommender.create_user_item_matrix()
        recommender.compute_similarity()
        
        # Evaluate on test set
        print("\n" + "=" * 60)
        print(f"Using relevance threshold: rating >= {threshold}")
        metrics = recommender.evaluate_model(test_ratings, k=10, top_k=1, threshold=threshold)
        
        # Display results
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print("\nRating Prediction Metrics:")
        print(f"  RMSE (Root Mean Square Error): {metrics['rmse']:.4f}")
        print(f"  MAE (Mean Absolute Error):     {metrics['mae']:.4f}")
        print(f"  Number of predictions:         {metrics['n_samples']:,}")
        
        print("\nRanking Metrics (Top-1 Recommendations):")
        print(f"  Precision@1: {metrics['precision@k']:.4f}")
        print(f"  Recall@1:    {metrics['recall@k']:.4f}")
        print(f"  F1@1:        {metrics['f1@k']:.4f}")
        print(f"  NDCG@1:      {metrics['ndcg@k']:.4f}")
        print(f"  Users evaluated: {metrics['n_users']:,}")
        
        print("=" * 60)
        
        # Evaluate across multiple thresholds
        print("\n" + "=" * 60)
        print("Threshold Analysis (0.0 to 5.0, step 0.5)")
        print("=" * 60)
        threshold_results = recommender.evaluate_thresholds(
            test_ratings=test_ratings,
            k=10,
            top_k=1,
            max_users=100,
            threshold_min=0.0,
            threshold_max=5.0,
            threshold_step=0.5
        )
        
        # Display threshold results
        print("\n" + "=" * 60)
        print("Threshold Analysis Results")
        print("=" * 60)
        print(f"\n{'Threshold':<12} {'Precision@1':<15} {'Recall@1':<15} {'F1@1':<15} {'Users':<10}")
        print("-" * 70)
        for threshold in sorted(threshold_results.keys()):
            metrics = threshold_results[threshold]
            print(f"{threshold:<12.1f} {metrics['precision@k']:<15.4f} {metrics['recall@k']:<15.4f} "
                  f"{metrics['f1@k']:<15.4f} {metrics['n_users']:<10}")
        print("=" * 60)


def main_interactive(method: str = 'user', show_both_cf: bool = True):
    """
    Interactive mode for new users.
    
    Parameters:
    -----------
    method : str
        'user' for user-based CF or 'item' for item-based CF (primary method)
    show_both_cf : bool
        If True, shows both user-based and item-based CF recommendations (default: True)
    """
    method_name = "User-Based" if method == 'user' else "Item-Based"
    print("=" * 60)
    print(f"Movie Recommendation System - Interactive Mode")
    if show_both_cf:
        print("(Showing both User-Based and Item-Based CF recommendations)")
    else:
        print(f"({method_name} only)")
    print("=" * 60)
    
    # Initialize recommender
    recommender = CollaborativeFilteringRecommender(
        ratings_file='ratings_full.csv',
        movies_file='movies_clean.csv',
        method=method
    )
    
    # Load model (uses cache if available to skip retraining)
    print("\nLoading model...")
    recommender.load_model(use_cache=True)
    
    # Run interactive mode
    interactive_new_user_mode(recommender, show_both_cf=show_both_cf)


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
    from pathlib import Path
    
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


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Movie Recommendation System using Collaborative Filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run standard mode for user 1 (default)
  python movie_recommender.py
  
  # Run standard mode for a specific user
  python movie_recommender.py --user-id 42
  
  # Evaluate all methods (user-based CF, item-based CF, and content-based)
  python movie_recommender.py --evaluate
  
  # Evaluate specific method
  python movie_recommender.py --evaluate --method user
  python movie_recommender.py --evaluate --method item
  python movie_recommender.py --evaluate --method content
  
  # Run interactive mode with user-based CF (default)
  python movie_recommender.py --interactive
  
  # Run interactive mode with item-based CF
  python movie_recommender.py --interactive --method item
  
  # Clear all cache files
  python movie_recommender.py --clear-cache
  
  # Clear cache for specific method
  python movie_recommender.py --clear-cache --method user
        """
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode for new users'
    )
    
    parser.add_argument(
        '--evaluate', '-e',
        action='store_true',
        help='Evaluate the model using train/test split and display metrics'
    )
    
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['user', 'item', 'content'],
        default=None,
        help='Method: "user" for user-based CF, "item" for item-based CF, "content" for content-based. If omitted in evaluation mode, evaluates all methods.'
    )
    
    parser.add_argument(
        '--user-id',
        type=int,
        default=1,
        metavar='ID',
        help='User ID to get recommendations for in standard mode (default: 1)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        metavar='FLOAT',
        help='Proportion of data to use for testing in evaluation mode (default: 0.2)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=4.0,
        metavar='FLOAT',
        help='Rating threshold to consider a movie as "relevant" in evaluation (default: 4.0)'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached model files and exit'
    )
    
    args = parser.parse_args()
    
    # Handle cache clearing
    if args.clear_cache:
        print("=" * 60)
        print("Clearing Cache")
        print("=" * 60)
        clear_cache(method=args.method)
        print("=" * 60)
        sys.exit(0)
    
    # Check which mode to run
    if args.evaluate:
        # Evaluation mode
        if args.method:
            # Evaluate specific method
            main_evaluate(method=args.method, test_size=args.test_size, threshold=args.threshold)
        else:
            # Evaluate all methods: user-based CF, item-based CF, and content-based
            print("=" * 60)
            print("Comprehensive Model Evaluation - All Methods")
            print("=" * 60)
            print("\nEvaluating all recommendation methods...")
            print("This will evaluate: User-Based CF, Item-Based CF, and Content-Based")
            print("=" * 60)
            
            methods_to_evaluate = ['user', 'item', 'content']
            
            for idx, method in enumerate(methods_to_evaluate, 1):
                print(f"\n\n{'=' * 60}")
                print(f"[{idx}/{len(methods_to_evaluate)}] Evaluating {method.upper()}-based method...")
                print("=" * 60)
                main_evaluate(method=method, test_size=args.test_size, threshold=args.threshold)
                
                # Add spacing between methods (except after last one)
                if idx < len(methods_to_evaluate):
                    print("\n")
            
            print("\n" + "=" * 60)
            print("Evaluation Complete - All Methods")
            print("=" * 60)
    elif args.interactive:
        # Interactive mode
        method = args.method if args.method else 'user'
        main_interactive(method=method)
    else:
        # Standard mode
        main(user_id=args.user_id, method=args.method)

