"""
Movie Recommendation System using Collaborative Filtering
Supports both User-Based and Item-Based Collaborative Filtering
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Set
import warnings
import ast
import re
warnings.filterwarnings('ignore')


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering based movie recommendation system.
    Supports both user-based and item-based filtering approaches.
    """
    
    def __init__(self, ratings_file: str, movies_file: str, method: str = 'user'):
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
        """
        self.method = method
        self.ratings_file = ratings_file
        self.movies_file = movies_file
        self.ratings_df = None
        self.movies_df = None
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_ids = None
        self.movie_ids = None
        
    def load_data(self, sample_size: int = None):
        """
        Load ratings and movies data.
        
        Parameters:
        -----------
        sample_size : int, optional
            If provided, sample this many rows from ratings for faster processing
        """
        print("Loading ratings data...")
        if sample_size:
            # Sample data for faster processing on large datasets
            self.ratings_df = pd.read_csv(self.ratings_file, nrows=sample_size)
            print(f"Loaded {sample_size:,} ratings (sampled)")
        else:
            self.ratings_df = pd.read_csv(self.ratings_file)
            print(f"Loaded {len(self.ratings_df):,} ratings")
        
        print("Loading movies data...")
        self.movies_df = pd.read_csv(self.movies_file)
        print(f"Loaded {len(self.movies_df):,} movies")
        
        # Filter ratings to only include movies that exist in movies_df
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(self.movies_df['movieId'])]
        print(f"Filtered to {len(self.ratings_df):,} ratings after movie matching")
        
    def create_user_item_matrix(self):
        """Create a user-item rating matrix."""
        print("Creating user-item matrix...")
        
        # Create pivot table: users as rows, movies as columns
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        # Store user and movie IDs for reference
        self.user_ids = self.user_item_matrix.index.values
        self.movie_ids = self.user_item_matrix.columns.values
        
        print(f"Matrix shape: {self.user_item_matrix.shape[0]:,} users × {self.user_item_matrix.shape[1]:,} movies")
        
        # Convert to sparse matrix for efficiency
        self.user_item_matrix_sparse = csr_matrix(self.user_item_matrix.values)
        
    def compute_similarity(self, n_neighbors: int = 50):
        """
        Compute similarity matrix based on the chosen method.
        
        Parameters:
        -----------
        n_neighbors : int
            Number of nearest neighbors to consider (for efficiency)
        """
        print(f"Computing {self.method}-based similarity matrix...")
        
        if self.method == 'user':
            # User-based: compute similarity between users
            # Use cosine similarity on user vectors
            similarity = cosine_similarity(self.user_item_matrix_sparse)
            self.similarity_matrix = pd.DataFrame(
                similarity,
                index=self.user_ids,
                columns=self.user_ids
            )
            print("User similarity matrix computed")
            
        elif self.method == 'item':
            # Item-based: compute similarity between movies
            # Transpose the matrix: movies as rows, users as columns
            item_user_matrix = self.user_item_matrix.T
            item_user_matrix_sparse = csr_matrix(item_user_matrix.values)
            
            similarity = cosine_similarity(item_user_matrix_sparse)
            self.similarity_matrix = pd.DataFrame(
                similarity,
                index=self.movie_ids,
                columns=self.movie_ids
            )
            print("Item similarity matrix computed")
        else:
            raise ValueError("Method must be 'user' or 'item'")
    
    def get_user_ratings(self, user_id: int) -> pd.Series:
        """Get all ratings for a specific user."""
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found in the dataset")
        
        user_ratings = self.user_item_matrix.loc[user_id]
        return user_ratings[user_ratings > 0]  # Return only rated movies
    
    def predict_rating_user_based(self, user_id: int, movie_id: int, k: int = 50) -> float:
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
        user_ratings = self.user_item_matrix.loc[user_id]
        user_mean = user_ratings[user_ratings > 0].mean()
        
        # Get similar users
        user_similarities = self.similarity_matrix.loc[user_id].sort_values(ascending=False)
        user_similarities = user_similarities[user_similarities.index != user_id]  # Exclude self
        top_k_users = user_similarities.head(k)
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for similar_user_id, similarity in top_k_users.items():
            if similarity <= 0:
                continue
                
            similar_user_rating = self.user_item_matrix.loc[similar_user_id, movie_id]
            
            if similar_user_rating > 0:
                similar_user_mean = self.user_item_matrix.loc[similar_user_id]
                similar_user_mean = similar_user_mean[similar_user_mean > 0].mean()
                
                numerator += similarity * (similar_user_rating - similar_user_mean)
                denominator += abs(similarity)
        
        if denominator == 0:
            return user_mean
        
        predicted_rating = user_mean + (numerator / denominator)
        # Clamp rating between 0.5 and 5.0
        predicted_rating = max(0.5, min(5.0, predicted_rating))
        
        return predicted_rating
    
    def predict_rating_item_based(self, user_id: int, movie_id: int, k: int = 50) -> float:
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
        user_ratings = self.user_item_matrix.loc[user_id]
        user_rated_movies = user_ratings[user_ratings > 0]
        
        if len(user_rated_movies) == 0:
            return 0.0
        
        # Get similar movies
        if movie_id not in self.similarity_matrix.index:
            return 0.0
        
        movie_similarities = self.similarity_matrix.loc[movie_id].sort_values(ascending=False)
        movie_similarities = movie_similarities[movie_similarities.index != movie_id]  # Exclude self
        
        # Only consider movies the user has rated
        rated_movie_similarities = movie_similarities[movie_similarities.index.isin(user_rated_movies.index)]
        top_k_movies = rated_movie_similarities.head(k)
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for similar_movie_id, similarity in top_k_movies.items():
            if similarity <= 0:
                continue
            
            user_rating = user_rated_movies[similar_movie_id]
            numerator += similarity * user_rating
            denominator += abs(similarity)
        
        if denominator == 0:
            # Return mean of only rated movies (consistent with user-based CF)
            return user_rated_movies.mean()
        
        predicted_rating = numerator / denominator
        # Clamp rating between 0.5 and 5.0
        predicted_rating = max(0.5, min(5.0, predicted_rating))
        
        return predicted_rating
    
    def recommend_movies(self, user_id: int, n_recommendations: int = 10, k: int = 50) -> List[Tuple[int, str, float]]:
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
        
        # Get all movie IDs
        all_movie_ids = set(self.movie_ids)
        
        # Movies to predict (not yet rated)
        unrated_movies = all_movie_ids - rated_movie_ids
        
        print(f"Predicting ratings for {len(unrated_movies):,} unrated movies...")
        
        # Predict ratings for unrated movies
        predictions = []
        
        for movie_id in list(unrated_movies)[:1000]:  # Limit to 1000 for efficiency
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
        
        user_similarities = self.similarity_matrix.loc[user_id].sort_values(ascending=False)
        user_similarities = user_similarities[user_similarities.index != user_id]
        
        return [(uid, sim) for uid, sim in user_similarities.head(n).items()]
    
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
        
        movie_similarities = self.similarity_matrix.loc[movie_id].sort_values(ascending=False)
        movie_similarities = movie_similarities[movie_similarities.index != movie_id]
        
        similar_movies = []
        for similar_movie_id, similarity in movie_similarities.head(n).items():
            movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie_id]
            if len(movie_info) > 0:
                title = movie_info.iloc[0]['title']
                similar_movies.append((similar_movie_id, title, similarity))
        
        return similar_movies
    
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
        
        # Create a new row for this user
        new_user_row = pd.Series(0, index=self.movie_ids)
        
        # Add ratings
        for movie_id, rating in user_ratings.items():
            if movie_id in self.movie_ids:
                new_user_row[movie_id] = rating
        
        # Add to user-item matrix
        self.user_item_matrix.loc[new_user_id] = new_user_row
        
        # Update user_ids
        self.user_ids = self.user_item_matrix.index.values
        
        # Recompute similarity if needed (for user-based CF)
        if self.method == 'user':
            # Recompute similarity matrix to include new user
            self.user_item_matrix_sparse = csr_matrix(self.user_item_matrix.values)
            similarity = cosine_similarity(self.user_item_matrix_sparse)
            self.similarity_matrix = pd.DataFrame(
                similarity,
                index=self.user_ids,
                columns=self.user_ids
            )
        
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


def interactive_new_user_mode(recommender):
    """
    Interactive mode for new users to rate movies and get recommendations.
    
    Parameters:
    -----------
    recommender : CollaborativeFilteringRecommender
        Initialized recommender object
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
    print("Step 1: Select Your Favorite Genres")
    print("=" * 60)
    print("\nThis helps us show you movies you're more likely to know and enjoy!")
    
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
        print("Step 3: Rate Movies from Your Preferred Genres")
        print("=" * 60)
        
        if selected_genres:
            movies_to_rate = recommender.get_movies_by_genres(selected_genres, n=30)
            print(f"\nHere are popular movies from your selected genres ({', '.join(selected_genres)}):")
        else:
            movies_to_rate = [(m[0], m[1], m[2], '') for m in recommender.get_popular_movies(n=30)]
            print(f"\nHere are some popular movies. Please rate at least {min_ratings} of them:")
        
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
    
    # Get recommendations
    recommendations = recommender.recommend_movies(new_user_id, n_recommendations=15, k=50)
    
    if not recommendations:
        print("Sorry, we couldn't generate recommendations. Try rating more diverse movies.")
        return
    
    print(f"\n🎬 Your Personalized Movie Recommendations:")
    print("=" * 60)
    for i, (movie_id, title, pred_rating) in enumerate(recommendations, 1):
        print(f"{i:2d}. {title}")
        print(f"    Predicted Rating: {pred_rating:.2f}/5.0")
    
    print("\n" + "=" * 60)
    print("Thank you for using our recommendation system!")
    print("=" * 60)


def main(user_id: int = 1, sample_size: int = 100000, method: str = None):
    """
    Example usage of the Collaborative Filtering Recommender.
    
    Parameters:
    -----------
    user_id : int
        User ID to get recommendations for (default: 1)
    sample_size : int
        Number of ratings to sample (0 for full dataset)
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
        
        # Load data
        if sample_size > 0:
            recommender.load_data(sample_size=sample_size)
        else:
            recommender.load_data()
        
        # Create user-item matrix
        recommender.create_user_item_matrix()
        
        # Compute similarity
        recommender.compute_similarity()
        
        # Check if user exists
        if user_id not in recommender.user_ids:
            print(f"\n⚠ Error: User {user_id} not found in the dataset.")
            print(f"Available user IDs range from {recommender.user_ids.min()} to {recommender.user_ids.max()}")
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
        
        # Get recommendations
        recommendations = recommender.recommend_movies(user_id, n_recommendations=10)
        
        print(f"\nTop 10 Recommendations for User {user_id} ({method_name}):")
        print("-" * 60)
        for i, (movie_id, title, pred_rating) in enumerate(recommendations, 1):
            print(f"{i:2d}. {title} (Predicted Rating: {pred_rating:.2f})")
        
        # Show similar movies (only for item-based CF)
        if cf_method == 'item':
            print("\n" + "=" * 60)
            print("Finding Similar Movies")
            print("=" * 60)
            
            # Try to find a movie from user's top recommendations or their rated movies
            sample_movie_id = None
            
            # First, try to use the top recommendation
            if recommendations and len(recommendations) > 0:
                sample_movie_id = recommendations[0][0]  # Use top recommended movie
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


def main_interactive(method: str = 'user', sample_size: int = 100000):
    """
    Interactive mode for new users.
    
    Parameters:
    -----------
    method : str
        'user' for user-based CF or 'item' for item-based CF
    sample_size : int
        Number of ratings to sample (0 for full dataset)
    """
    method_name = "User-Based" if method == 'user' else "Item-Based"
    print("=" * 60)
    print(f"Movie Recommendation System - Interactive Mode ({method_name})")
    print("=" * 60)
    
    # Initialize recommender
    recommender = CollaborativeFilteringRecommender(
        ratings_file='ratings_full.csv',
        movies_file='movies_clean.csv',
        method=method
    )
    
    # Load data
    print("\nLoading data...")
    if sample_size > 0:
        recommender.load_data(sample_size=sample_size)
    else:
        recommender.load_data()  # Full dataset
    
    # Create user-item matrix
    recommender.create_user_item_matrix()
    
    # Compute similarity
    recommender.compute_similarity()
    
    # Run interactive mode
    interactive_new_user_mode(recommender)


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
  
  # Run interactive mode with user-based CF (default)
  python movie_recommender.py --interactive
  
  # Run interactive mode with item-based CF
  python movie_recommender.py --interactive --method item
  
  # Run with custom sample size and user ID
  python movie_recommender.py --user-id 100 --sample-size 50000
        """
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode for new users'
    )
    
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['user', 'item'],
        default=None,
        help='Collaborative filtering method: "user" for user-based, "item" for item-based, or omit to show both'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100000,
        metavar='N',
        help='Number of ratings to sample for faster processing (default: 100000, use 0 for full dataset)'
    )
    
    parser.add_argument(
        '--user-id',
        type=int,
        default=1,
        metavar='ID',
        help='User ID to get recommendations for in standard mode (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Check if user wants interactive mode
    if args.interactive:
        # For interactive mode, default to 'user' if not specified
        method = args.method if args.method else 'user'
        main_interactive(method=method, sample_size=args.sample_size)
    else:
        # In standard mode, if method is specified, use it; otherwise show both
        main(user_id=args.user_id, sample_size=args.sample_size, method=args.method)

