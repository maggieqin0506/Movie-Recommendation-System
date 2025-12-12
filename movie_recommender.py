"""
Movie Recommendation System - Main Entry Point
This module provides backward compatibility and the main() function for standard mode.
"""

from collaborative_filtering import CollaborativeFilteringRecommender


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
        
        if len(methods_to_run) > 1 and cf_method != methods_to_run[-1]:
            print("\n")


if __name__ == "__main__":
    from cli import main_cli
    main_cli()
