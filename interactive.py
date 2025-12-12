"""
Interactive mode for new users to rate movies and get recommendations.
"""

import ast
from collaborative_filtering import CollaborativeFilteringRecommender


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

