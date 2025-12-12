"""
Evaluation runner functions for model evaluation.
"""

from collaborative_filtering import CollaborativeFilteringRecommender
from plotting import plot_precision_recall_vs_k


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


def evaluate_and_plot_precision_recall_vs_k(test_size: float = 0.2, threshold: float = 4.0,
                                            max_users: int = 100, save_path: str = None,
                                            show_plot: bool = True):
    """
    Evaluate precision and recall vs k for all recommendation methods and create plots.
    
    Parameters:
    -----------
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    threshold : float
        Rating threshold to consider a movie as "relevant" (default: 4.0)
    max_users : int
        Maximum number of users to evaluate (default: 100)
    save_path : str, optional
        Path to save the plot (default: None, don't save)
    show_plot : bool
        Whether to display the plot (default: True)
    """
    print("=" * 60)
    print("Precision and Recall vs Number of Recommendations (k)")
    print("Evaluating all methods: User-Based CF, Item-Based CF, Content-Based")
    print("=" * 60)
    
    # Initialize recommenders
    user_recommender = CollaborativeFilteringRecommender(
        ratings_file='ratings_full.csv',
        movies_file='movies_clean.csv',
        method='user'
    )
    
    item_recommender = CollaborativeFilteringRecommender(
        ratings_file='ratings_full.csv',
        movies_file='movies_clean.csv',
        method='item'
    )
    
    content_recommender = CollaborativeFilteringRecommender(
        ratings_file='ratings_full.csv',
        movies_file='movies_clean.csv',
        method='user'  # Method doesn't matter for content-based
    )
    
    # Load data for all recommenders
    print("\nLoading data...")
    user_recommender.load_data()
    item_recommender.load_data()
    content_recommender.load_data()
    
    # Split into train and test (use same split for all methods)
    print(f"\nSplitting data into train/test sets (test_size={test_size})...")
    train_ratings, test_ratings = user_recommender.train_test_split(test_size=test_size)
    
    print(f"Training set: {len(train_ratings):,} ratings")
    print(f"Test set: {len(test_ratings):,} ratings")
    
    # Build models on training data
    print("\nBuilding models on training data...")
    
    # User-based CF
    print("  - Building user-based CF model...")
    user_recommender.ratings_df = train_ratings
    user_recommender.create_user_item_matrix()
    user_recommender.compute_similarity()
    
    # Item-based CF
    print("  - Building item-based CF model...")
    item_recommender.ratings_df = train_ratings
    item_recommender.create_user_item_matrix()
    item_recommender.compute_similarity()
    
    # Content-based
    print("  - Building content-based model...")
    content_recommender.ratings_df = train_ratings
    content_recommender.create_user_item_matrix()
    try:
        content_recommender._load_content_based_system()
    except Exception as e:
        print(f"  Error loading content-based system: {e}")
        print("  Skipping content-based evaluation.")
        content_recommender = None
    
    # Evaluate all methods
    results_dict = {}
    
    # User-based CF
    print("\n" + "=" * 60)
    print("Evaluating User-Based Collaborative Filtering...")
    print("=" * 60)
    user_results = user_recommender.evaluate_precision_recall_vs_k(
        test_ratings=test_ratings,
        k=10,
        k_values=list(range(1, 11)),
        threshold=threshold,
        max_users=max_users
    )
    results_dict['User-Based CF'] = user_results
    
    # Item-based CF
    print("\n" + "=" * 60)
    print("Evaluating Item-Based Collaborative Filtering...")
    print("=" * 60)
    item_results = item_recommender.evaluate_precision_recall_vs_k(
        test_ratings=test_ratings,
        k=10,
        k_values=list(range(1, 11)),
        threshold=threshold,
        max_users=max_users
    )
    results_dict['Item-Based CF'] = item_results
    
    # Content-based
    if content_recommender is not None:
        print("\n" + "=" * 60)
        print("Evaluating Content-Based Filtering...")
        print("=" * 60)
        content_results = content_recommender.evaluate_content_based_precision_recall_vs_k(
            train_ratings=train_ratings,
            test_ratings=test_ratings,
            k_values=list(range(1, 11)),
            threshold=threshold,
            max_users=max_users
        )
        results_dict['Content-Based'] = content_results
    
    # Create and display plot
    print("\n" + "=" * 60)
    print("Generating precision and recall vs k plots...")
    print("=" * 60)
    plot_precision_recall_vs_k(results_dict, save_path=save_path, show_plot=show_plot)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Summary: Precision and Recall by k")
    print("=" * 60)
    print(f"\n{'k':<5} {'User-Based CF':<20} {'Item-Based CF':<20} {'Content-Based':<20}")
    print(f"{'':5} {'Prec':<10} {'Rec':<10} {'Prec':<10} {'Rec':<10} {'Prec':<10} {'Rec':<10}")
    print("-" * 75)
    
    for k in range(1, 11):
        user_prec = results_dict['User-Based CF'][k]['precision']
        user_rec = results_dict['User-Based CF'][k]['recall']
        item_prec = results_dict['Item-Based CF'][k]['precision']
        item_rec = results_dict['Item-Based CF'][k]['recall']
        
        line = f"{k:<5} {user_prec:<10.4f} {user_rec:<10.4f} {item_prec:<10.4f} {item_rec:<10.4f}"
        
        if 'Content-Based' in results_dict:
            cb_prec = results_dict['Content-Based'][k]['precision']
            cb_rec = results_dict['Content-Based'][k]['recall']
            line += f" {cb_prec:<10.4f} {cb_rec:<10.4f}"
        
        print(line)
    
    print("=" * 60)

