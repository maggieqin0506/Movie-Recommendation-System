"""
Command-line interface for the movie recommendation system.
"""

import sys
import argparse
from collaborative_filtering import CollaborativeFilteringRecommender
from movie_recommender import main
from interactive import main_interactive
from evaluation_runners import main_evaluate, evaluate_and_plot_precision_recall_vs_k
from cache_utils import clear_cache


def parse_args():
    """Parse command-line arguments."""
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
  
  # Generate precision/recall vs k plots for all methods
  python movie_recommender.py --plot
  
  # Generate and save precision/recall plot
  python movie_recommender.py --plot --save-plot precision_recall.png
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
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate precision and recall vs k plots for all methods (k=1 to 10)'
    )
    
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to save the precision/recall plot (e.g., "precision_recall.png")'
    )
    
    return parser.parse_args()


def main_cli():
    """Main CLI entry point."""
    args = parse_args()
    
    # Handle cache clearing
    if args.clear_cache:
        print("=" * 60)
        print("Clearing Cache")
        print("=" * 60)
        clear_cache(method=args.method)
        print("=" * 60)
        sys.exit(0)
    
    # Check which mode to run
    if args.plot:
        # Plot precision/recall vs k for all methods
        evaluate_and_plot_precision_recall_vs_k(
            test_size=args.test_size,
            threshold=args.threshold,
            max_users=100,
            save_path=args.save_plot,
            show_plot=True
        )
    elif args.evaluate:
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


if __name__ == "__main__":
    main_cli()

