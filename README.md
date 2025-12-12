# Movie Recommendation System

A comprehensive movie recommendation system that supports multiple recommendation approaches: user-based collaborative filtering, item-based collaborative filtering, and content-based filtering.

## Features

- **User-Based Collaborative Filtering**: Recommends movies based on similar users' preferences
- **Item-Based Collaborative Filtering**: Recommends movies based on similarity between movies
- **Content-Based Filtering**: Recommends movies based on movie content (genres, tags, embeddings)
- **Interactive Mode for New Users**: Rate movies and get personalized recommendations
  - Genre-based movie selection
  - Movie search functionality
  - Flexible rating system
- **Efficient Processing**: Uses sparse matrices for handling large datasets
- **Caching System**: Automatically caches models to skip retraining on subsequent runs
- **Comprehensive**: Uses full dataset for best recommendation quality

## Installation

### Using Virtual Environment (Recommended)

Create and activate a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Direct Installation

Alternatively, install the required dependencies directly:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-learn >= 0.24.0

## Dataset

The system uses three CSV files:
- `ratings_full.csv`: User ratings with columns: userId, movieId, rating, timestamp, year, title
- `movies_clean.csv`: Movie metadata with columns: movieId, title, genres, year, etc.
- `movies_full.csv`: Extended movie features (required for content-based filtering, includes genres and tags)

**Note:** By default, the system filters movies to only include those released after 2015 (`min_year=2015`). This can be adjusted when calling `load_data()` or `load_model()`. Only users who have rated movies after 2015 are included in the system.

## Usage

### Basic Usage

**Recommended approach (uses caching):**

```python
from movie_recommender import CollaborativeFilteringRecommender

# Initialize with user-based collaborative filtering
recommender = CollaborativeFilteringRecommender(
    ratings_file='ratings_full.csv',
    movies_file='movies_clean.csv',
    method='user'  # or 'item' for item-based CF, 'content' for content-based
)

# Load model (automatically loads data, creates matrices, and computes similarity)
# Uses cache if available to skip retraining
recommender.load_model(use_cache=True)

# Get recommendations for a user
recommendations = recommender.recommend_movies(user_id=1, n_recommendations=10)

for movie_id, title, pred_rating in recommendations:
    print(f"{title}: {pred_rating:.2f}")
```

**Manual approach (if you need more control):**

```python
# Load data (full dataset)
recommender.load_data()

# Create user-item matrix
recommender.create_user_item_matrix()

# Compute similarity matrix
recommender.compute_similarity()

# Get recommendations for a user
recommendations = recommender.recommend_movies(user_id=1, n_recommendations=10)
```

### Running the Example

#### Standard Mode (Existing Users)

Run the included example for existing users in the dataset:

```bash
# If using virtual environment, activate it first:
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate     # On Windows

# Run with default settings (user 1, shows both user-based and item-based):
python movie_recommender.py

# Get recommendations for a specific user:
python movie_recommender.py --user-id 42

# Show help for all options:
python movie_recommender.py --help
```

This will:
1. Load the ratings and movies data (using cache if available)
2. Create user-item matrices
3. Compute similarity matrices
4. Generate recommendations for the specified user (default: user 1)
5. Show recommendations from both user-based and item-based CF (if method not specified)
6. Show content-based recommendations
7. Show similar movies (for item-based CF)

**Note:** If the specified user ID doesn't exist, the system will show an error message with the available user ID range. Only users who have rated movies released after 2015 are included.

#### Interactive Mode (New Users)

For new users who want to get recommendations by rating movies:

```bash
# Activate virtual environment first:
source venv/bin/activate  # On macOS/Linux

# Run in interactive mode (user-based CF, default):
python movie_recommender.py --interactive

# Run in interactive mode with item-based CF:
python movie_recommender.py --interactive --method item

# Run in interactive mode with content-based filtering:
python movie_recommender.py --interactive --method content
```

This interactive mode will:
1. **Step 1**: Ask you to select your favorite genres (optional)
2. **Step 2**: Let you search for specific movies you know and rate them
3. **Step 3**: Show you movies from your preferred genres (or popular movies) to rate
4. Generate personalized recommendations based on your ratings

**Features:**
- Genre-based filtering: Select genres you like to see relevant movies
- Movie search: Search for movies you've watched to rate them
- Flexible rating: Rate at least 5 movies (0.5 to 5.0 scale)
- Skip option: Skip movies you haven't seen

### User-Based Collaborative Filtering

User-based CF finds users with similar preferences and recommends movies they liked:

```python
recommender = CollaborativeFilteringRecommender(
    ratings_file='ratings_full.csv',
    movies_file='movies_clean.csv',
    method='user'
)

# Get similar users
similar_users = recommender.get_similar_users(user_id=1, n=10)
```

### Adding New Users Programmatically

You can also add new users programmatically:

```python
# Create a dictionary of movie_id -> rating
new_user_ratings = {
    1: 5.0,   # Toy Story
    110: 4.5, # Braveheart
    260: 4.0  # Star Wars
}

# Add the new user
new_user_id = recommender.add_new_user_ratings(new_user_ratings)

# Get recommendations
recommendations = recommender.recommend_movies(new_user_id, n_recommendations=10)
```

### Item-Based Collaborative Filtering

Item-based CF finds movies similar to ones the user has rated:

```python
recommender = CollaborativeFilteringRecommender(
    ratings_file='ratings_full.csv',
    movies_file='movies_clean.csv',
    method='item'
)

# Get similar movies
similar_movies = recommender.get_similar_movies(movie_id=1, n=10)
```

### Content-Based Filtering

Content-based filtering recommends movies based on movie content (genres, tags, embeddings):

```python
recommender = CollaborativeFilteringRecommender(
    ratings_file='ratings_full.csv',
    movies_file='movies_clean.csv',
    method='user'  # Method doesn't matter for content-based, but required for init
)

# Load model
recommender.load_model()

# Get content-based recommendations
recommendations = recommender.recommend_movies_content_based(user_id=1, n_recommendations=10)
```

## How It Works

### User-Based Collaborative Filtering

1. **Create User-Item Matrix**: Build a matrix where rows are users and columns are movies
2. **Compute User Similarity**: Calculate cosine similarity between user rating vectors
3. **Predict Ratings**: For each unrated movie, find similar users who rated it and compute weighted average
4. **Recommend**: Return top N movies with highest predicted ratings

### Item-Based Collaborative Filtering

1. **Create User-Item Matrix**: Same as user-based
2. **Compute Item Similarity**: Calculate cosine similarity between movie rating vectors
3. **Predict Ratings**: For each unrated movie, find similar movies the user rated and compute weighted average
4. **Recommend**: Return top N movies with highest predicted ratings

## Model Evaluation

The system includes comprehensive evaluation metrics to assess recommendation quality:

### Available Metrics

1. **Rating Prediction Metrics**:
   - **RMSE (Root Mean Square Error)**: Measures prediction accuracy (lower is better)
   - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual ratings (lower is better)

2. **Ranking Metrics**:
   - **Precision@K**: Fraction of recommended items that are relevant (higher is better)
   - **Recall@K**: Fraction of relevant items that were recommended (higher is better)
   - **F1@K**: Harmonic mean of Precision and Recall (higher is better)
   - **NDCG@K**: Normalized Discounted Cumulative Gain - measures ranking quality with position discounting (higher is better)

### Running Evaluation

```bash
# Evaluate all methods (user-based CF, item-based CF, and content-based)
python movie_recommender.py --evaluate

# Evaluate specific method
python movie_recommender.py --evaluate --method user
python movie_recommender.py --evaluate --method item
python movie_recommender.py --evaluate --method content

# Evaluate with custom test size
python movie_recommender.py --evaluate --test-size 0.3
```

### Programmatic Evaluation

```python
from movie_recommender import CollaborativeFilteringRecommender

# Initialize and load model
recommender = CollaborativeFilteringRecommender(
    ratings_file='ratings_full.csv',
    movies_file='movies_clean.csv',
    method='user'
)
recommender.load_model()

# Split into train/test
train_ratings, test_ratings = recommender.train_test_split(test_size=0.2)

# Rebuild model on training data
recommender.ratings_df = train_ratings
recommender.create_user_item_matrix()
recommender.compute_similarity()

# Evaluate on test set
metrics = recommender.evaluate_model(test_ratings, k=10, top_k=10, threshold=4.0)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"Precision@10: {metrics['precision@k']:.4f}")
print(f"Recall@10: {metrics['recall@k']:.4f}")
```

## Performance Considerations

- The system always uses the full dataset for best recommendation quality
- The system uses sparse matrices for memory efficiency
- **Caching**: Models are automatically cached to `.cache/` directory to skip retraining on subsequent runs
- Similarity computation can be time-consuming for very large datasets
- Consider using approximate nearest neighbors for production systems
- Evaluation can be slow for large test sets - consider sampling test data
- By default, only movies released after 2015 are included to improve performance

## API Reference

### CollaborativeFilteringRecommender

#### Methods

**Model Loading:**
- `load_model(min_year=2015, use_cache=True)`: Load data and build model (recommended). Uses cache if available to skip retraining.
- `load_data(min_year=2015)`: Load ratings and movies data (full dataset)
- `create_user_item_matrix()`: Create user-item rating matrix
- `compute_similarity(n_neighbors=10, use_cache=True)`: Compute similarity matrix

**Recommendations:**
- `recommend_movies(user_id, n_recommendations=10, k=10)`: Get collaborative filtering movie recommendations
- `recommend_movies_content_based(user_id, n_recommendations=10)`: Get content-based movie recommendations
- `get_user_ratings(user_id)`: Get all ratings for a user
- `get_similar_users(user_id, n=10)`: Get users similar to a given user (user-based only)
- `get_similar_movies(movie_id, n=10)`: Get movies similar to a given movie (item-based only)
- `get_popular_movies(n=50)`: Get popular movies based on average ratings
- `get_movies_by_genres(genres, n=50)`: Get movies filtered by genres
- `search_movies(query, n=20)`: Search for movies by title
- `get_all_genres()`: Get list of all available genres

**Evaluation:**
- `train_test_split(test_size=0.2, random_state=42)`: Split ratings into train/test sets
- `evaluate_rating_prediction(test_ratings, k=10)`: Evaluate rating prediction (RMSE, MAE)
- `evaluate_ranking(test_ratings, k=10, top_k=10, threshold=4.0)`: Evaluate ranking quality (Precision@K, Recall@K, NDCG@K)
- `evaluate_model(test_ratings, k=10, top_k=10, threshold=4.0)`: Comprehensive model evaluation
- `evaluate_content_based(train_ratings, test_ratings, top_k=10, threshold=4.0)`: Evaluate content-based recommendations
- `calculate_diversity(recommendations)`: Calculate diversity of recommendations

**User Management:**
- `add_new_user_ratings(user_ratings)`: Add ratings for a new user and get their user ID

## Example Output

```
Top 10 Recommendations for User 1:
 1. The Shawshank Redemption (Predicted Rating: 4.85)
 2. The Godfather (Predicted Rating: 4.82)
 3. Pulp Fiction (Predicted Rating: 4.78)
 ...
```

## Command-Line Arguments

The script supports several command-line arguments:

- `--interactive` or `-i`: Run in interactive mode for new users
- `--evaluate` or `-e`: Evaluate the model using train/test split and display metrics. If no method specified, evaluates all three methods (user, item, content).
- `--method` or `-m`: Choose recommendation method (`user`, `item`, or `content`). In standard mode, omit to show both user-based and item-based CF. In interactive mode, defaults to `user` if not specified. In evaluation mode, omit to evaluate all methods.
- `--user-id ID`: User ID to get recommendations for in standard mode (default: 1)
- `--test-size FLOAT`: Proportion of data to use for testing in evaluation mode (default: 0.2)
- `--threshold FLOAT`: Rating threshold to consider a movie as "relevant" in evaluation (default: 4.0)
- `--clear-cache`: Clear all cached model files and exit
- `--plot`: Generate precision and recall vs k plots for all methods (k=1 to 10)
- `--save-plot PATH`: Path to save the precision/recall plot (e.g., "precision_recall.png")
- `--help` or `-h`: Show help message

### Examples

```bash
# Standard mode for user 1 (default) - shows both user-based and item-based
python movie_recommender.py

# Standard mode with only user-based CF
python movie_recommender.py --method user

# Standard mode with only item-based CF
python movie_recommender.py --method item

# Standard mode for a specific user with item-based CF
python movie_recommender.py --user-id 42 --method item

# Interactive mode with user-based CF (default)
python movie_recommender.py --interactive

# Interactive mode with item-based CF
python movie_recommender.py -i -m item

# Evaluate the model
python movie_recommender.py --evaluate

# Evaluate with item-based CF and custom test size
python movie_recommender.py --evaluate --method item --test-size 0.3

# Clear all cache files
python movie_recommender.py --clear-cache

# Clear cache for specific method
python movie_recommender.py --clear-cache --method user

# Generate precision/recall vs k plots for all methods
python movie_recommender.py --plot

# Generate and save precision/recall plot
python movie_recommender.py --plot --save-plot precision_recall.png
```

## Notes
- Ratings are clamped between 0.5 and 5.0
- Only movies present in both ratings and movies datasets are considered
- The system filters out movies the user has already rated
- By default, only movies released after 2015 are included (configurable via `min_year` parameter)
- User-based CF generally works better for new users with few ratings
- Item-based CF can be more stable but requires more user ratings
- Content-based filtering uses movie embeddings from `content-based.py` and requires `movies_full.csv`
- Models are automatically cached to `.cache/` directory for faster subsequent runs
- Initial Dataset Available: https://www.kaggle.com/datasets/aalichao/cmpe-279-datasets/settings
- Modeling Dataset Available: https://www.kaggle.com/datasets/aalichao/modelingdataset/settings
