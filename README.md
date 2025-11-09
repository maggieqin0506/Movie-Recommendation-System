# Movie Recommendation System

A collaborative filtering-based movie recommendation system that supports both user-based and item-based filtering approaches.

## Features

- **User-Based Collaborative Filtering**: Recommends movies based on similar users' preferences
- **Item-Based Collaborative Filtering**: Recommends movies based on similarity between movies
- **Interactive Mode for New Users**: Rate movies and get personalized recommendations
  - Genre-based movie selection
  - Movie search functionality
  - Flexible rating system
- **Efficient Processing**: Uses sparse matrices for handling large datasets
- **Flexible**: Can work with sampled or full datasets

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
- `movies_full.csv`: Extended movie features (optional, not used in current implementation)

## Usage

### Basic Usage

```python
from movie_recommender import CollaborativeFilteringRecommender

# Initialize with user-based collaborative filtering
recommender = CollaborativeFilteringRecommender(
    ratings_file='ratings_full.csv',
    movies_file='movies_clean.csv',
    method='user'  # or 'item' for item-based CF
)

# Load data (sample for faster processing)
recommender.load_data(sample_size=100000)

# Create user-item matrix
recommender.create_user_item_matrix()

# Compute similarity matrix
recommender.compute_similarity()

# Get recommendations for a user
recommendations = recommender.recommend_movies(user_id=1, n_recommendations=10)

for movie_id, title, pred_rating in recommendations:
    print(f"{title}: {predicted_rating:.2f}")
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
1. Load the ratings and movies data
2. Create user-item matrices
3. Compute similarity matrices
4. Generate recommendations for the specified user (default: user 1)
5. Show similar movies

**Note:** If the specified user ID doesn't exist, the system will show an error message with the available user ID range.

#### Interactive Mode (New Users)

For new users who want to get recommendations by rating movies:

```bash
# Activate virtual environment first:
source venv/bin/activate  # On macOS/Linux

# Run in interactive mode (user-based CF, default):
python movie_recommender.py --interactive

# Run in interactive mode with item-based CF:
python movie_recommender.py --interactive --method item

# Run with custom sample size:
python movie_recommender.py --interactive --sample-size 50000

# Run with full dataset (no sampling):
python movie_recommender.py --interactive --sample-size 0
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

## Performance Considerations

- For large datasets (millions of ratings), use `sample_size` parameter to sample data
- The system uses sparse matrices for memory efficiency
- Similarity computation can be time-consuming for very large datasets
- Consider using approximate nearest neighbors for production systems

## API Reference

### CollaborativeFilteringRecommender

#### Methods

- `load_data(sample_size=None)`: Load ratings and movies data
- `create_user_item_matrix()`: Create user-item rating matrix
- `compute_similarity(n_neighbors=50)`: Compute similarity matrix
- `recommend_movies(user_id, n_recommendations=10, k=50)`: Get movie recommendations
- `get_user_ratings(user_id)`: Get all ratings for a user
- `get_similar_users(user_id, n=10)`: Get users similar to a given user (user-based only)
- `get_similar_movies(movie_id, n=10)`: Get movies similar to a given movie (item-based only)

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
- `--method` or `-m`: Choose collaborative filtering method (`user` or `item`). In standard mode, omit to show both methods. In interactive mode, defaults to `user` if not specified.
- `--user-id ID`: User ID to get recommendations for in standard mode (default: 1)
- `--sample-size N`: Number of ratings to sample (default: 100000, use 0 for full dataset)
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

# Interactive mode with full dataset
python movie_recommender.py --interactive --sample-size 0

# Standard mode with custom user and sample size
python movie_recommender.py --user-id 100 --sample-size 50000
```

## Notes

- Ratings are clamped between 0.5 and 5.0
- Only movies present in both ratings and movies datasets are considered
- The system filters out movies the user has already rated
- User-based CF generally works better for new users with few ratings
- Item-based CF can be more stable but requires more user ratings
