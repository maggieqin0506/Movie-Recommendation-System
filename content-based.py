import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

def user_recommendation_ann(df, X_emb, nn, liked_movie_ids, k=10, candidates=500, movie_ratings=None):
    """
    Generate content-based recommendations using ANN.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Movies dataframe with movieId and clean_title columns
    X_emb : np.ndarray
        Movie embeddings array
    nn : NearestNeighbors
        Fitted NearestNeighbors model
    liked_movie_ids : List[int]
        List of movie IDs the user has liked/rated
    k : int
        Number of recommendations to return
    candidates : int
        Number of candidate movies to consider
    movie_ratings : dict, optional
        Dictionary mapping movie_id to rating. If provided, uses weighted average based on ratings.
        If None, treats all movies equally (simple average).
        
    Returns:
    --------
    List[Tuple[int, str, float]]
        List of (movie_id, title, similarity_score) tuples
    """
    id_to_idx = pd.Series(df.index.values, index=df["movieId"]).to_dict()
    liked_idx = [id_to_idx[m] for m in liked_movie_ids if m in id_to_idx]
    if not liked_idx:
        return []

    # Use weighted average if ratings are provided, otherwise simple average
    if movie_ratings is not None:
        # Weight embeddings by ratings (normalize ratings to 0-1 scale for weighting)
        weights = np.array([movie_ratings.get(mid, 3.0) for mid in liked_movie_ids])
        # Normalize weights: convert 0.5-5.0 rating scale to 0-1 weight scale
        weights = (weights - 0.5) / 4.5  # Maps 0.5->0, 5.0->1
        weights = weights / (weights.sum() + 1e-12)  # Normalize to sum to 1
        
        # Weighted average of embeddings
        user_vec = np.average(X_emb[liked_idx], axis=0, weights=weights)
    else:
        # Simple average (original behavior)
        user_vec = X_emb[liked_idx].mean(axis=0)
    
    user_vec /= (np.linalg.norm(user_vec) + 1e-12)

    # ANN retrieval by cosine distance
    dist, idx = nn.kneighbors(user_vec.reshape(1, -1), n_neighbors=min(candidates, X_emb.shape[0]))
    idx = idx[0]; dist = dist[0]
    sims = 1.0 - dist  # cosine similarity = 1 - cosine distance

    # drop liked items
    liked_set = set(liked_idx)
    keep = [(i, s) for i, s in zip(idx, sims) if i not in liked_set]

    # top-k
    keep = sorted(keep, key=lambda x: x[1], reverse=True)[:k]
    return [(int(df.iloc[i]["movieId"]), str(df.iloc[i]["clean_title"]), float(s)) for i, s in keep]


def generate_movie_embeddings(movies_full_file='movies_full.csv', embeddings_file='movie_embeddings.npy'):
    """
    Generate movie embeddings from genres and tags.
    
    Parameters:
    -----------
    movies_full_file : str
        Path to movies_full.csv file
    embeddings_file : str
        Path where to save the embeddings file
        
    Returns:
    --------
    np.ndarray
        Movie embeddings array
    """
    print("Generating movie embeddings from genres and tags...")
    
    # Read movies_full.csv
    df = pd.read_csv(movies_full_file)
    print(f"  Loaded {len(df):,} movies")
    
    # Extract genre and tag columns
    genre_cols = df.columns[5:30]
    tag_cols = df.columns[30:1158]
    
    print(f"  Using {len(genre_cols)} genre columns and {len(tag_cols)} tag columns")
    
    # Convert to vectors, set values < 0.2 to 0
    G = df[genre_cols].astype(float).values
    T = df[tag_cols].astype(float).values
    
    T_dn = T.copy()
    T_dn[T_dn < 0.2] = 0.0
    
    # Apply IDF weighting: make common tags less important, rare tags more important
    print("  Applying IDF weighting to tags...")
    eps = 1e-6
    num_movies = T_dn.shape[0]
    tag_present = (T_dn > 0).sum(axis=0)
    idf = np.log((num_movies + 1) / (tag_present + 1) + eps)
    T_w = T_dn * idf
    
    # Combine and normalize
    print("  Combining features and normalizing...")
    X_raw = np.hstack([G, T_w])  # shape: (n_movies, 25 + 1000+)
    X = normalize(X_raw, norm="l2", axis=1)
    
    # Dimensionality reduction using SVD
    print("  Applying SVD for dimensionality reduction (this may take a moment)...")
    svd = TruncatedSVD(n_components=200, random_state=42)
    X_emb = svd.fit_transform(X)  # shape: (n_movies, 200)
    X_emb = normalize(X_emb, norm="l2", axis=1)  # L2 normalize so dot product == cosine similarity
    
    # Save embeddings
    print(f"  Saving embeddings to {embeddings_file}...")
    np.save(embeddings_file, X_emb)
    print(f"  ✓ Embeddings generated and saved successfully!")
    
    return X_emb


def load_content_based_system(movies_full_file='movies_full.csv', embeddings_file='movie_embeddings.npy', 
                              auto_generate=True):
    """
    Load content-based recommendation system components.
    Automatically generates embeddings if they don't exist.
    
    Parameters:
    -----------
    movies_full_file : str
        Path to movies_full.csv file
    embeddings_file : str
        Path to movie_embeddings.npy file
    auto_generate : bool
        If True, automatically generate embeddings if file doesn't exist (default: True)
        
    Returns:
    --------
    tuple
        (df, X_emb, nn) - movies dataframe, embeddings, and NearestNeighbors model
    """
    import os
    
    # Read movies_full.csv
    df = pd.read_csv(movies_full_file)
    
    # Check if embeddings exist, generate if not
    if not os.path.exists(embeddings_file):
        if auto_generate:
            print(f"Embeddings file not found. Generating embeddings...")
            X_emb = generate_movie_embeddings(movies_full_file, embeddings_file)
        else:
            raise FileNotFoundError(
                f"{embeddings_file} not found. Run content-based.py first to generate embeddings, "
                f"or set auto_generate=True to generate automatically."
            )
    else:
        # Load existing embeddings
        print(f"Loading movie embeddings from {embeddings_file}...")
        X_emb = np.load(embeddings_file)
    
    # Initialize NearestNeighbors model
    print("Initializing NearestNeighbors model...")
    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(X_emb)
    
    return df, X_emb, nn


# For direct execution (when running this file as script)
if __name__ == "__main__":
    """
    Standalone script to generate embeddings and test content-based recommendations.
    Normally, embeddings are generated automatically when needed by movie_recommender.py.
    """
    import os
    
    print("=" * 60)
    print("Content-Based Movie Recommendation System")
    print("=" * 60)
    
    # Generate embeddings
    if not os.path.exists("movie_embeddings.npy"):
        print("\nGenerating movie embeddings...")
        X_emb = generate_movie_embeddings()
    else:
        print("\nLoading existing embeddings...")
        X_emb = np.load("movie_embeddings.npy")
    
    # Load system and test
    print("\n" + "=" * 60)
    print("Testing Content-Based Recommendations")
    print("=" * 60)
    
    df, X_emb, nn = load_content_based_system(auto_generate=False)
    
    # Test with sample movie IDs
    test_movie_ids = [1, 296, 318]
    print(f"\nTesting with liked movies: {test_movie_ids}")
    recs = user_recommendation_ann(df, X_emb, nn, liked_movie_ids=test_movie_ids, k=10)
    
    print(f"\nTop 10 Recommendations:")
    print("-" * 60)
    for mid, title, sim in recs:
        print(f"{title:40s} (id={mid})  sim={sim:.3f}")
