import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# View full columns, run once to see all columns in movies_full.csv if needed
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# Read movies_full.csv
df2 = pd.read_csv("movies_full.csv")
print("Shape:", df2.shape)
df2.head()

#把genre和tags单独列出来
genre_cols = df2.columns[5:30]
tag_cols   = df2.columns[30:1158]
id_cols = ["movieId", "clean_title"]

# 弄成一个vector，把数值小于0.2的降为0
G = df2[genre_cols].astype(float).values
T = df2[tag_cols].astype(float).values

T_dn = T.copy()
T_dn[T_dn < 0.2] = 0.0

# 让常见tag的重量t变小 让稀有tag的重量变大
eps = 1e-6
num_movies = T_dn.shape[0]
tag_present = (T_dn > 0).sum(axis=0)
idf = np.log((num_movies + 1) / (tag_present + 1) + eps)  # shape: (num_tags,)
T_w = T_dn * idf

# 合并然后norm
X_raw = np.hstack([G, T_w])          # shape: (n_movies, 25 + 1000+)
X = normalize(X_raw, norm="l2", axis=1)

svd = TruncatedSVD(n_components=200, random_state=42)
X_emb = svd.fit_transform(X)                   # shape: (n_movies, 200)
X_emb = normalize(X_emb, norm="l2", axis=1)    # L2 so dot == cosine
np.save("movie_embeddings.npy", X_emb)         # 保存embedding之后推荐的时候需要


# 利用ANN进行推荐
nn = NearestNeighbors(metric="cosine", algorithm="auto")
nn.fit(X_emb)

def user_recommendation_ann(df, X_emb, liked_movie_ids, k=10, candidates=500):
    id_to_idx = pd.Series(df.index.values, index=df["movieId"]).to_dict()
    liked_idx = [id_to_idx[m] for m in liked_movie_ids if m in id_to_idx]
    if not liked_idx:
        return []

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

# IF NEEDED, 根据title获取movie_id
def get_movie_ids_from_titles(df, titles):
    if isinstance(titles, str):
        titles = [titles]
    ids = df.loc[df["title"].isin(titles), "movie_id"].tolist()
    return ids

# 推荐结果
X_emb = np.load("movie_embeddings.npy")
recs = user_recommendation_ann(df2, X_emb, liked_movie_ids=[1, 296, 318], k=10)

for mid, title, sim in recs:
    print(f"{title:40s} (id={mid})  sim={sim:.3f}")
