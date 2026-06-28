import pandas as pd
import ast

print("Loading files...")

movies = pd.read_csv("movies_metadata.csv", low_memory=False)
credits = pd.read_csv("credits.csv")
keywords = pd.read_csv("keywords.csv")
links = pd.read_csv("links.csv")

print("Cleaning movies...")

movies = movies[movies["id"].str.isnumeric()]
movies["id"] = movies["id"].astype(int)

links = links.dropna(subset=["tmdbId"])
links["tmdbId"] = links["tmdbId"].astype(int)

# Merge links to get tmdbId
movies = movies.merge(links, left_on="id", right_on="tmdbId")

# Helper to extract names from JSON columns
def extract_names(text):
    try:
        items = ast.literal_eval(text)
        return " ".join([i["name"] for i in items])
    except:
        return ""

print("Processing genres, keywords, cast...")

movies["genres"] = movies["genres"].apply(extract_names)
keywords["keywords"] = keywords["keywords"].apply(extract_names)

credits["cast"] = credits["cast"].apply(
    lambda x: " ".join([i["name"] for i in ast.literal_eval(x)[:5]])
)

# Merge all
movies = movies.merge(keywords, on="id")
movies = movies.merge(credits[["id", "cast"]], on="id")

print("Building content text...")

movies["content_text"] = (
    movies["title"].fillna("") + " " +
    movies["overview"].fillna("") + " " +
    movies["genres"].fillna("") + " " +
    movies["keywords"].fillna("") + " " +
    movies["cast"].fillna("")
)

final = movies[[
    "id", "title", "content_text", "tmdbId", "vote_average", "release_date"
]].copy()

final.columns = ["movieId", "title", "content_text", "tmdbId", "rating", "release_date"]

print("Saving cleaned dataset...")
final.to_csv("tmdb_clean.csv", index=False)

print("DONE ✅ tmdb_clean.csv created")
