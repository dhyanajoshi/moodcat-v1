import pandas as pd
import ast

# Load the REAL TMDB metadata
df = pd.read_csv("movies_metadata.csv", low_memory=False)

# Keep only what MoodCat needs
df = df[[
    "title",
    "overview",
    "genres",
    "poster_path",
    "release_date",
    "vote_average"
]]

# Drop junk rows
df = df.dropna(subset=["title", "overview", "poster_path"])

# Convert genres JSON to words
def parse_genres(g):
    try:
        lst = ast.literal_eval(g)
        return " ".join([d["name"] for d in lst])
    except:
        return ""

df["genres"] = df["genres"].apply(parse_genres)

# Build content text for TF-IDF
df["content_text"] = (
    df["title"].fillna("") + " " +
    df["overview"].fillna("") + " " +
    df["genres"].fillna("")
)

# Save
df.to_csv("tmdb_clean.csv", index=False)

print("✅ tmdb_clean.csv created correctly")
