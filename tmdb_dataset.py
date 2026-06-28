import requests
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
import time
import requests

def safe_get(url, params, retries=5):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(2 + i)
    return None

API_KEY = os.getenv("TMDB_API_KEY")

def fetch_movies(pages=40):
    all_movies = []

    for page in range(1, pages + 1):
        time.sleep(0.6)
        url = "https://api.themoviedb.org/3/discover/movie"
        params = {
            "api_key": API_KEY,
            "sort_by": "popularity.desc",
            "primary_release_date.gte": "2005-01-01",
            "page": page
        }

        data = safe_get(url, params)
        if not data:
            continue

        for m in data["results"]:
            all_movies.append({
                "id": m["id"],
                "title": m["title"],
                "overview": m["overview"],
                "popularity": m["popularity"],
                "vote": m["vote_average"],
                "type": "movie"
            })

    return pd.DataFrame(all_movies)


def fetch_tv(pages=40):
    all_tv = []

    for page in range(1, pages + 1):
        time.sleep(0.6)
        url = "https://api.themoviedb.org/3/discover/tv"
        params = {
            "api_key": API_KEY,
            "sort_by": "popularity.desc",
            "first_air_date.gte": "2005-01-01",
            "page": page
        }

        data = safe_get(url, params)
        if not data:
            continue


        for t in data["results"]:
            all_tv.append({
                "id": t["id"],
                "title": t["name"],
                "overview": t["overview"],
                "popularity": t["popularity"],
                "vote": t["vote_average"],
                "type": "tv"
            })

    return pd.DataFrame(all_tv)


print("Fetching Movies...")
movies = fetch_movies()

print("Fetching TV Shows...")
tv = fetch_tv()

df = pd.concat([movies, tv]).drop_duplicates(subset=["title"])

df.to_csv("tmdb_content.csv", index=False)

print("✅ Dataset created: tmdb_content.csv")
