# app.py
import os
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import warnings
import re
import time
from PIL import Image
#from textblob import TextBlob
#from dotenv import load_dotenv
#load_dotenv()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore
analyzer = SentimentIntensityAnalyzer()
from rapidfuzz import process
import nltk
@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("brown")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")

load_nltk()

warnings.filterwarnings("ignore")

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(page_title="🍿 MoodCat 🎥", layout="wide")

# Session state
if "saved" not in st.session_state:
    st.session_state.saved = []


# ------------
# Background
# ------------
st.markdown("""
<style>

/* Apply gradient to the REAL full page (not just app box) */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(
        -45deg,
        #00bca1,
        #00a3a4,
        #008ba0,
        #005a5b,
        #003840
    );
    background-size: 400% 400%;
    animation: gradientFlow 30s ease infinite;
    color: #ffffff;
}

/* Gradient animation */
@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* REMOVE the white gap Streamlit adds at top */
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* Make the top toolbar transparent (keeps Deploy button!) */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar glass effect */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.2);
}

/* Buttons */
.stButton>button {
    background-color: #1b9aaa;
    color: white;
    border-radius: 8px;
    border: none;
}

/* Text readability */
h1, h2, h3, h4, p, label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load MovieLens dataset
# ---------------------------
def load_data():
    base = "ml_dataset"  
    u_item = os.path.join(base, "u.item")
    u_data = os.path.join(base, "u.data")

    if os.path.exists(u_item) and os.path.exists(u_data):
        movies = pd.read_csv(
            u_item, sep="|", encoding="latin-1", header=None,
            names=[
                "movieId", "title", "release_date", "video_release_date", "IMDb_URL", "unknown",
                "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
                "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                "Sci-Fi", "Thriller", "War", "Western"
            ], low_memory=False
        )
        ratings = pd.read_csv(
            u_data, sep="\t", header=None,
            names=["userId", "movieId", "rating", "timestamp"]
        )
        return movies, ratings

    raise FileNotFoundError("Expected MovieLens files not found in ml_dataset/")


# ---------------------------
# Prepare movies
# ---------------------------
@st.cache_data(show_spinner=False)
def prepare_movies(movies_df):
    movies = movies_df.copy()
    if "Comedy" in movies.columns and "Drama" in movies.columns:
        genre_cols = movies.columns[5:]
        def content_text_row(r):
            genres = " ".join([g for g in genre_cols if r.get(g, 0) == 1])
            return f"{r['title']} {genres}".strip()
        movies["content_text"] = movies.apply(content_text_row, axis=1)
    else:
        movies["content_text"] = movies["title"].fillna("")
    movies["movieId"] = movies["movieId"].astype(int)
    return movies

# -------------------------------------------
# Feature Builders(TF-IDF and truncatedSVD)
# -------------------------------------------
@st.cache_data(show_spinner=False)
def build_tfidf(movies):
    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vect.fit_transform(movies["content_text"].fillna(""))
    return vect, tfidf_matrix

@st.cache_data(show_spinner=False)
def build_cf(ratings_df):
    user_item = ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    user_ids = list(user_item.index)
    movie_ids = list(user_item.columns)
    comp = min(50, min(user_item.shape) - 1) if min(user_item.shape) > 2 else 2
    svd = TruncatedSVD(n_components=comp, random_state=42)
    latent = svd.fit_transform(user_item.values)
    return user_item, user_ids, movie_ids, svd, latent


# -----------------
# Mood and Genres
# -----------------
MOOD_TO_GENRES = {
    "happy": ["Comedy", "Romance", "Adventure", "Musical", "Animation"],
    "sad": ["Drama", "Romance"],
    "relaxed": ["Children", "Animation", "Musical"],
    "excited": ["Action", "Thriller", "Sci-Fi", "Adventure"],
    "neutral": ["Drama", "Mystery"],
    "romantic": ["Romance"],
    "wholesome": ["Children", "Animation", "Comedy"],
    "sentimental": ["Drama", "Romance", "War"],
    "dark": ["Film-Noir", "Horror", "Thriller"],
    "thoughtful": ["Documentary", "Mystery", "Drama"],
    "truecrime": ["Crime", "Mystery", "Thriller"],
    "documentary": ["Documentary"],
}

MOOD_KEYWORDS = {
    "happy": ["happy", "joy", "glad", "excited", "yay", "😁", "😊", "😄", "woohoo", "cheerful", "hype", "lit"],
    "sad": ["sad", "sadd", "upset", "down", "blue", "😢", "😭", "depressed", "heartbroken", "meh","the world is ending"],
    "romantic": ["romantic", "love", "relationship", "💖", "💕", "😍", "crush", "bae", "smitten","giddy","swoon"],
    "wholesome": ["wholesome", "family", "feel good", "heartwarming", "🥰", "🤗", "cozy"],
    "sentimental": ["sentimental", "nostalgic", "memories", "💭", "🥹"],
    "dark": ["dark", "intense", "horror", "serious", "😈", "🔪", "thriller", "spooky","bored"],
    "excited": ["thrill", "action", "adventure", "exciting", "⚡", "🔥", "rush"],
    "relaxed": ["calm", "relax", "light", "😌", "chill", "peaceful", "serene"],
    "thoughtful": ["mystery", "deep", "thoughtful", "mind bending", "🤯", "pondering", "intriguing"],
    "truecrime": ["crime", "murder", "investigation", "detective", "true crime", "🕵️‍♂️", "suspense"],
    "documentary": ["documentary", "real story", "biography", "real events", "📚", "learning", "educational"],
    "neutral": []  # fallback
}

INTENT_MAP = {
    "burnout": ["done with everything", "tired of life", "exhausted", "over it", "i can't anymore"],
    "comfort": ["sad", "need comfort", "feeling low", "lonely", "bad day"],
    "chaotic_fun": ["fucked", "mess", "chaos", "wild", "no thoughts"],
    "romance_hot": ["sexy", "hot", "horny", "romantic", "love vibes"],
    "feel_good": ["happy", "good vibes", "chill", "relaxed"],
}

INTENT_TO_MOOD = {
    "burnout": "wholesome",
    "comfort": "wholesome",
    "chaotic_fun": "dark",
    "romance_hot": "romantic",
    "feel_good": "happy"
}



# ----------------------------------------------------------------------------------------------------
# Poster Fetching(from tmdb api integration); api key is in the  .toml file + imentive api integration
# ----------------------------------------------------------------------------------------------------
def get_tmdb_key():
    return st.secrets.get("TMDB_API_KEY", None)
def get_omdb_key():
    return st.secrets.get("OMDB_API_KEY", None)



def parse_title_year(title):
    year_match = re.search(r"\((\d{4})\)", title)
    year = year_match.group(1) if year_match else None

    title = re.sub(r"\(\d{4}\)", "", title).strip()
    title = title.replace(", The", "").replace(", A", "").replace(", An", "")

    return title.strip(), year

@st.cache_data(show_spinner=False)
def poster_for_title(title):
    """
    Fetch poster for a movie using TMDB first, then OMDb as fallback.
    Lazy loads, caches results in session_state, retries once if needed.
    """
    if "poster_cache" not in st.session_state:
        st.session_state.poster_cache = {}
    if title in st.session_state.poster_cache:
        return st.session_state.poster_cache[title]
    url = "https://via.placeholder.com/300x450?text=No+Image"
    tmdb_api_key = get_tmdb_key()
    clean_title, year = parse_title_year(title)
    if tmdb_api_key:
        for attempt in range(2): 
            try:
                params = {"api_key": tmdb_api_key, "query": clean_title}
                if year and attempt == 0:
                    params["year"] = year
                r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=5)
                if r.status_code == 200:
                    results = r.json().get("results", [])
                    if results and len(results) > 0:
                        for movie in results:
                            if movie.get("poster_path"):
                                poster_path = movie["poster_path"]
                                url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                                break
            except Exception as e:
                print(f"TMDB fetch failed for '{title}': {e}")
            time.sleep(0.3)
    if url.endswith("No+Image"):
        omdb_api_key = get_omdb_key()
        if omdb_api_key:
            try:
                params = {"apikey": omdb_api_key, "t": clean_title}
                if year:
                    params["y"] = year
                r = requests.get("http://www.omdbapi.com/", params=params, timeout=5)
                data = r.json()
                if data.get("Poster") and data["Poster"] != "N/A":
                    url = data["Poster"]
            except Exception as e:
                print(f"OMDb fetch failed for '{title}': {e}")
    st.session_state.poster_cache[title] = url
    return url

@st.cache_data(show_spinner=True)
def preload_posters(movies_df):
    movies_df["poster_url"] = movies_df["title"].apply(poster_for_title)
    return movies_df

# ---------------------------
# Hybrid Recommender
# ---------------------------
W_CF, W_CONTENT, W_POP = 0.5, 0.2, 0.1

def get_mood_candidate_ids(movies_df, mood):
    genres = MOOD_TO_GENRES.get(mood, [])
    if not genres:
        return movies_df["movieId"].values
    valid_genres = [g for g in genres if g in movies_df.columns]
    mask = movies_df[valid_genres].sum(axis=1) > 0 if valid_genres else [True]*len(movies_df)
    return movies_df.loc[mask, "movieId"].values

def predict_cf_scores_for_user(user_id, user_item, user_ids, movie_ids, svd_model, latent_matrix):
    if user_id not in user_ids:
        return {}
    uid_index = user_ids.index(user_id)
    preds = np.dot(latent_matrix[uid_index].reshape(1, -1), svd_model.components_).flatten()
    return dict(zip(movie_ids, preds))

def get_seed_index(seed_title, movies_df):
    if not seed_title:
        return None
    clean_title, _ = parse_title_year(seed_title)
    choices = movies_df["title"].tolist()
    match = process.extractOne(clean_title, choices)
    if match and match[1] > 80: 
        return movies_df[movies_df["title"] == match[0]].index[0]
    return None

def recommend_hybrid(user_id, mood, seed_title, n, movies_df, ratings_df, tfidf_matrix, user_item, user_ids, movie_ids, svd_model, latent_matrix):
    mood_candidate_ids = get_mood_candidate_ids(movies_df, mood)
    cf_scores = predict_cf_scores_for_user(user_id, user_item, user_ids, movie_ids, svd_model, latent_matrix)
    candidate_ids = list(mood_candidate_ids)
    if cf_scores:
        candidate_ids = sorted(candidate_ids, key=lambda x: cf_scores.get(x, 0), reverse=True)

    cf_s = pd.Series(cf_scores).reindex(candidate_ids).fillna(0)
    cf_s = pd.Series(MinMaxScaler().fit_transform(cf_s.values.reshape(-1, 1)).flatten(), index=candidate_ids)
    seed_row = get_seed_index(seed_title, movies_df)
    if seed_row is not None:
        sims = linear_kernel(tfidf_matrix[seed_row], tfidf_matrix).flatten()
        content_s = pd.Series({mid: sims[idx] for idx, mid in enumerate(movies_df["movieId"]) if mid in candidate_ids})
    else:
        content_s = pd.Series(0, index=candidate_ids)
    pop = ratings_df.groupby("movieId")["rating"].count()
    pop_s = pd.Series([pop.get(mid, 0) for mid in candidate_ids], index=candidate_ids)
    pop_s = pd.Series(MinMaxScaler().fit_transform(pop_s.values.reshape(-1, 1)).flatten(), index=candidate_ids)

    final = W_CF * cf_s + W_CONTENT * content_s + W_POP * pop_s

    res = pd.DataFrame({
        "movieId": candidate_ids,
        "title": [movies_df.loc[movies_df["movieId"]==mid, "title"].values[0] for mid in candidate_ids],
        "final_score": final
    }).sort_values("final_score", ascending=False)
    top_pool = res.head(600)
    res = top_pool.sample(n=min(n, len(top_pool)), weights=top_pool["final_score"], replace=False)

    return res

# ---------------------------
# Intent Detection
# ---------------------------
def detect_intent(text):
    text = text.lower()
    for intent, keywords in INTENT_MAP.items():
        if any(k in text for k in keywords):
            return intent
    return "feel_good"

# ---------------------------
# Fallback local mood detection
# ---------------------------
def preprocess_text(text):
    text = text.lower()
    emoji_map = {
        "😁": "happy", "😄": "happy", "😊": "happy",
        "😭": "sad", "😢": "sad",
        "💖": "love", "💕": "love", "😍": "love",
        "🥰": "wholesome", "🤗": "wholesome",
        "😌": "relaxed", "⚡": "excited", "🔥": "excited",
        "🤯": "thoughtful", "🕵️‍♂️": "truecrime"
    }
    for e, word in emoji_map.items():
        text = text.replace(e, f" {word} ")
    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_mood_locally(user_text):
    scores = analyzer.polarity_scores(user_text)
    compound = scores["compound"]
    if compound >= 0.6:
        return "happy"
    elif 0.2 <= compound < 0.6:
        return "wholesome"
    elif -0.2 < compound < 0.2:
        return "neutral"
    elif -0.6 < compound <= -0.2:
        return "sad"
    else:
        return "dark"



# ---------------------------
# Display Grid
# ---------------------------
movies_df_raw, ratings_df = load_data()
movies = prepare_movies(movies_df_raw)
vect, tfidf_matrix = build_tfidf(movies)
user_item, user_ids, movie_ids, svd_model, latent_matrix = build_cf(ratings_df)

def display_grid(recs, grid_id="main"):
    cols = st.columns(3, gap="large")
    for i, row in recs.reset_index(drop=True).iterrows():
        c = cols[i % 3]
        with c:
            poster_url = poster_for_title(row["title"])
            st.image(poster_url, use_container_width=True)
            st.markdown(f"**{row['title']}**")
            st.caption(f"Score: {row['final_score']:.3f}")
            button_key = f"save_{row['movieId']}_{grid_id}_{i}"
            if st.button("💾 Save", key=button_key):
                if row["movieId"] not in st.session_state.saved:
                    st.session_state.saved.append(row["movieId"])

# ---------------------------
# Main App
# ---------------------------
try:
    movies_df_raw, ratings_df = load_data()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

movies = prepare_movies(movies_df_raw)

vect, tfidf_matrix = build_tfidf(movies)
user_item, user_ids, movie_ids, svd_model, latent_matrix = build_cf(ratings_df)

img = Image.open("assets/MoodCat.png")
left, center, right = st.columns([1, 2, 1])
with center:
    st.image(img, use_container_width=300)

st.markdown(
    "<p style='text-align:center; color:#4b0082; font-size:18px; font-weight:bold;'>"
    "Your mood, your movie, your vibe ✨"
    "</p>",
    unsafe_allow_html=True
)
   

st.success(f"Loaded {len(movies)} movies and {len(ratings_df)} ratings")
#------------------------
# Sidebar Controls
#------------------------

with st.sidebar:
    st.header("⚙️ Controls")
    uid = st.number_input("User ID", min_value=1, max_value=max(ratings_df["userId"]), value=1)
    mood_choice = st.radio(
    "What’s the mood whispering to you? ✨ ",
    [
        "😊 Feel-Good",
        "🌧 In My Feels",
        "⚡ Adrenaline Rush",
        "😌 Unwind Mode",
        "💞 Cupid's Arrow",
        "🤗 Comfort Watch",
        "🕰 Memory Lane",
        "🌑 Edge of Seat",
        "🧠 Mind Workout",
        "🕵 Bond 007", 
        "📚 Learn Something",
        "🎲 Go With The Flow",
    ]
    )
    mood_map = {
        "😊 Feel-Good": "happy",
        "🌧 In My Feels": "sad",
        "⚡ Adrenaline Rush": "excited",
        "😌 Unwind Mode": "relaxed",
        "💞 Cupid's Arrow": "romantic",
        "🤗 Comfort Watch": "wholesome",
        "🕰 Memory Lane": "sentimental",
        "🌑 Edge of Seat": "dark",
        "🧠 Mind Workout": "thoughtful",
        "🕵 Bond 007": "truecrime",
        "📚 Learn Something": "documentary",
        "🎲 Go With The Flow": "neutral",
}
    mood = mood_map[mood_choice]

def recommend_by_seed_movie(seed_title, n, movies_df, tfidf_matrix, ratings_df, user_item=None, user_ids=None, movie_ids=None, svd_model=None, latent_matrix=None):
    if not seed_title:
        return pd.DataFrame(columns=["movieId", "title", "final_score"])
    clean_seed, _ = parse_title_year(seed_title)
    clean_seed = clean_seed.lower().strip()
    choices = [parse_title_year(t)[0].lower().strip() for t in movies_df["title"]]
    match = process.extractOne(clean_seed, choices)
    if not match or match[1] < 70:  
        st.warning(f"Could not find a good match for '{seed_title}'. Showing general recommendations.")
        return pd.DataFrame(columns=["movieId", "title", "final_score"])
    seed_idx = choices.index(match[0]) 
    candidate_ids = list(movies_df["movieId"].values)
    sims = linear_kernel(tfidf_matrix[seed_idx], tfidf_matrix).flatten()
    content_s = pd.Series(sims, index=movies_df["movieId"])
    cf_s = pd.Series(0, index=candidate_ids)
    if user_item is not None and user_ids is not None and movie_ids is not None \
       and svd_model is not None and latent_matrix is not None:
        pass  
    pop = ratings_df.groupby("movieId")["rating"].count()
    pop_s = pd.Series([pop.get(mid, 0) for mid in candidate_ids], index=candidate_ids)
    pop_s = pd.Series(MinMaxScaler().fit_transform(pop_s.values.reshape(-1,1)).flatten(), index=candidate_ids)
    W_CONTENT, W_POP, W_CF = 0.7, 0.2, 0.1
    final_score = W_CONTENT * content_s + W_POP * pop_s + W_CF * cf_s

    res = pd.DataFrame({
        "movieId": candidate_ids,
        "title": [movies_df.loc[movies_df["movieId"] == mid, "title"].values[0] for mid in candidate_ids],
        "final_score": final_score
    }).sort_values("final_score", ascending=False)
    top_pool = res.head(600)
    res = top_pool.sample(
        n=min(n, len(top_pool)),
        weights=top_pool["final_score"],
        replace=(len(top_pool) < n)  
    ).reset_index(drop=True)

    return res

# -----------------------------
# Sidebar: Seed Movie Feature
# -----------------------------
with st.sidebar:
    n_recs_seed = st.slider("How many similar movies?", 3, 30, 6, key="seed_slider")
    st.header("🎬 Your Movie Pick")
    seed_in = st.text_input("Enter a movie you like:")
    if st.sidebar.button("Get Similar Movies", key="seed_btn"):
        seed_title = seed_in.strip() if seed_in else None
        if seed_title:
            recs_seed = recommend_by_seed_movie(
                seed_title=seed_title,
                n=n_recs_seed,
                movies_df=movies,
                tfidf_matrix=tfidf_matrix,
                ratings_df=ratings_df,
                user_item=user_item,
                user_ids=user_ids,
                movie_ids=movie_ids,
                svd_model=svd_model,
                latent_matrix=latent_matrix,
            )
            st.subheader(f"🎬 Movies similar to '{seed_title}'")
            display_grid(recs_seed, grid_id="seed_movie")
        else:
            st.warning("Please enter a movie title first!")

# ---------------------------
# Text-based mood detection
# ---------------------------
with st.sidebar:
    st.markdown("### 📝 Describe the vibe you’re looking for")
    user_text = st.text_area("Your feelings here...")

    if st.button("Analyze Mood & Recommend"):
        if user_text.strip():
            preprocessed_text = preprocess_text(user_text)
            intent = detect_intent(preprocessed_text)
            text_mood = INTENT_TO_MOOD[intent]
            st.success(f"Detected mood: **{text_mood.capitalize()}**")
            recs_text = recommend_hybrid(
                user_id=None,
                mood=text_mood,
                seed_title=None,
                n=n_recs_seed,
                movies_df=movies,
                ratings_df=ratings_df,
                tfidf_matrix=tfidf_matrix,
                user_item=user_item,
                user_ids=user_ids,
                movie_ids=movie_ids,
                svd_model=svd_model,
                latent_matrix=latent_matrix,
            )

        st.markdown("---") 
        st.subheader("🎬 Recommendations Based on Your Text Mood")
        display_grid(recs_text, grid_id="text_mood")  
        with st.expander("💾 Saved Movies"):
            if st.session_state.saved:
                for mid in st.session_state.saved:
                    title = movies.loc[movies["movieId"] == mid, "title"].values
                    st.write("🍿", title[0] if len(title) > 0 else mid)
                if st.button("Clear saved", key="clear_saved_text_mood"): 
                    st.session_state.saved = []
                else:
                    st.caption("No saved movies yet.")
            else:
                st.warning("Please type something first!")


if "recs" not in st.session_state:
    st.session_state.recs = None
if st.button("🔎 Get Recommendations"):
    with st.spinner("Generating!!!!!"):
        st.session_state.recs = recommend_hybrid(
            uid, mood,
            seed_in.strip() if seed_in else None,
            n_recs_seed,
            movies, ratings_df, tfidf_matrix,
            user_item, user_ids, movie_ids,
            svd_model, latent_matrix
        )
if st.session_state.recs is not None:
    st.subheader("Top Recommendations ✨")
    col1, col2 = st.columns([4,1])
    with col2:
        if st.button("🔀 Shuffle"):
            st.session_state.recs = recommend_hybrid(
                uid, mood,
                seed_in.strip() if seed_in else None,
                n_recs_seed,
                movies, ratings_df, tfidf_matrix,
                user_item, user_ids, movie_ids,
                svd_model, latent_matrix
            )

    display_grid(st.session_state.recs)

if st.button("Surprise Me! 🎲"):
    pool = movies[movies["content_text"].str.contains("|".join(MOOD_TO_GENRES.get(mood, [])), na=False)] \
        if MOOD_TO_GENRES.get(mood) else movies

    surprise = pool.sample(1).iloc[0]
    st.success(f"Your surprise pick: **{surprise['title']}**")
    st.image(poster_for_title(surprise["title"]), width=200)
    st.markdown("### 💾 Saved Movies")
    if st.session_state.saved:
        for i, mid in enumerate(st.session_state.saved):
            title = movies.loc[movies["movieId"]==mid,"title"].values
            st.write("🍿", title[0] if len(title)>0 else mid)
            remove_key = f"remove_{mid}_{i}"
        if st.button("❌ Remove", key=remove_key):
            st.session_state.saved.remove(mid)
else:
    st.caption("No saved movies yet.")



fid = st.number_input("Friend's User ID", min_value=1, max_value=max(ratings_df["userId"]), value=1)
if st.button("Show Friend's Picks"):
    with st.spinner("Fetching..."):
        recs = recommend_hybrid(
            fid, mood, None, n_recs_seed,
            movies, ratings_df, tfidf_matrix,
            user_item, user_ids, movie_ids,
            svd_model, latent_matrix
        )
    display_grid(recs, grid_id="friend")
