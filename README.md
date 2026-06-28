
# 😺 MoodCat — Mood-Based Hybrid Movie Recommender

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Hybrid%20Recommender-brightgreen)
![TMDB API](https://img.shields.io/badge/TMDB-API-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

MoodCat is an AI-powered **hybrid movie recommendation system** that suggests movies based on a user’s **current mood**, watch history, content similarity, and popularity trends — all inside an interactive Streamlit dashboard.

Unlike traditional recommenders that rely only on past activity, MoodCat adds an **emotional intelligence layer** using sentiment analysis and mood-to-genre mapping to make recommendations more human-centric.

---

## 🚀 Features

- Mood-based movie recommendations *(Happy, Sad, Romantic, Wholesome, Dark, True Crime, Documentary, and more)*
- Text-based mood detection using **TextBlob**
- Hybrid recommender system *(Collaborative + Content + Mood + Popularity)*
- Shuffle / regenerate recommendations dynamically
- **Surprise Me** feature
- Save movies to watch later
- Friend recommendation mode *(based on another user ID)*
- Animated glassmorphism UI with gradient background
- Live movie posters fetched using **TMDB API**
- Fast performance with **Streamlit caching**

---

## 🧠 Hybrid Recommendation Architecture

MoodCat combines multiple recommendation strategies into a single hybrid score:

| Component | Technique | Weight |
|---|---|---|
| Collaborative Filtering | Truncated SVD on MovieLens ratings | **50%** |
| Content Similarity | TF-IDF on titles & genres | **20%** |
| Mood Relevance | Genre mapping based on mood | **20%** |
| Popularity | Rating frequency normalization | **10%** |

---

## 🛠️ ML / AI Concepts Used

- TF-IDF Vectorization
- Cosine Similarity
- Truncated SVD (Collaborative Filtering)
- Sentiment Analysis (TextBlob)
- Feature Scaling (MinMaxScaler)
- Hybrid Recommendation System Design

---

## 🗂️ Dataset Used

This project uses the **MovieLens 100K Dataset**.

- `u.item` — Movie metadata & genres  
- `u.data` — User ratings

Movie posters are dynamically fetched using the **TMDB API**.

---

## 🔐 Environment Variable (TMDB API)

Create a `.env` file in the root directory:

```
TMDB_API_KEY="your_key_here"
```

Or use Streamlit secrets:

`.streamlit/secrets.toml`

```
TMDB_API_KEY = "cb080ccb4e224a39cce6e2d61296edf7"
```

---

## ▶️ How to Run Locally

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit app

```
streamlit run app.py
```

---

## 📁 Project Structure

```
MoodCat/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── assets/
│   └── MoodCat.png
│
└── ml_dataset/
    ├── u.item
    └── u.data
```

---

## 💡 What Makes MoodCat Different?

Traditional systems say:

> “You watched X, so watch Y.”

MoodCat says:

> **“You feel X, so watch Y.”**

This adds an emotional and contextual layer to content discovery.

---

## 🌟 Future Improvements

- Web series recommendations
- Music recommendations
- Online deployment (Streamlit Cloud)
- Explainable recommendation scores
- User login & history tracking
- Analytics dashboard for trends

---

## 🧑‍💻 Author

**Dhyana Joshi**  
B.Tech — Computer Science and Design  
G. H. Patel College of Engineering and Technology
