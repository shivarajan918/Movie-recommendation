import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TMDB Movie Recommendation Engine",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 TMDB Movie Recommendation Engine")
st.caption("Content-based recommendation system powered by text vectorization and cosine similarity.")
st.markdown("---")


# ==========================================
# 2. CACHED DATA PIPELINE & PREPROCESSING
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    """Loads datasets, merges them, parses JSON strings, and forms the clean text tag column."""
    # Check if files exist to avoid silent crashing
    if not os.path.exists('tmdb_5000_movies.csv') or not os.path.exists('tmdb_5000_credits.csv'):
        return None

    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    # Merge datasets on title
    movies = movies.merge(credits, on='title')
    
    # Feature selection
    movies = movies[['movie_id', 'overview', 'keywords', 'genres', 'title', 'cast', 'crew']]
    movies.dropna(inplace=True)

    # Notebook parser helpers
    def convert(text):
        L = []
        for i in ast.literal_eval(text):
            L.append(i['name'])
        return L

    def convert3(text):
        L = []
        counter = 0
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    def fetch_director(text):
        L = []
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    def collapse(L):
        L1 = []
        for i in L:
            L1.append(i.replace(" ", ""))
        return L1

    # Parse dictionary strings into clean token lists
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)

    # Remove spaces within structural entities
    movies['cast'] = movies['cast'].apply(collapse)
    movies['crew'] = movies['crew'].apply(collapse)
    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)

    # Convert descriptions into token arrays
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Aggregate columns into master string arrays
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    # Construct production dataframe
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    return new_df


# ==========================================
# 3. CACHED ML EMBEDDINGS & SIMILARITY
# ==========================================
@st.cache_resource
def compute_similarity_matrix(dataframe):
    """Calculates bag-of-words text matrix and maps pairwise document cosine similarities."""
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(dataframe['tags']).toarray()
    similarity = cosine_similarity(vector)
    return similarity


# ==========================================
# 4. INITIALIZATION & DATA SANITY CONTROLS
# ==========================================
new_df = load_and_preprocess_data()

if new_df is None:
    st.error("⚠️ Data files not found!")
    st.info("Please make sure `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are uploaded to your working GitHub repository folder root.")
    st.stop()

# Build calculation matrix
similarity = compute_similarity_matrix(new_df)


# ==========================================
# 5. STREAMLIT INTERACTIVE GRAPHICAL FRONTEND
# ==========================================
# Populate structured alphabetical select options for nice UX
movie_options = sorted(new_df['title'].unique())
selected_movie = st.selectbox("Select a movie to get similar recommendations:", movie_options)

if st.button("Generate Recommendations", type="primary"):
    with st.spinner("Processing token space vectors..."):
        try:
            # Query index position matching movie title
            movie_index = new_df[new_df['title'] == selected_movie].index[0]
            distances = similarity[movie_index]
            
            # Sort slice tracking matrix coordinates
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
            
            st.success(f"Top matches calculated for '{selected_movie}':")
            st.markdown("---")
            
            # Display recommendations cleanly
            for rank, i in enumerate(movies_list, start=1):
                movie_title = new_df.iloc[i[0]].title
                match_percentage = round(float(i[1]) * 100, 1)
                
                # Render using native stream container boxes
                with st.container():
                    st.subheader(f"{rank}. 🍿 {movie_title}")
                    st.caption(f"Vector Match Confidence: `{match_percentage}%`")
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"Could not execute query index cleanly. Details: {e}")