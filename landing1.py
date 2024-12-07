from flask import Flask, request, render_template, jsonify
import psycopg2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from difflib import get_close_matches

# Initialize Flask app
app = Flask(__name__)

# Database connection configuration
db_config = {
    'host': 'localhost',
    'database': 'movies_series_episodes',
    'user': 'postgres',
    'password': 'postgres'
}

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables for FAISS index and data
faiss_index = None
all_texts = []
movies_data = []  # To store all movie records
series_data = []  # To store all series records
media_names = []  # To store just names of movies and series for spelling corrections

# Connect to PostgreSQL
def get_db_connection():
    return psycopg2.connect(**db_config)

# Function to get data and build the FAISS index
def build_faiss_index():
    global faiss_index, all_texts, movies_data, series_data, media_names

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch movie data
    cursor.execute("SELECT name, director, genre FROM movies")
    movies_data = cursor.fetchall()

    # Fetch series data
    cursor.execute("SELECT name, season_number, episode_number FROM web_series")
    series_data = cursor.fetchall()

    cursor.close()
    conn.close()

    # Prepare data for embedding
    all_texts = []
    media_names = []

    # Combine movies and series data into text for embedding
    for movie in movies_data:
        all_texts.append(f"{movie[0]} {movie[1]} {movie[2]}".lower().strip())
        media_names.append(movie[0].lower().strip())

    for series in series_data:
        all_texts.append(
            f"{series[0]} season {series[1]} episode {series[2]}".lower().strip()
        )
        media_names.append(series[0].lower().strip())

    # Generate embeddings
    embeddings = model.encode(all_texts)

    # Create FAISS index
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings))

# Function to search for a movie or series by name
def search_media_by_name(query):
    global faiss_index, all_texts

    if faiss_index is None:
        build_faiss_index()

    # Preprocess query
    query = query.lower().strip()

    # Generate embedding for the query
    query_embedding = model.encode([query])

    # Search for the nearest neighbor using FAISS
    D, I = faiss_index.search(np.array(query_embedding), k=1)

    # Retrieve the matched text and its distance score
    match_text = all_texts[I[0][0]]
    match_distance = D[0][0]

    return match_text, match_distance

# Function to search for movies by genre
def get_movies_by_genre(genre):
    genre = genre.lower().strip()
    matching_movies = [movie[0] for movie in movies_data if movie[2].lower() == genre]
    return matching_movies

# Function to handle spelling errors for movie and series names
def get_closest_media_name(media_name):
    media_name = media_name.lower().strip()
    closest_matches = get_close_matches(media_name, media_names, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_media', methods=['POST'])
def check_media():
    input_text = request.form['media_name']

    # Check if input is a genre (only for movies)
    movie_genre_matches = get_movies_by_genre(input_text)
    if movie_genre_matches:
        result = f"Movies in genre '{input_text}': " + ", ".join(movie_genre_matches)
    else:
        # Handle media name search (movies or series)
        corrected_media_name = get_closest_media_name(input_text)
        if corrected_media_name:
            match_text, match_distance = search_media_by_name(corrected_media_name)
            result = f"Closest match: '{match_text}'"
        else:
            result = f"No match found for '{input_text}'."

    return render_template('index.html', result=result)

if __name__ == '__main__':
    # Build the FAISS index once on startup
    build_faiss_index()
    app.run(debug=True)
