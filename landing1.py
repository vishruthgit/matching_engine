from flask import Flask, request, render_template, jsonify
import psycopg2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

# Connect to PostgreSQL
def get_db_connection():
    return psycopg2.connect(**db_config)

# Function to get data and build the FAISS index
def build_faiss_index():
    global faiss_index, all_texts

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch movie data
    cursor.execute("SELECT name, director, genre FROM movies")
    movies_data = cursor.fetchall()

    # Fetch web series data
    cursor.execute("SELECT name, season_number, episode_number FROM web_series")
    web_series_data = cursor.fetchall()

    cursor.close()
    conn.close()

    # Prepare data for embedding
    all_texts = []

    # Combine movies and web series data into text for embedding
    for movie in movies_data:
        all_texts.append(f"{movie[0]} {movie[1]} {movie[2]}".lower().strip())

    for series in web_series_data:
        all_texts.append(f"{series[0]} season {series[1]} episode {series[2]}".lower().strip())

    # Generate embeddings
    embeddings = model.encode(all_texts)

    # Create FAISS index
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings))

# Function to search for a matching movie or series
def search_for_match(query):
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_media', methods=['POST'])
def check_media():
    media_name = request.form['media_name']

    # Search for a match in the database
    match_text, match_distance = search_for_match(media_name)

    # Adjusted similarity threshold
    similarity_threshold = 0.7
    if match_distance < similarity_threshold:
        result = f"Found a match: '{match_text}'"
    else:
        result = f"No close match found for '{media_name}'."

    return render_template('index.html', result=result)

if __name__ == '__main__':
    # Build the FAISS index once on startup
    build_faiss_index()
    app.run(debug=True)
