import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from scipy.sparse import csr_matrix

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Change this in production

# Database setup
def init_db():
    with sqlite3.connect('movies.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                    (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS movies 
                    (movieId INTEGER PRIMARY KEY, title TEXT, genres TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS ratings 
                    (userId INTEGER, movieId INTEGER, rating REAL, 
                     PRIMARY KEY (userId, movieId))''')
        conn.commit()

# Load and optimize dataset
def load_data():
    # For demo, using a subset of MovieLens-like data
    # In production, use MovieLens 25M dataset or similar
    movies_data = [
        (1, 'The Matrix', 'Action|Sci-Fi'),
        (2, 'Titanic', 'Drama|Romance'),
        (3, 'Inception', 'Action|Sci-Fi|Thriller'),
        (4, 'The Room', 'Drama'),
        (5, 'Avatar', 'Action|Adventure|Sci-Fi')
    ]
    ratings_data = [
        (1, 1, 5.0), (1, 2, 3.0), (1, 3, 4.0), (1, 4, 1.0),
        (2, 1, 4.0), (2, 3, 5.0), (2, 5, 2.0),
        (3, 2, 4.0), (3, 3, 5.0), (3, 4, 2.0), (3, 5, 3.0),
        (4, 1, 3.0), (4, 2, 4.0), (4, 5, 5.0)
    ]
    
    with sqlite3.connect('movies.db') as conn:
        c = conn.cursor()
        c.executemany('INSERT OR REPLACE INTO movies VALUES (?, ?, ?)', movies_data)
        c.executemany('INSERT OR REPLACE INTO ratings VALUES (?, ?, ?)', ratings_data)
        conn.commit()

# Get user-movie matrix (sparse for scalability)
def get_user_movie_matrix():
    with sqlite3.connect('movies.db') as conn:
        df = pd.read_sql_query('SELECT userId, movieId, rating FROM ratings', conn)
    user_movie_matrix = df.pivot_table(index='userId', columns='movieId', values='rating', sparse=True)
    user_movie_matrix = user_movie_matrix.fillna(0)
    return csr_matrix(user_movie_matrix), user_movie_matrix.index, user_movie_matrix.columns

# Content-based filtering
def get_content_based_recommendations(movie_id, num_recommendations=3):
    with sqlite3.connect('movies.db') as conn:
        movies_df = pd.read_sql_query('SELECT * FROM movies', conn)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = movies_df[movies_df['movieId'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices][['movieId', 'title']].to_dict('records')

# Collaborative filtering
def get_collaborative_recommendations(user_id, num_recommendations=3):
    user_movie_matrix, user_ids, movie_ids = get_user_movie_matrix()
    similarity_matrix = cosine_similarity(user_movie_matrix)
    sim_df = pd.DataFrame(similarity_matrix, index=user_ids, columns=user_ids)
    
    user_ratings = user_movie_matrix[user_ids == user_id].toarray().flatten()
    unrated_movies = np.where(user_ratings == 0)[0]
    
    movie_scores = {}
    user_sim_scores = sim_df[user_id]
    for idx in unrated_movies:
        movie_id = movie_ids[idx]
        movie_ratings = user_movie_matrix[:, idx].toarray().flatten()
        weighted_sum = np.sum(movie_ratings * user_sim_scores)
        sim_sum = np.sum(user_sim_scores[movie_ratings > 0])
        if sim_sum > 0:
            movie_scores[movie_id] = weighted_sum / sim_sum
    
    recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    with sqlite3.connect('movies.db') as conn:
        movies_df = pd.read_sql_query('SELECT * FROM movies', conn)
    return [{'movieId': int(k), 'title': movies_df[movies_df['movieId'] == int(k)]['title'].iloc[0]} 
            for k, _ in recommended_movies]

# Routes
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    with sqlite3.connect('movies.db') as conn:
        movies = pd.read_sql_query('SELECT * FROM movies LIMIT 10', conn).to_dict('records')
    return render_template('index.html', movies=movies)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect('movies.db') as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()
            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                return redirect(url_for('index'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        try:
            with sqlite3.connect('movies.db') as conn:
                c = conn.cursor()
                c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                         (username, hashed_password))
                conn.commit()
                c.execute('SELECT id FROM users WHERE username = ?', (username,))
                session['user_id'] = c.fetchone()[0]
                return redirect(url_for('index'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists')
    return render_template('register.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    movie_id = int(request.form['movie_id'])
    collab_recs = get_collaborative_recommendations(session['user_id'])
    content_recs = get_content_based_recommendations(movie_id)
    return render_template('recommendations.html', 
                         collab_recs=collab_recs, 
                         content_recs=content_recs)

@app.route('/rate', methods=['POST'])
def rate():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    movie_id = int(request.form['movie_id'])
    rating = float(request.form['rating'])
    with sqlite3.connect('movies.db') as conn:
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO ratings (userId, movieId, rating) VALUES (?, ?, ?)',
                 (session['user_id'], movie_id, rating))
        conn.commit()
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    load_data()
    app.run(debug=True)