import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import pickle
import sqlite3

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import MovieDataLoader
from src.models.recommender import MovieRecommenderDL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_movie_data_from_db():
    """Get all movie data from SQLite database."""
    conn = sqlite3.connect('data/tmdb_movies.db')
    try:
        # Get all movies
        movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
        
        # Get genres
        genres_df = pd.read_sql_query("""
            SELECT m.movie_id, GROUP_CONCAT(g.name) as genres
            FROM movies m
            JOIN movie_genres mg ON m.movie_id = mg.movie_id
            JOIN genres g ON mg.genre_id = g.id
            GROUP BY m.movie_id
        """, conn)
        
        # Merge movies with genres
        movies_df = pd.merge(movies_df, genres_df, on='movie_id', how='left')
        
        return movies_df
    finally:
        conn.close()

def train_and_save_model():
    """Train the model and save it to disk."""
    try:
        # Initialize data loader and recommender
        data_loader = MovieDataLoader()
        recommender = MovieRecommenderDL(embedding_dim=50)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data = data_loader.preprocess_data()
        
        # Load movie data from SQLite
        logger.info("Loading movie data from database...")
        movies_df = get_movie_data_from_db()
        
        # Add movies data to the data dictionary
        data['movies'] = movies_df
        
        # Train model
        logger.info("Training model...")
        recommender.fit(data)
        
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            'model': recommender,
            'user_movie_matrix': data['user_movie_matrix'],
            'user_to_idx': data['user_to_idx'],
            'movie_to_idx': data['movie_to_idx'],
            'movies': data['movies'],
            'ratings': data['ratings'],
            'users': data['users']
        }
        
        # Save model and data
        logger.info("Saving model and data...")
        with open(models_dir / "trained_model.pkl", "wb") as f:
            pickle.dump(save_data, f)
        
        logger.info("Model trained and saved successfully!")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model() 