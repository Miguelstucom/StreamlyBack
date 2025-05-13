import pandas as pd
import requests
import sqlite3
import time
from typing import Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TMDB API configuration
TMDB_API_KEY = ("f05ca11483ec2e8a4a420b03ad147f2b"
                "")  # Replace with your TMDB API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original"


def create_database():
    """Create SQLite database and tables for TMDB movie data."""
    conn = sqlite3.connect('data/tmdb_movies.db')
    cursor = conn.cursor()

    # Create movies table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS movies (
        movie_id INTEGER PRIMARY KEY,
        tmdb_id INTEGER UNIQUE,
        imdb_id TEXT,
        title TEXT,
        original_title TEXT,
        overview TEXT,
        tagline TEXT,
        release_date TEXT,
        runtime INTEGER,
        budget INTEGER,
        revenue INTEGER,
        popularity REAL,
        vote_average REAL,
        vote_count INTEGER,
        status TEXT,
        adult BOOLEAN,
        video BOOLEAN,
        poster_path TEXT,
        backdrop_path TEXT,
        homepage TEXT,
        original_language TEXT
    )
    ''')

    # Create genres table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS genres (
        id INTEGER PRIMARY KEY,
        name TEXT
    )
    ''')

    # Create movie_genres table (many-to-many relationship)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS movie_genres (
        movie_id INTEGER,
        genre_id INTEGER,
        FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
        FOREIGN KEY (genre_id) REFERENCES genres (id),
        PRIMARY KEY (movie_id, genre_id)
    )
    ''')

    # Create production_companies table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS production_companies (
        id INTEGER PRIMARY KEY,
        name TEXT,
        logo_path TEXT,
        origin_country TEXT
    )
    ''')

    # Create movie_production_companies table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS movie_production_companies (
        movie_id INTEGER,
        company_id INTEGER,
        FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
        FOREIGN KEY (company_id) REFERENCES production_companies (id),
        PRIMARY KEY (movie_id, company_id)
    )
    ''')

    # Create production_countries table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS production_countries (
        iso_3166_1 TEXT PRIMARY KEY,
        name TEXT
    )
    ''')

    # Create movie_production_countries table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS movie_production_countries (
        movie_id INTEGER,
        country_code TEXT,
        FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
        FOREIGN KEY (country_code) REFERENCES production_countries (iso_3166_1),
        PRIMARY KEY (movie_id, country_code)
    )
    ''')

    # Create spoken_languages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS spoken_languages (
        iso_639_1 TEXT PRIMARY KEY,
        name TEXT,
        english_name TEXT
    )
    ''')

    # Create movie_spoken_languages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS movie_spoken_languages (
        movie_id INTEGER,
        language_code TEXT,
        FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
        FOREIGN KEY (language_code) REFERENCES spoken_languages (iso_639_1),
        PRIMARY KEY (movie_id, language_code)
    )
    ''')

    # Create collections table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS collections (
        id INTEGER PRIMARY KEY,
        name TEXT,
        poster_path TEXT,
        backdrop_path TEXT
    )
    ''')

    # Create movie_collections table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS movie_collections (
        movie_id INTEGER,
        collection_id INTEGER,
        FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
        FOREIGN KEY (collection_id) REFERENCES collections (id),
        PRIMARY KEY (movie_id, collection_id)
    )
    ''')

    conn.commit()
    return conn


def fetch_movie_data(tmdb_id: int) -> Dict[str, Any]:
    """Fetch movie data from TMDB API."""
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-US'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching movie {tmdb_id}: {str(e)}")
        return None


def insert_movie_data(conn: sqlite3.Connection, movie_data: Dict[str, Any], movie_id: int):
    """Insert movie data into the database."""
    cursor = conn.cursor()

    # Insert movie
    cursor.execute('''
    INSERT OR REPLACE INTO movies (
        movie_id, tmdb_id, imdb_id, title, original_title, overview, tagline,
        release_date, runtime, budget, revenue, popularity,
        vote_average, vote_count, status, adult, video,
        poster_path, backdrop_path, homepage, original_language
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        movie_id,
        movie_data['id'],
        movie_data.get('imdb_id'),
        movie_data.get('title'),
        movie_data.get('original_title'),
        movie_data.get('overview'),
        movie_data.get('tagline'),
        movie_data.get('release_date'),
        movie_data.get('runtime'),
        movie_data.get('budget'),
        movie_data.get('revenue'),
        movie_data.get('popularity'),
        movie_data.get('vote_average'),
        movie_data.get('vote_count'),
        movie_data.get('status'),
        movie_data.get('adult', False),
        movie_data.get('video', False),
        movie_data.get('poster_path'),
        movie_data.get('backdrop_path'),
        movie_data.get('homepage'),
        movie_data.get('original_language')
    ))

    # Insert genres
    for genre in movie_data.get('genres', []):
        cursor.execute('INSERT OR IGNORE INTO genres (id, name) VALUES (?, ?)',
                      (genre['id'], genre['name']))
        cursor.execute('INSERT OR IGNORE INTO movie_genres (movie_id, genre_id) VALUES (?, ?)',
                      (movie_id, genre['id']))

    # Insert production companies
    for company in movie_data.get('production_companies', []):
        cursor.execute('''
        INSERT OR REPLACE INTO production_companies (id, name, logo_path, origin_country)
        VALUES (?, ?, ?, ?)
        ''', (company['id'], company['name'], company.get('logo_path'),
              company.get('origin_country')))
        cursor.execute('''
        INSERT OR IGNORE INTO movie_production_companies (movie_id, company_id)
        VALUES (?, ?)
        ''', (movie_id, company['id']))

    # Insert production countries
    for country in movie_data.get('production_countries', []):
        cursor.execute('''
        INSERT OR REPLACE INTO production_countries (iso_3166_1, name)
        VALUES (?, ?)
        ''', (country['iso_3166_1'], country['name']))
        cursor.execute('''
        INSERT OR IGNORE INTO movie_production_countries (movie_id, country_code)
        VALUES (?, ?)
        ''', (movie_id, country['iso_3166_1']))

    # Insert spoken languages
    for language in movie_data.get('spoken_languages', []):
        cursor.execute('''
        INSERT OR REPLACE INTO spoken_languages (iso_639_1, name, english_name)
        VALUES (?, ?, ?)
        ''', (language['iso_639_1'], language['name'],
              language.get('english_name')))
        cursor.execute('''
        INSERT OR IGNORE INTO movie_spoken_languages (movie_id, language_code)
        VALUES (?, ?)
        ''', (movie_id, language['iso_639_1']))

    # Insert collection if exists
    if movie_data.get('belongs_to_collection'):
        collection = movie_data['belongs_to_collection']
        cursor.execute('''
        INSERT OR REPLACE INTO collections (id, name, poster_path, backdrop_path)
        VALUES (?, ?, ?, ?)
        ''', (collection['id'], collection['name'],
              collection.get('poster_path'),
              collection.get('backdrop_path')))
        cursor.execute('''
        INSERT OR IGNORE INTO movie_collections (movie_id, collection_id)
        VALUES (?, ?)
        ''', (movie_id, collection['id']))

    conn.commit()


def main():
    # Create data directory if it doesn't exist
    Path('data').mkdir(exist_ok=True)

    # Read links.csv
    links_df = pd.read_csv('data/links.csv')
    total_movies = len(links_df)
    
    # Create database and tables
    conn = create_database()
    
    # Process all movies
    processed = 0
    for _, row in links_df.iterrows():
        movie_id = row['movieId']
        tmdb_id = row['tmdbId']
        if pd.isna(tmdb_id):
            continue
            
        processed += 1
        logger.info(f"Processing movie {processed}/{total_movies} - ID: {movie_id} (TMDB ID: {tmdb_id})")
        movie_data = fetch_movie_data(int(tmdb_id))
        
        if movie_data:
            insert_movie_data(conn, movie_data, int(movie_id))
            logger.info(f"Successfully processed movie: {movie_data.get('title')}")
        else:
            logger.warning(f"Failed to fetch data for movie ID: {movie_id}")
        
        # Add delay to respect API rate limits
        time.sleep(0.25)
    
    conn.close()
    logger.info(f"Finished processing {processed} movies")


if __name__ == "__main__":
    main()