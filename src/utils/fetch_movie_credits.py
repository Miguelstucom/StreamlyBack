import sqlite3
import requests
import time
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MovieCreditsFetcher:
    def __init__(self, db_path: str, api_key: str):
        self.db_path = db_path
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        
    def create_tables(self):
        """Create necessary tables for storing movie credits."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Create people table (actors, directors, producers)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    original_name TEXT,
                    gender INTEGER,
                    popularity REAL,
                    profile_path TEXT,
                    known_for_department TEXT
                )
            """)
            
            # Create movie_cast table (relationship between movies and actors)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS movie_cast (
                    movie_id INTEGER,
                    person_id INTEGER,
                    character TEXT,
                    credit_id TEXT,
                    cast_order INTEGER,
                    FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
                    FOREIGN KEY (person_id) REFERENCES people(id),
                    PRIMARY KEY (movie_id, person_id, credit_id)
                )
            """)
            
            # Create movie_crew table (relationship between movies and crew)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS movie_crew (
                    movie_id INTEGER,
                    person_id INTEGER,
                    department TEXT,
                    job TEXT,
                    credit_id TEXT,
                    FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
                    FOREIGN KEY (person_id) REFERENCES people(id),
                    PRIMARY KEY (movie_id, person_id, credit_id)
                )
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def fetch_movie_credits(self, tmdb_id: int) -> Dict:
        """Fetch credits for a specific movie from TMDB API."""
        url = f"{self.base_url}/movie/{tmdb_id}/credits"
        params = {
            'api_key': self.api_key,
            'language': 'en-EN'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching credits for movie {tmdb_id}: {str(e)}")
            return None
    
    def store_credits(self, movie_id: int, credits_data: Dict):
        """Store movie credits in the database."""
        if not credits_data:
            return
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Process cast (actors)
            for cast_member in credits_data.get('cast', []):
                # Insert or update person
                cursor.execute("""
                    INSERT OR REPLACE INTO people 
                    (id, name, original_name, gender, popularity, profile_path, known_for_department)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    cast_member['id'],
                    cast_member['name'],
                    cast_member['original_name'],
                    cast_member['gender'],
                    cast_member['popularity'],
                    cast_member['profile_path'],
                    cast_member['known_for_department']
                ))
                
                # Insert cast relationship
                cursor.execute("""
                    INSERT OR REPLACE INTO movie_cast 
                    (movie_id, person_id, character, credit_id, cast_order)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    movie_id,
                    cast_member['id'],
                    cast_member['character'],
                    cast_member['credit_id'],
                    cast_member['order']
                ))
            
            # Process crew (directors and producers)
            for crew_member in credits_data.get('crew', []):
                # Only store directors and producers
                if crew_member['job'] not in ['Director', 'Producer']:
                    continue
                
                # Insert or update person
                cursor.execute("""
                    INSERT OR REPLACE INTO people 
                    (id, name, original_name, gender, popularity, profile_path, known_for_department)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    crew_member['id'],
                    crew_member['name'],
                    crew_member['original_name'],
                    crew_member['gender'],
                    crew_member['popularity'],
                    crew_member['profile_path'],
                    crew_member['known_for_department']
                ))
                
                # Insert crew relationship
                cursor.execute("""
                    INSERT OR REPLACE INTO movie_crew 
                    (movie_id, person_id, department, job, credit_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    movie_id,
                    crew_member['id'],
                    crew_member['department'],
                    crew_member['job'],
                    crew_member['credit_id']
                ))
            
            conn.commit()
        finally:
            conn.close()
    
    def process_all_movies(self):
        """Process all movies in the database to fetch and store their credits."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Get all movie IDs and their TMDB IDs
            cursor.execute("SELECT movie_id, tmdb_id FROM movies")
            movies = cursor.fetchall()
            
            for movie_id, tmdb_id in movies:
                logger.info(f"Processing movie {movie_id} (TMDB ID: {tmdb_id})")
                
                # Fetch credits
                credits_data = self.fetch_movie_credits(tmdb_id)
                if credits_data:
                    # Store credits
                    self.store_credits(movie_id, credits_data)
                
                # Sleep to respect API rate limits
                time.sleep(0.25)
                
        finally:
            conn.close()

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize fetcher
    fetcher = MovieCreditsFetcher(
        db_path='data/tmdb_movies.db',
        api_key='f05ca11483ec2e8a4a420b03ad147f2b'
    )
    
    # Create tables
    fetcher.create_tables()
    
    # Process all movies
    fetcher.process_all_movies()

if __name__ == "__main__":
    main() 