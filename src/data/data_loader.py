import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieDataLoader:
    def __init__(self, db_path: str = "data/tmdb_movies.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)

    def load_movies(self) -> pd.DataFrame:
        """Carga las películas desde la base de datos."""
        logger.info("Cargando películas desde la base de datos: data/tmdb_movies.db")
        try:
            conn = sqlite3.connect('data/tmdb_movies.db')
            query = """
            SELECT m.*, GROUP_CONCAT(g.name) as genres
            FROM movies m
            LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
            LEFT JOIN genres g ON mg.genre_id = g.id
            GROUP BY m.movie_id
            """
            movies_df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Asegurarse de que movie_id sea entero
            movies_df['movieId'] = movies_df['movie_id'].astype(int)
            
            # Convertir géneros de string a lista
            movies_df['genres'] = movies_df['genres'].fillna('').apply(lambda x: x.split(',') if x else [])
            
            return movies_df
        except Exception as e:
            logger.error(f"Error cargando películas: {str(e)}")
            raise

    def load_ratings(self) -> pd.DataFrame:
        """Carga la tabla de calificaciones desde la base de datos."""
        logger.info(f"Cargando calificaciones desde la base de datos: {self.db_path}")
        ratings_df = pd.read_sql_query("SELECT * FROM ratings", self.conn)
        ratings_df['userId'] = ratings_df['user_id'].astype(int)
        ratings_df['movieId'] = ratings_df['movie_id'].astype(int)
        logger.info(f"Calificaciones cargadas: {len(ratings_df)}")
        return ratings_df

    def load_users(self) -> pd.DataFrame:
        """Carga la tabla de usuarios desde la base de datos."""
        logger.info(f"Cargando usuarios desde la base de datos: {self.db_path}")
        users_df = pd.read_sql_query("SELECT * FROM users", self.conn)
        logger.info(f"Usuarios cargados: {len(users_df)}")
        return users_df

    def create_user_movie_matrix(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        logger.info("Creando matriz usuario-película...")
        user_movie_matrix = ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        user_to_idx = {user: idx for idx, user in enumerate(user_movie_matrix.index)}
        movie_to_idx = {movie: idx for idx, movie in enumerate(user_movie_matrix.columns)}
        logger.info(f"Matriz creada con {len(user_to_idx)} usuarios y {len(movie_to_idx)} películas")
        return user_movie_matrix, user_to_idx, movie_to_idx

    def get_movie_features(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Procesando características de películas...")
        movies_df['genres'] = movies_df['genres'].fillna('(no genres listed)')
        genres = movies_df['genres'].str.get_dummies('|')
        movies_df['title_features'] = movies_df['title'].fillna('')
        features = pd.concat([
            movies_df[['movieId', 'title', 'title_features']],
            genres
        ], axis=1)
        logger.info(f"Características procesadas: {len(features.columns)} columnas")
        return features

    def preprocess_data(self) -> Dict:
        try:
            logger.info("Iniciando preprocesamiento de datos...")
            movies_df = self.load_movies()
            ratings_df = self.load_ratings()
            users_df = self.load_users()
            user_movie_matrix, user_to_idx, movie_to_idx = self.create_user_movie_matrix(ratings_df)
            movie_features = self.get_movie_features(movies_df)
            logger.info("Preprocesamiento completado exitosamente")
            return {
                "movies": movies_df,
                "ratings": ratings_df,
                "users": users_df,
                "user_movie_matrix": user_movie_matrix,
                "user_to_idx": user_to_idx,
                "movie_to_idx": movie_to_idx,
                "movie_features": movie_features
            }
        except Exception as e:
            logger.error(f"Error en el preprocesamiento: {str(e)}")
            raise 