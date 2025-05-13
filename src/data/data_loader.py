import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieDataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
    def load_movies(self, filename: str = "movies.csv") -> pd.DataFrame:
        """Carga el archivo CSV de películas."""
        file_path = self.data_dir / filename
        logger.info(f"Cargando archivo de películas: {file_path}")
        movies_df = pd.read_csv(file_path)
        # Asegurarse de que los IDs sean enteros
        movies_df['movieId'] = movies_df['movieId'].astype(int)
        logger.info(f"Películas cargadas: {len(movies_df)}")
        return movies_df
    
    def load_ratings(self, filename: str = "ratings.csv") -> pd.DataFrame:
        """Carga el archivo CSV de calificaciones."""
        file_path = self.data_dir / filename
        logger.info(f"Cargando archivo de calificaciones: {file_path}")
        ratings_df = pd.read_csv(file_path)
        # Asegurarse de que los IDs sean enteros
        ratings_df['userId'] = ratings_df['userId'].astype(int)
        ratings_df['movieId'] = ratings_df['movieId'].astype(int)
        logger.info(f"Calificaciones cargadas: {len(ratings_df)}")
        return ratings_df
    
    def load_users(self, filename: str = "users.csv") -> pd.DataFrame:
        """Carga el archivo CSV de usuarios."""
        file_path = self.data_dir / filename
        logger.info(f"Cargando archivo de usuarios: {file_path}")
        users_df = pd.read_csv(file_path)
        logger.info(f"Usuarios cargados: {len(users_df)}")
        return users_df
    
    def create_user_movie_matrix(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Crea la matriz usuario-película y los mapeos de índices."""
        logger.info("Creando matriz usuario-película...")
        # Crear matriz usuario-película
        user_movie_matrix = ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Crear mapeos de índices
        user_to_idx = {user: idx for idx, user in enumerate(user_movie_matrix.index)}
        movie_to_idx = {movie: idx for idx, movie in enumerate(user_movie_matrix.columns)}
        
        logger.info(f"Matriz creada con {len(user_to_idx)} usuarios y {len(movie_to_idx)} películas")
        return user_movie_matrix, user_to_idx, movie_to_idx
    
    def get_movie_features(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Extrae características relevantes de las películas."""
        logger.info("Procesando características de películas...")
        
        # Asegurarse de que los géneros no sean nulos
        movies_df['genres'] = movies_df['genres'].fillna('(no genres listed)')
        
        # Separar géneros en columnas dummy
        genres = movies_df['genres'].str.get_dummies('|')
        
        # Añadir el título como característica
        movies_df['title_features'] = movies_df['title'].fillna('')
        
        # Combinar todas las características
        features = pd.concat([
            movies_df[['movieId', 'title', 'title_features']],
            genres
        ], axis=1)
        print(features)
        
        logger.info(f"Características procesadas: {len(features.columns)} columnas")
        return features
    
    def preprocess_data(self) -> Dict:
        """Preprocesa todos los datos necesarios para el sistema de recomendación."""
        try:
            logger.info("Iniciando preprocesamiento de datos...")
            
            movies_df = self.load_movies()
            ratings_df = self.load_ratings()
            users_df = self.load_users()
            
            # Crear matriz usuario-película
            user_movie_matrix, user_to_idx, movie_to_idx = self.create_user_movie_matrix(ratings_df)
            
            # Obtener características de películas
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