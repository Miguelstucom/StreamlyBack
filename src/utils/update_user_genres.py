import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_user_genre_preferences(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, n_top_genres: int = 3) -> Dict[int, List[str]]:
    """
    Analiza las preferencias de géneros de cada usuario basado en sus calificaciones.
    
    Args:
        ratings_df: DataFrame con las calificaciones
        movies_df: DataFrame con las películas y sus géneros
        n_top_genres: Número de géneros principales a considerar por usuario
    
    Returns:
        Dict con el userId como clave y lista de géneros preferidos como valor
    """
    logger.info("Analizando preferencias de géneros...")
    
    # Separar los géneros en columnas dummy
    genres = movies_df['genres'].str.get_dummies('|')
    
    # Unir las calificaciones con los géneros
    user_movie_genres = pd.merge(
        ratings_df,
        pd.concat([movies_df[['movieId']], genres], axis=1),
        on='movieId'
    )
    
    # Calcular el score promedio por género para cada usuario
    genre_columns = genres.columns
    user_genre_scores = user_movie_genres.groupby('userId')[genre_columns].agg(
        lambda x: (x * user_movie_genres.loc[x.index, 'rating']).mean()
    )
    
    # Obtener los géneros más valorados por cada usuario
    user_preferences = {}
    for user_id in user_genre_scores.index:
        # Obtener los géneros ordenados por score
        user_scores = user_genre_scores.loc[user_id]
        top_genres = user_scores[user_scores > 0].sort_values(ascending=False)
        
        # Tomar los n_top_genres más valorados
        top_genres = top_genres.head(n_top_genres)
        
        # Guardar los géneros preferidos
        user_preferences[user_id] = top_genres.index.tolist()
    
    logger.info(f"Preferencias de géneros analizadas para {len(user_preferences)} usuarios")
    return user_preferences

def update_users_with_genres():
    """Actualiza el archivo users.csv con las preferencias de géneros."""
    try:
        # Leer los archivos
        data_dir = Path("data")
        users_file = data_dir / "users.csv"
        movies_file = data_dir / "movies.csv"
        ratings_file = data_dir / "ratings.csv"
        
        logger.info("Leyendo archivos...")
        users_df = pd.read_csv(users_file)
        movies_df = pd.read_csv(movies_file)
        ratings_df = pd.read_csv(ratings_file)
        
        # Crear una copia de seguridad
        backup_file = users_file.with_suffix('.csv.bak')
        users_df.to_csv(backup_file, index=False)
        logger.info(f"Copia de seguridad creada en: {backup_file}")
        
        # Obtener preferencias de géneros
        user_preferences = get_user_genre_preferences(ratings_df, movies_df)
        
        # Añadir columna de géneros preferidos
        users_df['preferred_genres'] = users_df['userId'].map(
            lambda x: '|'.join(user_preferences.get(x, ['No genres listed']))
        )
        
        # Guardar el archivo actualizado
        users_df.to_csv(users_file, index=False)
        logger.info("Archivo users.csv actualizado con preferencias de géneros")
        
        # Mostrar algunas estadísticas
        genre_counts = users_df['preferred_genres'].str.split('|').explode().value_counts()
        logger.info("\nDistribución de géneros preferidos:")
        for genre, count in genre_counts.items():
            logger.info(f"{genre}: {count} usuarios")
            
    except Exception as e:
        logger.error(f"Error al actualizar preferencias de géneros: {str(e)}")
        raise

if __name__ == "__main__":
    update_users_with_genres() 