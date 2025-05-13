import pandas as pd
from typing import List, Dict
import numpy as np

def clean_text(text: str) -> str:
    """Limpia el texto eliminando caracteres especiales y normalizando espacios."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())

def calculate_movie_stats(ratings_df: pd.DataFrame) -> Dict:
    """Calcula estadísticas básicas de las calificaciones."""
    stats = {
        "total_ratings": len(ratings_df),
        "average_rating": ratings_df['rating'].mean(),
        "rating_distribution": ratings_df['rating'].value_counts().to_dict()
    }
    return stats

def get_popular_movies(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, n: int = 10) -> List[Dict]:
    """Obtiene las películas más populares basadas en el número de calificaciones."""
    movie_ratings = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    
    movie_ratings.columns = ['movieId', 'rating_count', 'rating_mean']
    popular_movies = movie_ratings.sort_values('rating_count', ascending=False).head(n)
    
    # Unir con la información de las películas
    result = pd.merge(popular_movies, movies_df, on='movieId')
    return result.to_dict(orient='records') 