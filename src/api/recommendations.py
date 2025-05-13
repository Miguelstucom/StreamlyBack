from fastapi import APIRouter, HTTPException
from typing import List, Dict
import pandas as pd
import numpy as np
from src.models.recommender import MovieRecommenderDL
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Cargar datos
ratings_df = pd.read_csv('data/ratings.csv')
users_df = pd.read_csv('data/users.csv')

# Crear matriz usuario-película
user_movie_matrix = ratings_df.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# Crear mapeos de índices
user_to_idx = {user_id: idx for idx, user_id in enumerate(user_movie_matrix.index)}
movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(user_movie_matrix.columns)}

# Inicializar recomendador
recommender = MovieRecommenderDL()

# Preparar datos para el recomendador
data = {
    'user_movie_matrix': user_movie_matrix,
    'user_to_idx': user_to_idx,
    'movie_to_idx': movie_to_idx
}

# Entrenar modelo
try:
    recommender.fit(data)
    logger.info("Modelo entrenado exitosamente")
except Exception as e:
    logger.error(f"Error al entrenar el modelo: {str(e)}")
    raise HTTPException(status_code=500, detail="Error al inicializar el recomendador")

@router.get("/recommendations/user/{user_id}")
async def get_user_recommendations(user_id: int) -> List[Dict]:
    """Obtiene recomendaciones para un usuario específico."""
    try:
        recommendations = recommender.get_collaborative_recommendations(user_id)
        return recommendations
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener recomendaciones para usuario {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener recomendaciones")

@router.get("/recommendations/movie/{movie_id}")
async def get_movie_recommendations(movie_id: int) -> List[Dict]:
    """Obtiene películas similares a una película específica."""
    try:
        recommendations = recommender.get_content_based_recommendations(movie_id)
        return recommendations
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener recomendaciones para película {movie_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener recomendaciones") 