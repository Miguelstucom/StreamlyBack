from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
import sys
from pathlib import Path
from datetime import timedelta
import pandas as pd
import requests
import asyncio
import aiohttp
import sqlite3
import numpy as np
import logging
import pickle
import hashlib
import os
from jose import JWTError, jwt
from passlib.context import CryptContext

# Añadir el directorio src al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import MovieDataLoader
from src.models.recommender import MovieRecommenderDL
from src.auth.auth_handler import AuthHandler, ACCESS_TOKEN_EXPIRE_MINUTES
from src.auth.models import Token, UserLogin, UserResponse
from src.utils.generate_ratings_descriptions import generate_rating_description
from src.utils.chatbot import MovieChatbot
from src.utils.svd_recommender import SVDRecommender
from src.utils.genre_recommender import GenreRecommender

logger = logging.getLogger(__name__)

app = FastAPI(title="Movie Recommender API")

# Configurar CORS para Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el cargador de datos, el recomendador y el manejador de autenticación
data_loader = MovieDataLoader()
auth_handler = AuthHandler()

# Configurar OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# TMDB API configuration
TMDB_API_KEY = "f05ca11483ec2e8a4a420b03ad147f2b"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Load trained model and data
try:
    logger.info("Loading trained model and data...")
    with open("models/trained_model.pkl", "rb") as f:
        saved_data = pickle.load(f)
        recommender = saved_data['model']
        data = {
            'user_movie_matrix': saved_data['user_movie_matrix'],
            'user_to_idx': saved_data['user_to_idx'],
            'movie_to_idx': saved_data['movie_to_idx'],
            'movies': saved_data['movies'],
            'ratings': saved_data['ratings'],
            'users': saved_data['users']
        }
    logger.info("Model and data loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load trained model. Please run train_model.py first.")

# Initialize chatbot
chatbot = MovieChatbot()

# Configuración de seguridad
SECRET_KEY = "your-secret-key"  # Cambiar en producción
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Modelo para el registro de usuario
class UserRegister(BaseModel):
    username: str
    firstname: str
    lastname: str
    email: str
    password: str
    age: int
    occupation: str
    preferred_genres: List[int]  # Lista de genre_id

# Función para crear un nuevo usuario
def create_user(user: UserRegister):
    conn = sqlite3.connect("data/tmdb_movies.db")
    cursor = conn.cursor()
    try:
        # Verificar si el username o email ya existe
        cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (user.username, user.email))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username or email already registered")
        
        # Obtener el último userId y sumar 1
        cursor.execute("SELECT MAX(userId) FROM users")
        last_user_id = cursor.fetchone()[0]
        next_user_id = (last_user_id or 0) + 1
        
        # Insertar usuario con el siguiente userId
        cursor.execute("""
        INSERT INTO users (userId, username, firstname, lastname, email, passwordHash, age, occupation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            next_user_id,
            user.username,
            user.firstname,
            user.lastname,
            user.email,
            pwd_context.hash(user.password),
            user.age,
            user.occupation
        ))
        
        # Insertar géneros preferidos usando los genre_id proporcionados
        for genre_id in user.preferred_genres:
            cursor.execute("INSERT INTO user_genres (user_id, genre_id) VALUES (?, ?)", (next_user_id, genre_id))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# Endpoint para registrar un usuario
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserRegister):
    create_user(user)
    return {"message": "User registered successfully"}

async def get_tmdb_movie_details(tmdb_id: int) -> Dict:
    """Obtiene detalles de una película desde TMDB."""
    async with aiohttp.ClientSession() as session:
        url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "es-ES"  # Obtener descripción en español
        }
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "poster_path": f"{TMDB_IMAGE_BASE_URL}{data.get('poster_path')}" if data.get('poster_path') else None,
                    "overview": data.get('overview'),
                    "backdrop_path": f"{TMDB_IMAGE_BASE_URL}{data.get('backdrop_path')}" if data.get('backdrop_path') else None
                }
            return {"poster_path": None, "overview": None, "backdrop_path": None}

class RecommendationRequest(BaseModel):
    n_recommendations: int = 5

class MovieRecommendationRequest(RecommendationRequest):
    movie_id: int

class UserRecommendationRequest(RecommendationRequest):
    user_id: int

class ReviewRequest(BaseModel):
    rating: float
    movie_id: int
    description: str

    @validator('rating')
    def validate_rating(cls, v):
        if v < 0.5 or v > 5.0:
            raise ValueError('Rating must be between 0.5 and 5.0')
        # Redondear al 0.5 más cercano
        return round(v * 2) / 2

    @validator('description')
    def validate_description(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Description cannot be empty')
        if len(v) > 1000:  # Limitar a 1000 caracteres
            raise ValueError('Description must be less than 1000 characters')
        return v.strip()

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Obtiene el usuario actual basado en el token."""
    payload = auth_handler.verify_token(token)
    email = payload.get("sub")
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"email": email}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Endpoint para obtener token de acceso."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Buscar usuario por email
        cursor.execute('SELECT * FROM users WHERE email = ?', (form_data.username,))
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Email o contraseña incorrectos",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Convertir resultado a diccionario
        user_dict = {
            "userId": user[0],
            "username": user[1],
            "firstname": user[2],
            "lastname": user[3],
            "email": user[4],
            "passwordHash": user[5],
            "age": user[6],
            "occupation": user[7]
        }
        
        # Verificar contraseña
        if not pwd_context.verify(form_data.password, user_dict["passwordHash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Email o contraseña incorrectos",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Crear token de acceso
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_handler.create_access_token(
            data={"sub": user_dict["email"]}, expires_delta=access_token_expires
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )
    finally:
        if 'conn' in locals():
            conn.close()

@app.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    """Endpoint alternativo para login usando email y contraseña."""
    conn = sqlite3.connect("data/tmdb_movies.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE email = ?", (user_data.email,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Email o contraseña incorrectos",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_dict = {
            "userId": user[0],
            "username": user[1],
            "firstname": user[2],
            "lastname": user[3],
            "email": user[4],
            "passwordHash": user[5],
            "age": user[6],
            "occupation": user[7]
        }
        if not pwd_context.verify(user_data.password, user_dict["passwordHash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Email o contraseña incorrectos",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_handler.create_access_token(
            data={"sub": user_dict["email"]}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        conn.close()

@app.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """Obtiene información del usuario actual."""
    try:
        conn = sqlite3.connect("data/tmdb_movies.db")
        cursor = conn.cursor()
        
        # Obtener información del usuario directamente de la base de datos
        cursor.execute("SELECT * FROM users WHERE email = ?", (current_user['email'],))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
            
        user_dict = {
            "userId": user[0],
            "username": user[1],
            "firstname": user[2],
            "lastname": user[3],
            "email": user[4],
            "age": user[6],
            "occupation": user[7]
        }
        
        # Obtener géneros preferidos
        cursor.execute("""
        SELECT g.name FROM genres g
        JOIN user_genres ug ON g.id = ug.genre_id
        WHERE ug.user_id = ?
        """, (user_dict["userId"],))
        preferred_genres = [row[0] for row in cursor.fetchall()]
        
        name = f"{user_dict['firstname']} {user_dict['lastname']}".strip()
        return {
            "id": user_dict["userId"],
            "email": user_dict["email"],
            "name": name,
            "preferred_genres": preferred_genres
        }
    except Exception as e:
        logger.error(f"Error al obtener información del usuario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener información del usuario")
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/")
async def root():
    return {
        "message": "Bienvenido a la API de Recomendación de Películas",
        "endpoints": {
            "/login": "Login con email y contraseña",
            "/movies": "Lista todas las películas",
            "/recommendations/movie/{movie_id}": "Recomendaciones basadas en una película",
            "/recommendations/user/{user_id}": "Recomendaciones para un usuario"
        }
    }

@app.get("/movies")
async def get_movies(current_user: dict = Depends(get_current_user)):
    try:
        return {"movies": data['movies'].to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations/movie/{movie_id}")
async def get_movie_recommendations(
    movie_id: int,
    request: RecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        recommendations = recommender.get_content_based_recommendations(
            movie_id,
            request.n_recommendations
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/recommendations/user/{user_id}")
async def get_user_recommendations(
    user_id: int,
    request: RecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        recommendations = recommender.get_collaborative_recommendations(
            user_id,
            request.n_recommendations
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/worstrecommendations/user/{user_id}")
async def get_user_recommendations(
    user_id: int,
    request: RecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        recommendations = recommender.get_worst_collaborative_recommendations(
            user_id,
            request.n_recommendations
        )
        print("las mejores")
        print(recommendations)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/movies/genre/{genre}")
async def get_movies_by_genre(
    genre: str,
    current_user: dict = Depends(get_current_user),
    limit: int = 10,
    min_votes: int = 50  # Mínimo de votos requeridos
) -> List[Dict]:
    """Obtiene películas por género ordenadas por calificación ponderada tipo IMDb."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Get user ID from email
        cursor.execute('SELECT userId FROM users WHERE email = ?', (current_user['email'],))
        user_result = cursor.fetchone()
        if not user_result:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        user_id = user_result[0]
        
        # Get genre ID first
        cursor.execute('SELECT id FROM genres WHERE name = ?', (genre,))
        genre_result = cursor.fetchone()
        if not genre_result:
            return []
        genre_id = genre_result[0]
        
        # Calcular promedio global (C)
        cursor.execute('SELECT AVG(rating) FROM ratings')
        C = cursor.fetchone()[0]  # Promedio global de calificaciones
        
        # Obtener películas del género con sus calificaciones, excluyendo las ya vistas
        cursor.execute('''
        WITH MovieRatings AS (
            SELECT 
                m.*,
                COUNT(r.rating) as num_ratings,
                AVG(r.rating) as avg_rating
            FROM movies m
            JOIN movie_genres mg ON m.movie_id = mg.movie_id
            LEFT JOIN ratings r ON m.movie_id = r.movie_id
            WHERE mg.genre_id = ?
            AND m.movie_id NOT IN (
                SELECT movie_id 
                FROM user_film 
                WHERE user_id = ?
            )
            GROUP BY m.movie_id
        )
        SELECT *
        FROM MovieRatings
        WHERE num_ratings >= ?
        ORDER BY (num_ratings * avg_rating + ? * ?) / (num_ratings + ?) DESC
        LIMIT ?
        ''', (genre_id, user_id, min_votes, min_votes, C, min_votes, limit))
        
        movies = cursor.fetchall()
        if not movies:
            return []
            
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        movies_list = []
        for movie in movies:
            movie_dict = dict(zip(columns, movie))
            
            # Check if user has rated this movie
            cursor.execute('SELECT rating FROM ratings WHERE user_id = ? AND movie_id = ?', 
                         (user_id, movie_dict['movie_id']))
            user_rating = cursor.fetchone()
            if user_rating:
                continue  # Skip movies the user has already rated
            
            movies_list.append(movie_dict)
        
        return movies_list[:limit]  # Return limited number of movies
        
    except Exception as e:
        logger.error(f"Error al obtener películas por género: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/movies/user/preferred-genres")
async def get_movies_by_preferred_genres(
    current_user: dict = Depends(get_current_user),
    n_movies_per_genre: int = 5,
    min_ratings: int = 50  # Número mínimo de reseñas requeridas
):
    """Obtiene las mejores películas de los géneros preferidos del usuario."""
    try:
        # Obtener el usuario y sus géneros preferidos
        user = auth_handler.users_df[auth_handler.users_df['email'] == current_user['email']].iloc[0]
        user_id = int(user['userId'])
        preferred_genres = user['preferred_genres'].split('|') if 'preferred_genres' in user else []
        
        if not preferred_genres:
            raise HTTPException(status_code=404, detail="No se encontraron géneros preferidos para el usuario")
        
        # Obtener películas que el usuario ya ha visto
        user_ratings = data['ratings'][data['ratings']['userId'] == user_id]
        watched_movies = set(user_ratings['movieId'])
        
        # Calcular promedio de calificaciones para cada película
        movie_ratings = data['ratings'].groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_ratings.columns = ['movieId', 'rating_count', 'rating_mean']
        
        # Filtrar películas con menos del mínimo de reseñas requeridas
        movie_ratings = movie_ratings[movie_ratings['rating_count'] >= min_ratings]
        
        result = {}
        for genre in preferred_genres:
            # Obtener películas del género
            genre_movies = data['movies'][data['movies']['genres'].str.contains(genre, na=False)]
            
            # Filtrar películas no vistas
            unwatched_movies = genre_movies[~genre_movies['movieId'].isin(watched_movies)]
            
            # Unir con las calificaciones
            genre_result = pd.merge(unwatched_movies, movie_ratings, on='movieId')
            
            # Ordenar por calificación promedio y número de calificaciones
            genre_result = genre_result.sort_values(['rating_mean', 'rating_count'], ascending=[False, False])
            
            # Tomar las n_movies_per_genre mejores películas
            top_movies = genre_result.head(n_movies_per_genre)
            
            result[genre] = top_movies.to_dict(orient='records')
        
        return {
            "preferred_genres": preferred_genres,
            "min_ratings": min_ratings,
            "movies_by_genre": result
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/movies/top-rated")
async def get_top_rated_movies(
    current_user: dict = Depends(get_current_user),
    min_ratings: int = 50,  # Número mínimo de reseñas requeridas
    n_movies: int = 10
):
    """Obtiene las películas mejor valoradas con un mínimo de reseñas."""
    try:
        # Calcular promedio de calificaciones para cada película
        movie_ratings = data['ratings'].groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_ratings.columns = ['movieId', 'rating_count', 'rating_mean']
        
        # Filtrar películas con menos del mínimo de reseñas requeridas
        movie_ratings = movie_ratings[movie_ratings['rating_count'] >= min_ratings]
        
        # Unir con la información de las películas
        result = pd.merge(movie_ratings, data['movies'], on='movieId')
        
        # Unir con los links para obtener tmdbId
        result = pd.merge(result, data['links'][['movieId', 'tmdbId']], on='movieId', how='left')
        
        # Ordenar por calificación promedio y número de calificaciones
        result = result.sort_values(['rating_mean', 'rating_count'], ascending=[False, False])
        
        # Tomar las n_movies mejores películas
        top_movies = result.head(n_movies)
        
        # Obtener detalles de TMDB para cada película
        movie_details = []
        for _, movie in top_movies.iterrows():
            tmdb_id = int(movie['tmdbId']) if pd.notna(movie['tmdbId']) else None
            if tmdb_id:
                details = await get_tmdb_movie_details(tmdb_id)
                movie_dict = movie.to_dict()
                movie_dict.update(details)
                movie_details.append(movie_dict)
            else:
                movie_dict = movie.to_dict()
                movie_dict.update({
                    "poster_path": None,
                    "overview": None,
                    "backdrop_path": None
                })
                movie_details.append(movie_dict)
        
        return {
            "min_ratings": min_ratings,
            "total_movies": len(movie_ratings),
            "movies": movie_details
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/movies/{movie_id}/reviews")
async def get_movie_reviews(
    movie_id: int,
    current_user: dict = Depends(get_current_user),
    n_reviews: int = 5
):
    """Obtiene las últimas reseñas de una película específica."""
    try:
        # Get movie data from database
        movie_data = recommender._get_movie_data(movie_id)
        if not movie_data:
            raise HTTPException(status_code=404, detail=f"Película {movie_id} no encontrada")
        
        # Get reviews for the movie
        movie_reviews = data['ratings'][data['ratings']['movieId'] == movie_id].copy()
        
        if movie_reviews.empty:
            raise HTTPException(status_code=404, detail="No hay reseñas para esta película")
        
        # Sort by timestamp if available
        if 'timestamp' in movie_reviews.columns:
            movie_reviews = movie_reviews.sort_values('timestamp', ascending=False)
        
        # Take the latest n_reviews
        latest_reviews = movie_reviews.head(n_reviews)
        
        # Join with user information
        reviews_with_users = pd.merge(
            latest_reviews,
            data['users'][['userId', 'username', 'firstName', 'lastName']],
            on='userId',
            how='left'
        )
        
        # Generate descriptions for reviews
        reviews_with_users['description'] = reviews_with_users['rating'].apply(generate_rating_description)
        
        # Prepare response
        reviews_list = []
        for _, review in reviews_with_users.iterrows():
            review_dict = {
                "user_id": int(review['userId']),
                "username": review['username'],
                "user_name": f"{review['firstName']} {review['lastName']}".strip(),
                "rating": float(review['rating']),
                "description": review['description'],
                "timestamp": int(review['timestamp']) if 'timestamp' in review else None
            }
            reviews_list.append(review_dict)
        
        return {
            "movie_id": int(movie_id),
            "movie_title": movie_data['title'],
            "total_reviews": len(movie_reviews),
            "average_rating": float(movie_reviews['rating'].mean()),
            "latest_reviews": reviews_list
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener reseñas de la película {movie_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener reseñas de la película")

@app.post("/movies/{movie_id}/review")
async def post_movie_review(
    movie_id: int,
    review: ReviewRequest,
    current_user: dict = Depends(get_current_user)
):
    """Añade una reseña para una película específica."""
    try:
        # Print token and user information
        print("Token information:", current_user)
        print("Movie ID:", movie_id)
        print("Review data:", review.dict())
        
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Verify movie exists
        cursor.execute('SELECT movie_id FROM movies WHERE movie_id = ?', (movie_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Película {movie_id} no encontrada")
        
        # Get user ID from email
        cursor.execute('SELECT userId FROM users WHERE email = ?', (current_user['email'],))
        user_result = cursor.fetchone()
        if not user_result:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        user_id = user_result[0]
        print("User ID found:", user_id)
        
        # Verificar que el usuario ha visto la película
        cursor.execute('''
        SELECT COUNT(*) 
        FROM user_film 
        WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))
        has_watched = cursor.fetchone()[0] > 0
        
        if not has_watched:
            raise HTTPException(
                status_code=403,
                detail="No puedes reseñar una película que no has visto. Primero debes marcar la película como vista."
            )
        
        # Check if user has already reviewed this movie
        cursor.execute('SELECT rating FROM ratings WHERE user_id = ? AND movie_id = ?', 
                      (user_id, movie_id))
        if cursor.fetchone():
            raise HTTPException(
                status_code=400,
                detail="Ya has reseñado esta película. Puedes actualizar tu reseña existente."
            )
        
        # Insert new review
        timestamp = int(pd.Timestamp.now().timestamp())
        cursor.execute('''
        INSERT INTO ratings (user_id, movie_id, rating, description, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''', (user_id, movie_id, review.rating, review.description, timestamp))
        
        conn.commit()
        print("Review successfully inserted into database")
        
        return {
            "message": "Reseña añadida exitosamente",
            "review": {
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": review.rating,
                "description": review.description,
                "timestamp": timestamp
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al añadir reseña para la película {movie_id}: {str(e)}")
        print("Error details:", str(e))
        raise HTTPException(status_code=500, detail="Error al añadir reseña")
    finally:
        if 'conn' in locals():
            conn.close()

@app.put("/movies/{movie_id}/review")
async def update_movie_review(
    movie_id: int,
    review: ReviewRequest,
    current_user: dict = Depends(get_current_user)
):
    """Actualiza una reseña existente para una película específica."""
    try:
        # Verificar que la película existe
        if movie_id not in data['movies']['movieId'].values:
            raise HTTPException(status_code=404, detail="Película no encontrada")
        
        # Obtener el ID del usuario
        user = auth_handler.users_df[auth_handler.users_df['email'] == current_user['email']].iloc[0]
        user_id = int(user['userId'])
        
        # Verificar si el usuario ya ha reseñado esta película
        existing_review_idx = data['ratings'][
            (data['ratings']['userId'] == user_id) & 
            (data['ratings']['movieId'] == movie_id)
        ].index
        
        if existing_review_idx.empty:
            raise HTTPException(
                status_code=404,
                detail="No tienes una reseña existente para esta película"
            )
        
        # Actualizar la reseña
        data['ratings'].loc[existing_review_idx, 'rating'] = review.rating
        data['ratings'].loc[existing_review_idx, 'description'] = review.description
        data['ratings'].loc[existing_review_idx, 'timestamp'] = int(pd.Timestamp.now().timestamp())
        
        # Guardar el archivo actualizado
        data['ratings'].to_csv("data/ratings.csv", index=False)
        
        return {
            "message": "Reseña actualizada exitosamente",
            "review": {
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": review.rating,
                "description": review.description,
                "timestamp": int(pd.Timestamp.now().timestamp())
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

def get_movie_data(movie_id: int) -> Dict:
    """Get complete movie data from SQLite database."""
    conn = sqlite3.connect('data/tmdb_movies.db')
    cursor = conn.cursor()
    
    try:
        # Get basic movie info
        cursor.execute('''
        SELECT * FROM movies WHERE movie_id = ?
        ''', (movie_id,))
        movie = cursor.fetchone()
        
        if not movie:
            return None
            
        # Get column names
        columns = [description[0] for description in cursor.description]
        movie_dict = dict(zip(columns, movie))
        
        # Get genres
        cursor.execute('''
        SELECT g.name 
        FROM genres g
        JOIN movie_genres mg ON g.id = mg.genre_id
        WHERE mg.movie_id = ?
        ''', (movie_id,))
        movie_dict['genres'] = [row[0] for row in cursor.fetchall()]
        
        # Get production companies
        cursor.execute('''
        SELECT pc.name 
        FROM production_companies pc
        JOIN movie_production_companies mpc ON pc.id = mpc.company_id
        WHERE mpc.movie_id = ?
        ''', (movie_id,))
        movie_dict['production_companies'] = [row[0] for row in cursor.fetchall()]
        
        # Get production countries
        cursor.execute('''
        SELECT pc.name 
        FROM production_countries pc
        JOIN movie_production_countries mpc ON pc.iso_3166_1 = mpc.country_code
        WHERE mpc.movie_id = ?
        ''', (movie_id,))
        movie_dict['production_countries'] = [row[0] for row in cursor.fetchall()]
        
        # Get spoken languages
        cursor.execute('''
        SELECT sl.name 
        FROM spoken_languages sl
        JOIN movie_spoken_languages msl ON sl.iso_639_1 = msl.language_code
        WHERE msl.movie_id = ?
        ''', (movie_id,))
        movie_dict['spoken_languages'] = [row[0] for row in cursor.fetchall()]
        
        # Get collection info if exists
        cursor.execute('''
        SELECT c.name, c.poster_path, c.backdrop_path
        FROM collections c
        JOIN movie_collections mc ON c.id = mc.collection_id
        WHERE mc.movie_id = ?
        ''', (movie_id,))
        collection = cursor.fetchone()
        if collection:
            movie_dict['belongs_to_collection'] = {
                'name': collection[0],
                'poster_path': collection[1],
                'backdrop_path': collection[2]
            }
        
        return movie_dict
    finally:
        conn.close()

@app.get("/api/top-movies")
async def get_top_movies(limit: int = 10) -> List[Dict]:
    """Obtiene las películas mejor valoradas usando weighted rating al estilo IMDb."""
    try:
        # Calcular promedio global (C) y mínimo de votos requerido (m)
        C = data['ratings']['rating'].mean()
        m = 100  # mínimo de votos para que una película sea considerada

        # Agrupar por película y calcular promedio y cantidad de votos
        movie_ratings = data['ratings'].groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).reset_index()

        # Filtrar películas que cumplan con el mínimo de votos
        qualified = movie_ratings[movie_ratings['rating_count'] >= m].copy()

        # Calcular weighted rating
        qualified['weighted_rating'] = (
            (qualified['rating_count'] / (qualified['rating_count'] + m)) * qualified['avg_rating'] +
            (m / (qualified['rating_count'] + m)) * C
        )

        # Ordenar por weighted rating descendente
        top_movies = qualified.sort_values('weighted_rating', ascending=False).head(limit)

        # Obtener datos completos de cada película
        movies_data = []
        for _, row in top_movies.iterrows():
            movie_data = recommender._get_movie_data(int(row['movieId']))
            if movie_data:
                movie_data['average_rating'] = float(row['avg_rating'])
                movie_data['rating_count'] = int(row['rating_count'])
                movie_data['weighted_rating'] = float(row['weighted_rating'])
                movies_data.append(movie_data)

        return movies_data

    except Exception as e:
        logger.error(f"Error al obtener top películas: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener top películas")

@app.get("/api/movies/{movie_id}")
async def get_movie(movie_id: int) -> Dict:
    """Obtiene información detallada de una película."""
    try:
        movie_data = get_movie_data(movie_id)
        if not movie_data:
            raise HTTPException(status_code=404, detail=f"Película {movie_id} no encontrada")
        return movie_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener película {movie_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al obtener información de la película")

@app.post("/api/movies/search/{query}")
async def search_movies(query: str, limit: int = 10) -> List[Dict]:
    """Busca películas por título."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Buscar películas que coincidan con el query
        cursor.execute('''
        SELECT movie_id
        FROM movies
        WHERE LOWER(title) LIKE LOWER(?)
        ''', (f'%{query}%',))
        
        movie_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not movie_ids:
            raise HTTPException(status_code=404, detail=f"No se encontraron películas que coincidan con '{query}'")
        
        # Obtener datos completos de cada película
        movies_data = []
        for movie_id in movie_ids[:limit]:
            movie_data = get_movie_data(movie_id)
            if movie_data:
                movies_data.append(movie_data)
        
        return movies_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al buscar películas: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al buscar películas")

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_with_bot(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Chat with the movie recommendation bot."""
    try:
        response = chatbot.process_query(request.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing chat request")

@app.get("/api/movies/{movie_id}/credits")
async def get_movie_credits(movie_id: int):
    """Get cast and crew information for a specific movie."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Get movie title for reference
        cursor.execute("SELECT title FROM movies WHERE movie_id = ?", (movie_id,))
        movie = cursor.fetchone()
        if not movie:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        # Get cast (actors)
        cursor.execute("""
            SELECT p.id, p.name, p.original_name, p.profile_path, mc.character, mc.cast_order
            FROM movie_cast mc
            JOIN people p ON mc.person_id = p.id
            WHERE mc.movie_id = ?
            ORDER BY mc.cast_order
        """, (movie_id,))
        cast = [
            {
                "id": row[0],
                "name": row[1],
                "original_name": row[2],
                "profile_path": row[3],
                "character": row[4],
                "order": row[5]
            }
            for row in cursor.fetchall()
        ]
        
        # Get crew (directors and producers)
        cursor.execute("""
            SELECT p.id, p.name, p.original_name, p.profile_path, mc.job
            FROM movie_crew mc
            JOIN people p ON mc.person_id = p.id
            WHERE mc.movie_id = ? AND mc.job IN ('Director', 'Producer')
            ORDER BY mc.job
        """, (movie_id,))
        crew = [
            {
                "id": row[0],
                "name": row[1],
                "original_name": row[2],
                "profile_path": row[3],
                "job": row[4]
            }
            for row in cursor.fetchall()
        ]
        
        # Organize crew by job
        directors = [person for person in crew if person['job'] == 'Director']
        producers = [person for person in crew if person['job'] == 'Producer']
        
        return {
            "movie_id": movie_id,
            "title": movie[0],
            "cast": cast,
            "directors": directors,
            "producers": producers
        }
        
    except Exception as e:
        logger.error(f"Error getting movie credits: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving movie credits")
    finally:
        conn.close()

class UserFilmRequest(BaseModel):
    user_id: int
    movie_id: int

@app.post("/user/film")
async def add_user_film(
    request: UserFilmRequest,
    current_user: dict = Depends(get_current_user)
):
    """Registra una película vista por un usuario."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Verificar que el usuario existe y coincide con el usuario actual
        cursor.execute('SELECT userId FROM users WHERE email = ?', (current_user['email'],))
        user_result = cursor.fetchone()
        if not user_result or user_result[0] != request.user_id:
            raise HTTPException(
                status_code=403,
                detail="No tienes permiso para registrar películas para este usuario"
            )
        
        # Verificar que la película existe
        cursor.execute('SELECT movie_id FROM movies WHERE movie_id = ?', (request.movie_id,))
        if not cursor.fetchone():
            raise HTTPException(
                status_code=404,
                detail=f"Película {request.movie_id} no encontrada"
            )
        
        # Insertar el registro
        cursor.execute('''
        INSERT INTO user_film (user_id, movie_id)
        VALUES (?, ?)
        ''', (request.user_id, request.movie_id))
        conn.commit()
        
        return {
            "message": "Película registrada exitosamente",
            "user_id": request.user_id,
            "movie_id": request.movie_id
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al registrar película vista: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error al registrar película vista"
        )
    finally:
        if 'conn' in locals():
            conn.close()

# Inicializar el recomendador SVD
svd_recommender = SVDRecommender()

@app.get("/recommendations/svd/{user_id}")
async def get_svd_recommendations(
    user_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Obtiene recomendaciones de películas basadas en SVD para un usuario."""
    try:
        # Verificar que el usuario existe y coincide con el usuario actual
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        cursor.execute('SELECT userId FROM users WHERE email = ?', (current_user['email'],))
        user_result = cursor.fetchone()
        if not user_result or user_result[0] != user_id:
            raise HTTPException(
                status_code=403,
                detail="No tienes permiso para obtener recomendaciones para este usuario"
            )
        
        # Obtener recomendaciones
        recommendations = svd_recommender.get_recommendations(user_id)
        
        return {
            "user_id": user_id,
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener recomendaciones SVD: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error al obtener recomendaciones"
        )
    finally:
        if 'conn' in locals():
            conn.close()

# Inicializar el recomendador de géneros
genre_recommender = GenreRecommender()

@app.get("/recommendations/genres/{user_id}")
async def get_genre_recommendations(
    user_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Obtiene recomendaciones de géneros para un usuario."""
    try:
        # Verificar que el usuario existe y coincide con el usuario actual
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        cursor.execute('SELECT userId FROM users WHERE email = ?', (current_user['email'],))
        user_result = cursor.fetchone()
        if not user_result or user_result[0] != user_id:
            raise HTTPException(
                status_code=403,
                detail="No tienes permiso para obtener recomendaciones para este usuario"
            )
        
        # Obtener recomendaciones
        recommendations = genre_recommender.get_recommendations(user_id)
        
        return {
            "user_id": user_id,
            "recommended_genres": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener recomendaciones de géneros: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error al obtener recomendaciones de géneros"
        )
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/api/user/watch-history")
async def get_user_watch_history(
    current_user: dict = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
) -> Dict:
    """Obtiene el historial de películas vistas del usuario."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Get user ID from email
        cursor.execute('SELECT userId FROM users WHERE email = ?', (current_user['email'],))
        user_result = cursor.fetchone()
        if not user_result:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        user_id = user_result[0]
        
        # Obtener el total de películas vistas
        cursor.execute('''
        SELECT COUNT(DISTINCT movie_id) 
        FROM user_film 
        WHERE user_id = ?
        ''', (user_id,))
        total_movies = cursor.fetchone()[0]
        
        # Obtener películas vistas con detalles, ordenadas por rowid descendente
        cursor.execute('''
        WITH LatestViews AS (
            SELECT movie_id, MAX(rowid) as last_view
            FROM user_film
            WHERE user_id = ?
            GROUP BY movie_id
        )
        SELECT 
            m.*,
            GROUP_CONCAT(g.name) as genres,
            r.rating,
            r.description as review
        FROM LatestViews lv
        JOIN movies m ON lv.movie_id = m.movie_id
        LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        LEFT JOIN ratings r ON m.movie_id = r.movie_id AND r.user_id = ?
        GROUP BY m.movie_id
        ORDER BY lv.last_view DESC
        LIMIT ? OFFSET ?
        ''', (user_id, user_id, limit, offset))
        
        movies = cursor.fetchall()
        if not movies:
            return {
                "total_movies": 0,
                "current_page": offset // limit + 1,
                "total_pages": 0,
                "movies": []
            }
            
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        movies_list = []
        for movie in movies:
            movie_dict = dict(zip(columns, movie))
            
            # Convertir géneros de string a lista
            if movie_dict['genres']:
                movie_dict['genres'] = movie_dict['genres'].split(',')
            else:
                movie_dict['genres'] = []
            
            movies_list.append(movie_dict)
        
        # Calcular total de páginas
        total_pages = (total_movies + limit - 1) // limit
        
        return {
            "total_movies": total_movies,
            "current_page": offset // limit + 1,
            "total_pages": total_pages,
            "movies": movies_list
        }
        
    except Exception as e:
        logger.error(f"Error al obtener historial de películas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close() 