import sqlite3
import random
from datetime import datetime, timedelta
import logging
from passlib.context import CryptContext

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar el contexto de encriptación
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def generate_users():
    """Genera 600 usuarios nuevos con sus registros relacionados."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Obtener el último userId
        cursor.execute("SELECT MAX(userId) FROM users")
        last_user_id = cursor.fetchone()[0] or 0
        
        # Obtener lista de géneros
        cursor.execute("SELECT id, name FROM genres")
        genres = cursor.fetchall()
        
        # Obtener lista de películas
        cursor.execute("SELECT movie_id FROM movies")
        movies = [row[0] for row in cursor.fetchall()]
        
        # Lista de ocupaciones
        occupations = [
            "student", "engineer", "teacher", "doctor", "lawyer", "artist",
            "writer", "programmer", "designer", "manager", "scientist",
            "chef", "nurse", "architect", "accountant"
        ]
        
        # Iniciar transacción
        cursor.execute('BEGIN TRANSACTION')
        
        # Generar 600 usuarios
        for i in range(600):
            user_id = last_user_id + i + 1
            
            # Generar datos del usuario
            username = f"user{user_id}"
            firstname = f"User{user_id}"
            lastname = f"Last{user_id}"
            email = f"user{user_id}@example.com"
            password = pwd_context.hash("password123")  # Contraseña por defecto
            age = random.randint(18, 80)
            occupation = random.choice(occupations)
            
            # Seleccionar exactamente 3 géneros preferidos
            selected_genres = random.sample(genres, 3)
            preferred_genres = "|".join([genre[1] for genre in selected_genres])
            
            # Insertar usuario con preferred_genres
            cursor.execute('''
            INSERT INTO users (userId, username, firstname, lastname, email, passwordHash, age, occupation, preferred_genres)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, username, firstname, lastname, email, password, age, occupation, preferred_genres))
            
            # Insertar los 3 géneros en user_genres
            for genre_id, _ in selected_genres:
                cursor.execute('''
                INSERT INTO user_genres (user_id, genre_id)
                VALUES (?, ?)
                ''', (user_id, genre_id))
            
            # Generar registros de películas vistas (1-80 por usuario)
            num_movies = random.randint(1, 80)
            selected_movies = random.sample(movies, num_movies)
            
            # Fechas límite para las visualizaciones
            start_date = datetime(2018, 1, 1)
            end_date = datetime(2024, 3, 20)
            total_days = (end_date - start_date).days
            
            for movie_id in selected_movies:
                # Generar fecha aleatoria
                random_days = random.randint(0, total_days)
                view_date = start_date + timedelta(days=random_days)
                
                # Insertar registro de visualización
                cursor.execute('''
                INSERT INTO user_film (user_id, movie_id, view_date)
                VALUES (?, ?, ?)
                ''', (user_id, movie_id, view_date.strftime('%Y-%m-%d %H:%M:%S')))
            
            # Generar calificaciones (1-10 por usuario, pero no más que películas vistas)
            num_ratings = min(random.randint(1, 10), len(selected_movies))
            rated_movies = random.sample(selected_movies, num_ratings)
            
            for movie_id in rated_movies:
                # Generar calificación aleatoria (0.5 a 5.0 en incrementos de 0.5)
                rating = round(random.uniform(0.5, 5.0) * 2) / 2
                description = f"Review for movie {movie_id} by user {user_id}"
                timestamp = int(datetime.now().timestamp())
                
                cursor.execute('''
                INSERT INTO ratings (user_id, movie_id, rating, description, timestamp)
                VALUES (?, ?, ?, ?, ?)
                ''', (user_id, movie_id, rating, description, timestamp))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generados {i + 1} usuarios")
        
        # Confirmar transacción
        conn.commit()
        logger.info("Generación de usuarios completada exitosamente")
        
    except Exception as e:
        # Si hay error, hacer rollback
        conn.rollback()
        logger.error(f"Error durante la generación de usuarios: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    generate_users() 