import sqlite3
import random
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_user_film_table():
    """Crea la tabla user_film y la puebla con datos."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Crear tabla user_film
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_film (
            user_id INTEGER,
            movie_id INTEGER,
            PRIMARY KEY (user_id, movie_id),
            FOREIGN KEY (user_id) REFERENCES users(userId),
            FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
        )
        ''')
        
        # Obtener todos los ratings existentes
        cursor.execute('SELECT user_id, movie_id FROM ratings')
        existing_ratings = cursor.fetchall()
        
        # Insertar los ratings existentes en user_film
        cursor.executemany('''
        INSERT OR IGNORE INTO user_film (user_id, movie_id)
        VALUES (?, ?)
        ''', existing_ratings)
        
        # Obtener todos los usuarios y películas
        cursor.execute('SELECT userId FROM users')
        users = [row[0] for row in cursor.fetchall()]
        
        cursor.execute('SELECT movie_id FROM movies')
        movies = [row[0] for row in cursor.fetchall()]
        
        # Calcular cuántas películas adicionales necesitamos
        current_count = len(existing_ratings)
        target_count = 200000
        additional_needed = target_count - current_count
        
        logger.info(f"Insertando {additional_needed} registros adicionales...")
        
        # Insertar registros adicionales aleatorios
        new_records = []
        while len(new_records) < additional_needed:
            user_id = random.choice(users)
            movie_id = random.choice(movies)
            # Evitar duplicados
            if (user_id, movie_id) not in existing_ratings and (user_id, movie_id) not in new_records:
                new_records.append((user_id, movie_id))
        
        # Insertar los nuevos registros
        cursor.executemany('''
        INSERT OR IGNORE INTO user_film (user_id, movie_id)
        VALUES (?, ?)
        ''', new_records)
        
        # Verificar el conteo final
        cursor.execute('SELECT COUNT(*) FROM user_film')
        final_count = cursor.fetchone()[0]
        
        conn.commit()
        logger.info(f"Tabla user_film creada y poblada con {final_count} registros")
        
    except Exception as e:
        logger.error(f"Error al crear y poblar la tabla user_film: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    create_user_film_table() 