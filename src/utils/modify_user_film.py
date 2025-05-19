import sqlite3
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def modify_user_film_table():
    """Modifica la tabla user_film para permitir duplicados."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Crear una tabla temporal con la nueva estructura
        cursor.execute('''
        CREATE TABLE user_film_temp (
            user_id INTEGER,
            movie_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(userId),
            FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
        )
        ''')
        
        # Copiar los datos existentes
        cursor.execute('''
        INSERT INTO user_film_temp (user_id, movie_id)
        SELECT user_id, movie_id FROM user_film
        ''')
        
        # Eliminar la tabla original
        cursor.execute('DROP TABLE user_film')
        
        # Renombrar la tabla temporal
        cursor.execute('ALTER TABLE user_film_temp RENAME TO user_film')
        
        conn.commit()
        logger.info("Tabla user_film modificada exitosamente para permitir duplicados")
        
    except Exception as e:
        logger.error(f"Error al modificar la tabla user_film: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    modify_user_film_table() 