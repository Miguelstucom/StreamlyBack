import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_user(user_id: int):
    """Elimina un usuario y todos sus datos relacionados."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Iniciar transacción
        cursor.execute('BEGIN TRANSACTION')
        
        # Eliminar registros de user_film
        cursor.execute('DELETE FROM user_film WHERE user_id = ?', (user_id,))
        film_records = cursor.rowcount
        logger.info(f"Eliminados {film_records} registros de user_film")
        
        # Eliminar registros de user_genres
        cursor.execute('DELETE FROM user_genres WHERE user_id = ?', (user_id,))
        genre_records = cursor.rowcount
        logger.info(f"Eliminados {genre_records} registros de user_genres")
        
        # Eliminar registros de ratings
        cursor.execute('DELETE FROM ratings WHERE user_id = ?', (user_id,))
        rating_records = cursor.rowcount
        logger.info(f"Eliminados {rating_records} registros de ratings")
        
        # Eliminar el usuario
        cursor.execute('DELETE FROM users WHERE userId = ?', (user_id,))
        user_records = cursor.rowcount
        logger.info(f"Eliminado {user_records} usuario")
        
        # Confirmar transacción
        conn.commit()
        logger.info(f"Usuario {user_id} y todos sus datos relacionados eliminados exitosamente")
        
    except Exception as e:
        logger.error(f"Error al eliminar usuario: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    delete_user(611)