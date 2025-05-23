import sqlite3
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_users():
    """Elimina usuarios generados anteriormente y sus registros relacionados."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Iniciar transacción
        cursor.execute('BEGIN TRANSACTION')
        
        # Obtener el último userId original
        cursor.execute("SELECT MAX(userId) FROM users WHERE userId < 611")
        last_original_user = cursor.fetchone()[0] or 0
        
        # Eliminar registros relacionados en orden
        # 1. Eliminar ratings
        cursor.execute('DELETE FROM ratings WHERE user_id > ?', (last_original_user,))
        logger.info(f"Eliminados {cursor.rowcount} registros de ratings")
        
        # 2. Eliminar user_film
        cursor.execute('DELETE FROM user_film WHERE user_id > ?', (last_original_user,))
        logger.info(f"Eliminados {cursor.rowcount} registros de user_film")
        
        # 3. Eliminar user_genres
        cursor.execute('DELETE FROM user_genres WHERE user_id > ?', (last_original_user,))
        logger.info(f"Eliminados {cursor.rowcount} registros de user_genres")
        
        # 4. Finalmente, eliminar los usuarios
        cursor.execute('DELETE FROM users WHERE userId > ?', (last_original_user,))
        logger.info(f"Eliminados {cursor.rowcount} usuarios")
        
        # Confirmar transacción
        conn.commit()
        logger.info("Limpieza de usuarios completada exitosamente")
        
    except Exception as e:
        # Si hay error, hacer rollback
        conn.rollback()
        logger.error(f"Error durante la limpieza de usuarios: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    clean_users() 