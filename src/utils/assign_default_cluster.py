import sqlite3
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def assign_default_cluster():
    """Asigna el cluster 0 a todos los usuarios que no tienen cluster asignado."""
    try:
        # Conectar a la base de datos
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()

        # Encontrar usuarios sin cluster
        cursor.execute('''
        SELECT user_id 
        FROM users
        LEFT JOIN user_cluster uc ON users.userId = uc.user_id 
        WHERE uc.user_id IS NULL
        GROUP BY user_id
        ''')
        
        users_without_cluster = cursor.fetchall()
        n_users = len(users_without_cluster)
        
        if n_users == 0:
            logger.info("No hay usuarios sin cluster asignado.")
            return

        logger.info(f"Encontrados {n_users} usuarios sin cluster asignado.")

        # Insertar estos usuarios en user_cluster con cluster = 0
        cursor.executemany('''
        INSERT INTO user_cluster (user_id, cluster)
        VALUES (?, 0)
        ''', users_without_cluster)

        # Confirmar cambios
        conn.commit()
        
        logger.info(f"Cluster 0 asignado exitosamente a {n_users} usuarios.")

    except Exception as e:
        logger.error(f"Error al asignar cluster por defecto: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    assign_default_cluster() 