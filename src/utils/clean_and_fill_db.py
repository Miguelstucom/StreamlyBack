import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_and_fill_db():
    """Limpia las tablas de usuarios y rellena con datos de los archivos CSV."""
    try:
        # Conectar a la base de datos
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Limpiar tablas
        logger.info("Limpiando tablas...")
        tables = ['ratings', 'user_film', 'user_genres', 'users']
        for table in tables:
            cursor.execute(f'DELETE FROM {table}')
            logger.info(f"Tabla {table} limpiada")
        
        # Cargar datos de users.csv
        logger.info("Cargando datos de users.csv...")
        users_df = pd.read_csv('csvcopia/users.csv')
        users_df.to_sql('users', conn, if_exists='append', index=False)
        logger.info(f"Se cargaron {len(users_df)} usuarios")
        
        # Cargar datos de ratings.csv
        logger.info("Cargando datos de ratings.csv...")
        ratings_df = pd.read_csv('csvcopia/ratings.csv')
        ratings_df.to_sql('ratings', conn, if_exists='append', index=False)
        logger.info(f"Se cargaron {len(ratings_df)} ratings")
        
        # Cargar datos de user_film.csv
        logger.info("Cargando datos de user_film.csv...")
        user_film_df = pd.read_csv('csvcopia/user_film.csv')
        user_film_df.to_sql('user_film', conn, if_exists='append', index=False)
        logger.info(f"Se cargaron {len(user_film_df)} registros de user_film")
        
        # Cargar datos de user_genres.csv
        logger.info("Cargando datos de user_genres.csv...")
        user_genres_df = pd.read_csv('csvcopia/user_genres.csv')
        user_genres_df.to_sql('user_genres', conn, if_exists='append', index=False)
        logger.info(f"Se cargaron {len(user_genres_df)} registros de user_genres")
        
        # Hacer commit de los cambios
        conn.commit()
        logger.info("Base de datos actualizada exitosamente")
        
    except Exception as e:
        logger.error(f"Error al actualizar la base de datos: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    clean_and_fill_db() 