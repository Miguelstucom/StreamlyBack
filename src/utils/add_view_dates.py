import sqlite3
import random
from datetime import datetime, timedelta
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_view_dates():
    """Añade y rellena la columna view_date en la tabla user_film."""
    try:
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Añadir columna view_date si no existe
        cursor.execute('''
        ALTER TABLE user_film 
        ADD COLUMN view_date DATETIME
        ''')
        
        # Obtener todas las entradas de user_film
        cursor.execute('SELECT rowid FROM user_film')
        entries = cursor.fetchall()
        
        # Fechas límite
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 3, 20)  # Fecha actual
        
        # Calcular el rango total de días
        total_days = (end_date - start_date).days
        
        # Actualizar cada entrada con una fecha aleatoria
        for entry in entries:
            # Generar días aleatorios desde la fecha de inicio
            random_days = random.randint(0, total_days)
            random_date = start_date + timedelta(days=random_days)
            
            # Actualizar la entrada con la fecha aleatoria
            cursor.execute('''
            UPDATE user_film 
            SET view_date = ? 
            WHERE rowid = ?
            ''', (random_date.strftime('%Y-%m-%d %H:%M:%S'), entry[0]))
        
        conn.commit()
        logger.info("Columna view_date añadida y rellenada exitosamente")
        
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            logger.info("La columna view_date ya existe, procediendo a actualizar las fechas")
            # Si la columna ya existe, solo actualizamos las fechas
            update_view_dates(cursor, start_date, end_date)
            conn.commit()
        else:
            raise
    except Exception as e:
        logger.error(f"Error al añadir fechas de visualización: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def update_view_dates(cursor, start_date, end_date):
    """Actualiza las fechas de visualización existentes."""
    try:
        # Obtener todas las entradas de user_film
        cursor.execute('SELECT rowid FROM user_film')
        entries = cursor.fetchall()
        
        # Calcular el rango total de días
        total_days = (end_date - start_date).days
        
        # Actualizar cada entrada con una fecha aleatoria
        for entry in entries:
            # Generar días aleatorios desde la fecha de inicio
            random_days = random.randint(0, total_days)
            random_date = start_date + timedelta(days=random_days)
            
            # Actualizar la entrada con la fecha aleatoria
            cursor.execute('''
            UPDATE user_film 
            SET view_date = ? 
            WHERE rowid = ?
            ''', (random_date.strftime('%Y-%m-%d %H:%M:%S'), entry[0]))
        
        logger.info("Fechas de visualización actualizadas exitosamente")
        
    except Exception as e:
        logger.error(f"Error al actualizar fechas de visualización: {str(e)}")
        raise

if __name__ == "__main__":
    add_view_dates() 