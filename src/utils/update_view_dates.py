import sqlite3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_view_dates():
    """Actualiza los registros con view_date nulo, vacío o formato incorrecto en la tabla user_film."""
    try:
        # Conectar a la base de datos
        conn = sqlite3.connect('data/tmdb_movies.db')
        cursor = conn.cursor()
        
        # Obtener la fecha actual en formato YYYY-MM-DD HH:MM:SS
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Actualizar registros con view_date nulo, vacío o que no contengan hora
        cursor.execute('''
        UPDATE user_film 
        SET view_date = ? 
        WHERE view_date IS NULL 
           OR view_date = ''
           OR length(view_date) = 10  -- Solo fecha sin hora (YYYY-MM-DD)
        ''', (current_date,))
        
        # Obtener el número de registros actualizados
        rows_updated = cursor.rowcount
        
        # Hacer commit de los cambios
        conn.commit()
        
        logger.info(f"Se actualizaron {rows_updated} registros con la fecha {current_date}")
        
    except Exception as e:
        logger.error(f"Error al actualizar las fechas: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    update_view_dates() 