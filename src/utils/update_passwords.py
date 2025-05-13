import pandas as pd
from pathlib import Path
import bcrypt
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hash_password(password: str) -> str:
    """Genera un hash bcrypt de la contraseña."""
    # Convertir la contraseña a bytes
    password_bytes = password.encode('utf-8')
    # Generar el salt
    salt = bcrypt.gensalt()
    # Generar el hash
    hashed = bcrypt.hashpw(password_bytes, salt)
    # Devolver el hash como string
    return hashed.decode('utf-8')

def update_passwords():
    """Actualiza las contraseñas en el archivo users.csv."""
    try:
        # Leer el archivo users.csv
        users_file = Path("data/users.csv")
        logger.info(f"Leyendo archivo: {users_file}")
        users_df = pd.read_csv(users_file)
        
        # Verificar si la columna password existe
        if 'password' not in users_df.columns:
            logger.error("Error: La columna 'password' no existe en el archivo users.csv")
            return
        
        # Crear una copia de seguridad
        backup_file = users_file.with_suffix('.csv.bak')
        users_df.to_csv(backup_file, index=False)
        logger.info(f"Copia de seguridad creada en: {backup_file}")
        
        # Actualizar las contraseñas
        logger.info("Actualizando contraseñas...")
        users_df['password'] = users_df['password'].apply(hash_password)
        
        # Guardar el archivo actualizado
        users_df.to_csv(users_file, index=False)
        logger.info("Contraseñas actualizadas exitosamente")
        
    except Exception as e:
        logger.error(f"Error al actualizar contraseñas: {str(e)}")
        raise

if __name__ == "__main__":
    update_passwords() 