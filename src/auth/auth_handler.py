from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, status
import pandas as pd
from pathlib import Path
import logging
import bcrypt

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de seguridad
SECRET_KEY = "tu_clave_secreta_muy_segura"  # En producción, usar una clave segura y almacenarla en variables de entorno
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30000

class AuthHandler:
    def __init__(self, users_file: str = "data/users.csv"):
        self.users_file = Path(users_file)
        self.users_df = pd.read_csv(self.users_file)
        logger.info(f"Usuarios cargados: {len(self.users_df)}")
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifica si la contraseña coincide con el hash o es igual."""
        try:
            # Convertir la contraseña a bytes
            password_bytes = plain_password.encode('utf-8')
            # Convertir el hash a bytes
            hashed_bytes = hashed_password.encode('utf-8')
            # Verificar la contraseña
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception as e:
            # Si falla la verificación bcrypt, comparar directamente
            logger.warning(f"Error al verificar hash: {str(e)}")
            return plain_password == hashed_password
    
    def get_password_hash(self, password: str) -> str:
        """Genera un hash de la contraseña."""
        # Convertir la contraseña a bytes
        password_bytes = password.encode('utf-8')
        # Generar el salt
        salt = bcrypt.gensalt()
        # Generar el hash
        hashed = bcrypt.hashpw(password_bytes, salt)
        # Devolver el hash como string
        return hashed.decode('utf-8')
    
    def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Autentica un usuario con email y contraseña."""
        logger.info(f"Intentando autenticar usuario: {email}")
        user = self.users_df[self.users_df['email'] == email]
        if user.empty:
            logger.warning(f"Usuario no encontrado: {email}")
            return None
        
        user_data = user.iloc[0].to_dict()
        if not self.verify_password(password, user_data['password']):
            logger.warning(f"Contraseña incorrecta para usuario: {email}")
            return None
            
        logger.info(f"Usuario autenticado exitosamente: {email}")
        return user_data
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Crea un token JWT."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verifica un token JWT."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido o expirado",
                headers={"WWW-Authenticate": "Bearer"},
            ) 