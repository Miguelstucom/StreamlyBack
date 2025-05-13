import pandas as pd
import random
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Diccionario de descripciones por rating
RATING_DESCRIPTIONS = {
    0.5: [
        "Una experiencia realmente decepcionante",
        "No recomendaría esta película a nadie",
        "Una pérdida total de tiempo",
        "De las peores películas que he visto"
    ],
    1.0: [
        "Muy mala película",
        "No cumplió con mis expectativas",
        "Me arrepiento de haberla visto",
        "No la recomiendo"
    ],
    1.5: [
        "Película bastante mala",
        "No me gustó nada",
        "Me decepcionó bastante",
        "No la volvería a ver"
    ],
    2.0: [
        "Película mediocre",
        "No fue lo que esperaba",
        "Me aburrió bastante",
        "No la recomendaría"
    ],
    2.5: [
        "Regular tirando a mala",
        "No me convenció",
        "Me quedé con ganas de más",
        "No destacó en nada"
    ],
    3.0: [
        "Película regular",
        "Pasable pero nada especial",
        "Entretenida pero olvidable",
        "Ni buena ni mala"
    ],
    3.5: [
        "Bastante buena",
        "Me gustó bastante",
        "Entretenida y bien hecha",
        "Vale la pena verla"
    ],
    4.0: [
        "Muy buena película",
        "Me gustó mucho",
        "La recomendaría",
        "Bien hecha y entretenida"
    ],
    4.5: [
        "Excelente película",
        "Me encantó",
        "Altamente recomendable",
        "Una experiencia muy disfrutable"
    ],
    5.0: [
        "Una obra maestra",
        "De las mejores películas que he visto",
        "Absolutamente imprescionante",
        "Una experiencia inolvidable"
    ]
}

def generate_rating_description(rating: float) -> str:
    """Genera una descripción aleatoria para una calificación."""
    # Redondear al 0.5 más cercano
    rounded_rating = round(rating * 2) / 2
    # Si la calificación está fuera de rango, usar el límite más cercano
    rounded_rating = max(0.5, min(5.0, rounded_rating))
    # Seleccionar una descripción aleatoria
    return random.choice(RATING_DESCRIPTIONS[rounded_rating])

def add_descriptions_to_ratings():
    """Añade descripciones a las calificaciones en el archivo ratings.csv."""
    try:
        # Leer el archivo ratings.csv
        data_dir = Path("data")
        ratings_file = data_dir / "ratings.csv"
        logger.info(f"Leyendo archivo de calificaciones: {ratings_file}")
        
        ratings_df = pd.read_csv(ratings_file)
        
        # Crear una copia de seguridad
        backup_file = ratings_file.with_suffix('.csv.bak')
        ratings_df.to_csv(backup_file, index=False)
        logger.info(f"Copia de seguridad creada en: {backup_file}")
        
        # Generar descripciones
        logger.info("Generando descripciones para las calificaciones...")
        ratings_df['description'] = ratings_df['rating'].apply(generate_rating_description)
        
        # Guardar el archivo actualizado
        ratings_df.to_csv(ratings_file, index=False)
        logger.info("Archivo ratings.csv actualizado con descripciones")
        
        # Mostrar algunas estadísticas
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        logger.info("\nDistribución de calificaciones:")
        for rating, count in rating_counts.items():
            logger.info(f"Rating {rating}: {count} calificaciones")
            
    except Exception as e:
        logger.error(f"Error al generar descripciones: {str(e)}")
        raise

if __name__ == "__main__":
    add_descriptions_to_ratings() 