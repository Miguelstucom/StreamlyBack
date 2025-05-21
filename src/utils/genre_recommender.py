import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pickle
import logging
import sqlite3
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenreRecommender:
    def __init__(self):
        self.model = None
        self.model_path = Path('models/genre_svd_model.pkl')
        self.model_path.parent.mkdir(exist_ok=True)
        self.genre_mapping = {}  # Mapeo de IDs originales a IDs normalizados
        self.reverse_mapping = {}  # Mapeo inverso
        
    def prepare_data(self):
        """Prepara los datos de user_film y movie_genres para el entrenamiento."""
        try:
            conn = sqlite3.connect('data/tmdb_movies.db')
            
            # Obtener todos los géneros y crear mapeo
            cursor = conn.cursor()
            cursor.execute('SELECT id, name FROM genres ORDER BY id')
            genres = cursor.fetchall()
            
            # Crear mapeo de IDs originales a IDs normalizados (0 a n-1)
            self.genre_mapping = {genre[0]: idx for idx, genre in enumerate(genres)}
            self.reverse_mapping = {idx: genre[0] for idx, genre in enumerate(genres)}
            
            # Obtener datos de user_film y movie_genres
            query = """
            SELECT uf.user_id, g.id as genre_id, COUNT(*) as view_count
            FROM user_film uf
            JOIN movie_genres mg ON uf.movie_id = mg.movie_id
            JOIN genres g ON mg.genre_id = g.id
            GROUP BY uf.user_id, g.id
            """
            df = pd.read_sql_query(query, conn)
            
            # Normalizar los IDs de géneros
            df['genre_id'] = df['genre_id'].map(self.genre_mapping)
            
            # Normalizar view_count a escala 0-10
            max_views = df['view_count'].max()
            df['normalized_rating'] = (df['view_count'] / max_views) * 10
            
            # Crear matriz de usuario-género
            user_genre_matrix = df.pivot(
                index='user_id',
                columns='genre_id',
                values='normalized_rating'
            ).fillna(0)
            
            # Convertir a formato largo para Surprise
            df_long = df.melt(
                id_vars=['user_id', 'genre_id'],
                value_vars=['normalized_rating'],
                value_name='rating'
            )
            
            # Crear dataset para Surprise
            reader = Reader(rating_scale=(0, 10))
            data = Dataset.load_from_df(
                df_long[['user_id', 'genre_id', 'rating']],
                reader
            )
            
            # Dividir en train y test
            trainset, testset = train_test_split(data, test_size=0.2)
            
            logger.info(f"Matriz de usuario-género:\n{user_genre_matrix}")
            logger.info(f"Dimensiones de la matriz: {user_genre_matrix.shape}")
            
            return trainset, testset
            
        except Exception as e:
            logger.error(f"Error al preparar datos: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def train(self):
        """Entrena el modelo SVD."""
        try:
            trainset, testset = self.prepare_data()
            
            # Crear y entrenar modelo
            self.model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
            self.model.fit(trainset)
            
            # Evaluar modelo
            predictions = self.model.test(testset)
            rmse = np.sqrt(np.mean([(pred.r_ui - pred.est) ** 2 for pred in predictions]))
            logger.info(f"RMSE del modelo: {rmse}")
            
            # Guardar modelo y mapeos
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'genre_mapping': self.genre_mapping,
                    'reverse_mapping': self.reverse_mapping
                }, f)
            
            logger.info("Modelo entrenado y guardado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al entrenar modelo: {str(e)}")
            raise
    
    def load_model(self):
        """Carga el modelo guardado."""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.genre_mapping = saved_data['genre_mapping']
                    self.reverse_mapping = saved_data['reverse_mapping']
                logger.info("Modelo cargado exitosamente")
            else:
                logger.info("No se encontró modelo guardado, entrenando nuevo modelo...")
                self.train()
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            raise
    
    def get_recommendations(self, user_id, n_recommendations=3):
        """Obtiene recomendaciones de géneros para un usuario."""
        try:
            if self.model is None:
                self.load_model()
            
            conn = sqlite3.connect('data/tmdb_movies.db')
            cursor = conn.cursor()
            
            # Obtener todos los géneros
            cursor.execute('SELECT id, name FROM genres')
            all_genres = {row[0]: row[1] for row in cursor.fetchall()}
            logger.info(f"Total de géneros disponibles: {len(all_genres)}")
            
            # Obtener géneros vistos por el usuario con su conteo
            cursor.execute('''
            SELECT g.id, COUNT(*) as view_count
            FROM genres g
            JOIN movie_genres mg ON g.id = mg.genre_id
            JOIN user_film uf ON mg.movie_id = uf.movie_id
            WHERE uf.user_id = ?
            GROUP BY g.id
            ''', (user_id,))
            seen_genres = {row[0]: row[1] for row in cursor.fetchall()}
            logger.info(f"Géneros vistos por el usuario: {len(seen_genres)}")
            
            # Predecir ratings para todos los géneros
            predictions = []
            for genre_id in all_genres:
                try:
                    # Convertir ID original a ID normalizado
                    normalized_id = self.genre_mapping[genre_id]
                    pred = self.model.predict(user_id, normalized_id)
                    
                    # Ajustar el score basado en si ya ha visto el género
                    base_score = max(0, min(10, pred.est))
                    if genre_id in seen_genres:
                        # Reducir el score para géneros ya vistos
                        view_count = seen_genres[genre_id]
                        adjusted_score = base_score * (1 / (1 + 0.1 * view_count))
                    else:
                        # Aumentar el score para géneros no vistos
                        adjusted_score = base_score * 1.2
                    
                    predictions.append((genre_id, adjusted_score))
                    logger.info(f"Predicción para género {genre_id} ({all_genres[genre_id]}): {adjusted_score:.2f}")
                except Exception as e:
                    logger.error(f"Error al predecir para género {genre_id}: {str(e)}")
                    continue
            
            logger.info(f"Total de predicciones generadas: {len(predictions)}")
            
            # Ordenar por rating predicho
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Obtener los géneros recomendados
            recommended_genres = []
            for genre_id, score in predictions[:n_recommendations]:
                recommended_genres.append({
                    'genre_id': genre_id,
                    'name': all_genres[genre_id],
                    'predicted_score': float(score),
                    'already_seen': genre_id in seen_genres,
                    'view_count': seen_genres.get(genre_id, 0)
                })
            
            logger.info(f"Géneros recomendados: {recommended_genres}")
            
            if not recommended_genres:
                logger.warning("No se encontraron recomendaciones de géneros")
                # Si no hay recomendaciones, devolver los géneros más populares
                cursor.execute('''
                SELECT g.id, g.name, COUNT(*) as view_count
                FROM genres g
                JOIN movie_genres mg ON g.id = mg.genre_id
                JOIN user_film uf ON mg.movie_id = uf.movie_id
                GROUP BY g.id
                ORDER BY view_count DESC
                LIMIT ?
                ''', (n_recommendations,))
                popular_genres = cursor.fetchall()
                recommended_genres = [
                    {
                        'genre_id': row[0],
                        'name': row[1],
                        'predicted_score': 5.0,  # Score neutral para géneros populares
                        'already_seen': row[0] in seen_genres,
                        'view_count': seen_genres.get(row[0], 0)
                    }
                    for row in popular_genres
                ]
            
            return recommended_genres
            
        except Exception as e:
            logger.error(f"Error al obtener recomendaciones de géneros: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

if __name__ == "__main__":
    recommender = GenreRecommender()
    recommender.train() 