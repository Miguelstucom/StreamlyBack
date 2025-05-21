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

class SVDRecommender:
    def __init__(self):
        self.model = None
        self.model_path = Path('models/svd_model.pkl')
        self.model_path.parent.mkdir(exist_ok=True)
        
    def prepare_data(self):
        """Prepara los datos de user_film para el entrenamiento."""
        try:
            conn = sqlite3.connect('data/tmdb_movies.db')
            
            # Obtener datos de user_film
            query = """
            SELECT user_id, movie_id, COUNT(*) as view_count
            FROM user_film
            GROUP BY user_id, movie_id
            """
            df = pd.read_sql_query(query, conn)
            
            # Crear matriz de usuario-película
            user_movie_matrix = df.pivot(
                index='user_id',
                columns='movie_id',
                values='view_count'
            ).fillna(0)
            
            # Convertir a formato largo para Surprise
            df_long = df.melt(
                id_vars=['user_id', 'movie_id'],
                value_vars=['view_count'],
                value_name='rating'
            )
            
            # Crear dataset para Surprise
            reader = Reader(rating_scale=(0, df['view_count'].max()))
            data = Dataset.load_from_df(
                df_long[['user_id', 'movie_id', 'rating']],
                reader
            )
            
            # Dividir en train y test
            trainset, testset = train_test_split(data, test_size=0.2)
            
            logger.info(f"Matriz de usuario-película:\n{user_movie_matrix}")
            logger.info(f"Dimensiones de la matriz: {user_movie_matrix.shape}")
            
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
            self.model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
            self.model.fit(trainset)
            
            # Evaluar modelo
            predictions = self.model.test(testset)
            rmse = np.sqrt(np.mean([(pred.r_ui - pred.est) ** 2 for pred in predictions]))
            logger.info(f"RMSE del modelo: {rmse}")
            
            # Guardar modelo
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info("Modelo entrenado y guardado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al entrenar modelo: {str(e)}")
            raise
    
    def load_model(self):
        """Carga el modelo guardado."""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Modelo cargado exitosamente")
            else:
                logger.info("No se encontró modelo guardado, entrenando nuevo modelo...")
                self.train()
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            raise
    
    def get_recommendations(self, user_id, n_recommendations=5):
        """Obtiene recomendaciones para un usuario."""
        try:
            if self.model is None:
                self.load_model()
            
            conn = sqlite3.connect('data/tmdb_movies.db')
            cursor = conn.cursor()
            
            # Obtener todas las películas
            cursor.execute('SELECT movie_id FROM movies')
            all_movies = [row[0] for row in cursor.fetchall()]
            
            # Obtener películas ya vistas por el usuario
            cursor.execute('''
            SELECT DISTINCT movie_id 
            FROM user_film 
            WHERE user_id = ?
            ''', (user_id,))
            seen_movies = {row[0] for row in cursor.fetchall()}
            
            # Predecir ratings para películas no vistas
            predictions = []
            for movie_id in all_movies:
                if movie_id not in seen_movies:
                    pred = self.model.predict(user_id, movie_id)
                    predictions.append((movie_id, pred.est))
            
            # Ordenar por rating predicho
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Obtener detalles de las películas recomendadas
            recommended_movies = []
            for movie_id, _ in predictions[:n_recommendations]:
                cursor.execute('''
                SELECT m.*, 
                       GROUP_CONCAT(DISTINCT g.name) as genres
                FROM movies m
                LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                WHERE m.movie_id = ?
                GROUP BY m.movie_id
                ''', (movie_id,))
                
                movie = cursor.fetchone()
                if movie:
                    movie_dict = dict(zip([col[0] for col in cursor.description], movie))
                    movie_dict['genres'] = movie_dict['genres'].split(',') if movie_dict['genres'] else []
                    recommended_movies.append(movie_dict)
            
            return recommended_movies
            
        except Exception as e:
            logger.error(f"Error al obtener recomendaciones: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

if __name__ == "__main__":
    recommender = SVDRecommender()
    recommender.train() 