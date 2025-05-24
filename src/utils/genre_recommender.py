import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pickle
import logging
import sqlite3
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import math

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
        self.svd = None
        self.user_genre_matrix = None
        self.genre_names = None
        self.n_components = None
        self.explained_variance_ratio = None
        
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
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> Dict:
        """Calcula todas las métricas de evaluación."""
        # MSE y RMSE
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        
        # Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K
        precision_k = 0
        recall_k = 0
        ndcg_k = 0
        map_k = 0
        hit_rate_k = 0
        
        n_users = len(y_true)
        for i in range(n_users):
            # Obtener top-K predicciones y verdaderos valores
            pred_top_k = np.argsort(y_pred[i])[-k:]
            true_top_k = np.argsort(y_true[i])[-k:]
            
            # Precision@K
            hits = len(set(pred_top_k) & set(true_top_k))
            precision_k += hits / k
            
            # Recall@K
            total_relevant = len(true_top_k)
            recall_k += hits / total_relevant if total_relevant > 0 else 0
            
            # NDCG@K
            dcg = 0
            idcg = 0
            for j, item in enumerate(pred_top_k):
                if item in true_top_k:
                    dcg += 1 / math.log2(j + 2)
            for j in range(min(k, total_relevant)):
                idcg += 1 / math.log2(j + 2)
            ndcg_k += dcg / idcg if idcg > 0 else 0
            
            # MAP@K
            ap = 0
            hits = 0
            for j, item in enumerate(pred_top_k):
                if item in true_top_k:
                    hits += 1
                    ap += hits / (j + 1)
            map_k += ap / min(k, total_relevant) if total_relevant > 0 else 0
            
            # Hit Rate@K
            hit_rate_k += 1 if len(set(pred_top_k) & set(true_top_k)) > 0 else 0
        
        # Promediar métricas
        n_users = float(n_users)
        return {
            'MSE': mse,
            'RMSE': rmse,
            f'Precision@{k}': precision_k / n_users,
            f'Recall@{k}': recall_k / n_users,
            f'NDCG@{k}': ndcg_k / n_users,
            f'MAP@{k}': map_k / n_users,
            f'HitRate@{k}': hit_rate_k / n_users
        }
    
    def _plot_variance_analysis(self, max_components: int = 19):
        """Analiza y grafica la varianza explicada por los componentes."""
        # Calcular varianza explicada para diferentes números de componentes
        variances = []
        components_range = range(1, max_components + 1)
        
        for n in components_range:
            svd = TruncatedSVD(n_components=n, random_state=42)
            svd.fit(self.user_genre_matrix)
            variances.append(svd.explained_variance_ratio_.sum())
            logger.info(f"Componentes {n}: {variances[-1]*100:.2f}% de varianza explicada")
        
        # Encontrar el número de componentes para 95% de varianza
        target_variance = 0.95
        n_components_95 = next(i for i, v in enumerate(variances, 1) if v >= target_variance)
        
        # Calcular ahorro de dimensionalidad
        original_dim = self.user_genre_matrix.shape[1]
        reduction = ((original_dim - n_components_95) / original_dim) * 100
        
        logger.info(f"\nResumen de reducción de dimensionalidad:")
        logger.info(f"- Dimensión original: {original_dim}")
        logger.info(f"- Componentes necesarios para 95% varianza: {n_components_95}")
        logger.info(f"- Reducción de dimensionalidad: {reduction:.2f}%")
        
        # Crear gráfica
        plt.figure(figsize=(10, 6))
        plt.plot(components_range, variances, 'bo-')
        plt.axhline(y=target_variance, color='r', linestyle='--', label='95% varianza')
        plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} componentes')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.title('Análisis de Varianza Explicada por TruncatedSVD')
        plt.grid(True)
        plt.legend()
        
        # Guardar gráfica
        plt.savefig('variance_analysis.png')
        plt.close()
        
        return n_components_95
    
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
            
            # Obtener matriz de usuario-género
            conn = sqlite3.connect('data/tmdb_movies.db')
            cursor = conn.cursor()
            
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
            self.user_genre_matrix = df.pivot(
                index='user_id',
                columns='genre_id',
                values='normalized_rating'
            ).fillna(0)
            
            # Obtener nombres de géneros
            cursor.execute('SELECT id, name FROM genres ORDER BY id')
            genres = cursor.fetchall()
            self.genre_names = [genre[1] for genre in genres]
            
            # Entrenar modelo con TruncatedSVD
            n_components = self._plot_variance_analysis()
            logger.info(f"Número de componentes seleccionados para explicar 95% de varianza: {n_components}")
            
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd.fit(self.user_genre_matrix)
            self.explained_variance_ratio = self.svd.explained_variance_ratio_
            
            # Mostrar varianza explicada por cada componente
            for i, var in enumerate(self.explained_variance_ratio):
                logger.info(f"Componente {i+1}: {var:.4f} ({var*100:.2f}%)")
            
            # Calcular métricas
            train_pred = self.svd.inverse_transform(self.svd.transform(self.user_genre_matrix))
            
            # Dividir en train y test para métricas
            train_size = int(len(self.user_genre_matrix) * 0.8)
            train_matrix = self.user_genre_matrix.iloc[:train_size]
            test_matrix = self.user_genre_matrix.iloc[train_size:]
            
            train_metrics = self._calculate_metrics(train_matrix.values, train_pred[:train_size])
            test_metrics = self._calculate_metrics(test_matrix.values, train_pred[train_size:])
            
            # Logging de métricas
            logger.info("Métricas de entrenamiento:")
            for metric, value in train_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            logger.info("\nMétricas de prueba:")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return train_metrics, test_metrics
            
        except Exception as e:
            logger.error(f"Error al entrenar modelo: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
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
            
            # Obtener géneros vistos por el usuario
            cursor.execute('''
            SELECT g.id, COUNT(*) as view_count
            FROM genres g
            JOIN movie_genres mg ON g.id = mg.genre_id
            JOIN user_film uf ON mg.movie_id = uf.movie_id
            WHERE uf.user_id = ?
            GROUP BY g.id
            ''', (user_id,))
            seen_genres = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Obtener géneros preferidos del usuario
            cursor.execute('''
            SELECT g.id
            FROM genres g
            JOIN user_genres ug ON g.id = ug.genre_id
            WHERE ug.user_id = ?
            ''', (user_id,))
            preferred_genres = {row[0] for row in cursor.fetchall()}
            
            logger.info(f"Géneros vistos por el usuario: {len(seen_genres)}")
            logger.info(f"Géneros preferidos del usuario: {len(preferred_genres)}")
            
            # Predecir ratings para todos los géneros
            predictions = []
            for genre_id in all_genres:
                try:
                    # Convertir ID original a ID normalizado
                    normalized_id = self.genre_mapping[genre_id]
                    pred = self.model.predict(user_id, normalized_id)
                    
                    # Ajustar el score basado en si ya ha visto el género o es preferido
                    base_score = max(0, min(10, pred.est))
                    if genre_id in seen_genres or genre_id in preferred_genres:
                        # Reducir significativamente el score para géneros ya vistos o preferidos
                        view_count = seen_genres.get(genre_id, 0)
                        adjusted_score = base_score * (1 / (1 + 0.5 * (view_count + 1)))  # Penalización más fuerte
                    else:
                        # Aumentar el score para géneros no vistos ni preferidos
                        adjusted_score = base_score * 1.5  # Mayor bonus para géneros nuevos
                    
                    predictions.append((genre_id, adjusted_score))
                    logger.info(f"Predicción para género {genre_id} ({all_genres[genre_id]}): {adjusted_score:.2f}")
                except Exception as e:
                    logger.error(f"Error al predecir para género {genre_id}: {str(e)}")
                    continue
            
            logger.info(f"Total de predicciones generadas: {len(predictions)}")
            
            # Ordenar por rating predicho
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Obtener los géneros recomendados, saltando los ya vistos o preferidos
            recommended_genres = []
            for genre_id, score in predictions:
                if len(recommended_genres) >= n_recommendations:
                    break
                    
                if genre_id not in seen_genres and genre_id not in preferred_genres:  # Solo añadir géneros nuevos
                    recommended_genres.append({
                        'genre_id': genre_id,
                        'name': all_genres[genre_id],
                        'predicted_score': float(score),
                        'already_seen': False,
                        'is_preferred': False,
                        'view_count': 0
                    })
            
            logger.info(f"Géneros recomendados: {recommended_genres}")
            
            if not recommended_genres:
                logger.warning("No se encontraron recomendaciones de géneros nuevos")
                # Si no hay recomendaciones nuevas, devolver los géneros más populares no vistos ni preferidos
                cursor.execute('''
                SELECT g.id, g.name, COUNT(*) as view_count
                FROM genres g
                JOIN movie_genres mg ON g.id = mg.genre_id
                JOIN user_film uf ON mg.movie_id = uf.movie_id
                WHERE g.id NOT IN (
                    SELECT DISTINCT g2.id
                    FROM genres g2
                    JOIN movie_genres mg2 ON g2.id = mg2.genre_id
                    JOIN user_film uf2 ON mg2.movie_id = uf2.movie_id
                    WHERE uf2.user_id = ?
                )
                AND g.id NOT IN (
                    SELECT genre_id
                    FROM user_genres
                    WHERE user_id = ?
                )
                GROUP BY g.id
                ORDER BY view_count DESC
                LIMIT ?
                ''', (user_id, user_id, n_recommendations))
                popular_genres = cursor.fetchall()
                recommended_genres = [
                    {
                        'genre_id': row[0],
                        'name': row[1],
                        'predicted_score': 5.0,  # Score neutral para géneros populares
                        'already_seen': False,
                        'is_preferred': False,
                        'view_count': row[2]
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