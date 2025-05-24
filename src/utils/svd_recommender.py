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

class SVDRecommender:
    def __init__(self):
        self.model = None
        self.model_path = Path('models/svd_model.pkl')
        self.model_path.parent.mkdir(exist_ok=True)
        self.svd = None
        self.user_movie_matrix = None
        self.n_components = None
        self.explained_variance_ratio = None
        
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
    
    def _plot_variance_analysis(self, max_components: int = 500):
        """Analiza y grafica la varianza explicada por los componentes."""
        # Calcular varianza explicada para diferentes números de componentes
        variances = []
        components_range = []
        current_components = 1
        target_variance = 0.95
        
        while True:
            svd = TruncatedSVD(n_components=current_components, random_state=42)
            svd.fit(self.user_movie_matrix)
            current_variance = svd.explained_variance_ratio_.sum()
            variances.append(current_variance)
            components_range.append(current_components)
            
            logger.info(f"Componentes {current_components}: {current_variance*100:.2f}% de varianza explicada")
            
            if current_variance >= target_variance or current_components >= max_components:
                break
                
            # Incrementar el número de componentes
            if current_components < 50:
                current_components += 1
            elif current_components < 100:
                current_components += 5
            else:
                current_components += 10
        
        # Encontrar el número de componentes para diferentes niveles de varianza
        target_variances = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
        n_components_target = {}
        
        for target in target_variances:
            try:
                n_components = next(i for i, v in enumerate(variances, 1) if v >= target)
                n_components_target[target] = components_range[n_components-1]
            except StopIteration:
                n_components_target[target] = None
        
        # Si no alcanzamos ningún nivel objetivo, usar el mejor nivel alcanzado
        if not any(n_components_target.values()):
            best_variance = max(variances)
            best_components = components_range[variances.index(best_variance)]
            logger.warning(f"No se alcanzó ningún nivel objetivo de varianza. Mejor nivel alcanzado: {best_variance*100:.2f}% con {best_components} componentes")
            n_components_best = best_components
            best_target = best_variance
        else:
            best_target = max(v for v in target_variances if n_components_target[v] is not None)
            n_components_best = n_components_target[best_target]
        
        # Calcular ahorro de dimensionalidad
        original_dim = self.user_movie_matrix.shape[1]
        reduction = ((original_dim - n_components_best) / original_dim) * 100
        
        logger.info(f"\nResumen de reducción de dimensionalidad:")
        logger.info(f"- Dimensión original: {original_dim}")
        logger.info(f"- Componentes necesarios para {best_target*100:.2f}% varianza: {n_components_best}")
        logger.info(f"- Reducción de dimensionalidad: {reduction:.2f}%")
        
        # Crear gráfica
        plt.figure(figsize=(10, 6))
        plt.plot(components_range, variances, 'bo-')
        
        # Añadir líneas para cada nivel de varianza alcanzado
        for target in target_variances:
            if n_components_target[target] is not None:
                plt.axhline(y=target, color='r', linestyle='--', 
                          label=f'{target*100}% varianza ({n_components_target[target]} componentes)')
        
        # Añadir línea para el mejor nivel alcanzado si no alcanzamos ningún objetivo
        if not any(n_components_target.values()):
            plt.axhline(y=best_variance, color='g', linestyle='--',
                       label=f'Mejor nivel: {best_variance*100:.2f}% ({best_components} componentes)')
        
        plt.xlabel('Número de Componentes')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.title('Análisis de Varianza Explicada por TruncatedSVD')
        plt.grid(True)
        plt.legend()
        
        # Guardar gráfica
        plt.savefig('svd_variance_analysis.png')
        plt.close()
        
        return n_components_best
    
    def prepare_data(self):
        """Prepara los datos para el entrenamiento."""
        try:
            conn = sqlite3.connect('data/tmdb_movies.db')
            
            # Obtener datos de ratings
            query = """
            SELECT user_id, movie_id, rating
            FROM ratings
            """
            df = pd.read_sql_query(query, conn)
            
            # Crear matriz de usuario-película
            self.user_movie_matrix = df.pivot(
                index='user_id',
                columns='movie_id',
                values='rating'
            ).fillna(0)
            
            # No necesitamos hacer melt ya que los datos ya están en el formato correcto
            # para Surprise (user_id, movie_id, rating)
            
            # Crear dataset para Surprise
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                df[['user_id', 'movie_id', 'rating']],
                reader
            )
            
            # Dividir en train y test
            trainset, testset = train_test_split(data, test_size=0.2)
            
            logger.info(f"Matriz de usuario-película:\n{self.user_movie_matrix}")
            logger.info(f"Dimensiones de la matriz: {self.user_movie_matrix.shape}")
            
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
            
            # Guardar modelo
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info("Modelo entrenado y guardado exitosamente")
            
            # Entrenar modelo con TruncatedSVD
            n_components = self._plot_variance_analysis()
            logger.info(f"Número de componentes seleccionados para explicar 95% de varianza: {n_components}")
            
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd.fit(self.user_movie_matrix)
            self.explained_variance_ratio = self.svd.explained_variance_ratio_
            
            # Mostrar varianza explicada por cada componente
            for i, var in enumerate(self.explained_variance_ratio):
                logger.info(f"Componente {i+1}: {var:.4f} ({var*100:.2f}%)")
            
            # Calcular métricas
            train_pred = self.svd.inverse_transform(self.svd.transform(self.user_movie_matrix))
            
            # Dividir en train y test para métricas
            train_size = int(len(self.user_movie_matrix) * 0.8)
            train_matrix = self.user_movie_matrix.iloc[:train_size]
            test_matrix = self.user_movie_matrix.iloc[train_size:]
            
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
    
    def get_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Obtiene recomendaciones de películas para un usuario."""
        try:
            if self.model is None:
                self.load_model()
            
            conn = sqlite3.connect('data/tmdb_movies.db')
            cursor = conn.cursor()
            
            # Obtener películas ya vistas por el usuario
            cursor.execute('''
            SELECT movie_id
            FROM user_film
            WHERE user_id = ?
            ''', (user_id,))
            seen_movies = {row[0] for row in cursor.fetchall()}
            
            # Obtener todas las películas
            cursor.execute('SELECT movie_id FROM movies')
            all_movies = {row[0] for row in cursor.fetchall()}
            
            # Predecir ratings para todas las películas no vistas
            predictions = []
            for movie_id in all_movies - seen_movies:
                try:
                    pred = self.model.predict(user_id, movie_id)
                    predictions.append((movie_id, pred.est))
                except Exception as e:
                    logger.error(f"Error al predecir para película {movie_id}: {str(e)}")
                    continue
            
            # Ordenar por rating predicho
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Obtener las películas recomendadas
            recommended_movies = []
            for movie_id, score in predictions[:n_recommendations]:
                # Obtener detalles de la película
                cursor.execute('''
                SELECT m.movie_id, m.title, m.release_date, m.poster_path, m.overview, m.vote_average, m.vote_count,
                       GROUP_CONCAT(g.name) as genres
                FROM movies m
                LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                WHERE m.movie_id = ?
                GROUP BY m.movie_id
                ''', (movie_id,))
                movie = cursor.fetchone()
                
                if movie:
                    # Convertir géneros de string a lista
                    genres = movie[-1].split(',') if movie[-1] else []
                    
                    # Extraer el año de la fecha de lanzamiento
                    release_date = movie[2]
                    year = release_date.split('-')[0] if release_date else None
                    
                    recommended_movies.append({
                        'movie_id': movie[0],
                        'title': movie[1],
                        'year': year,
                        'poster_path': movie[3],
                        'overview': movie[4],
                        'vote_average': movie[5],
                        'vote_count': movie[6],
                        'genres': genres,
                        'predicted_rating': float(score)
                    })
            
            logger.info(f"Recomendaciones generadas para usuario {user_id}: {len(recommended_movies)} películas")
            for movie in recommended_movies:
                logger.info(f"Película recomendada: {movie['title']} (Score: {movie['predicted_rating']:.2f})")
            
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