import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pickle
import logging
import sqlite3
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import math
import os
from scipy.sparse import csr_matrix

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVDRecommender:
    def __init__(self, db_path: str = 'data/tmdb_movies.db', models_dir: str = 'models/svd'):
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models: Dict[int, SVD] = {}  # Dictionary to store models per cluster
        self.top_movies_per_cluster: Dict[int, List[int]] = {}  # Store top movies per cluster
        self.reader = Reader(rating_scale=(0, 1))

        # Crear directorio de modelos si no existe
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Intentar cargar modelos al inicializar
        self._initialize_models()

    def _initialize_models(self):
        """Inicializa los modelos al crear la instancia."""
        try:
            # Verificar si hay modelos guardados
            model_files = list(self.models_dir.glob('svd_model_cluster_*.pkl'))
            if not model_files:
                logger.warning("No se encontraron modelos guardados. Entrenando nuevos modelos...")
                self.train()
            else:
                # Intentar cargar modelos existentes
                if not self.load_models():
                    logger.warning("Error al cargar modelos existentes. Entrenando nuevos modelos...")
                    self.train()
        except Exception as e:
            logger.error(f"Error al inicializar modelos: {str(e)}")
            raise

    def _get_top_movies_per_cluster(self, cluster_id: int, limit: int = 1000) -> List[int]:
        """Obtiene las películas más vistas para un cluster específico."""
        query = """
        WITH UserClusterMovies AS (
            SELECT 
                uf.movie_id,
                COUNT(*) as view_count
            FROM user_film uf
            JOIN user_cluster uc ON uf.user_id = uc.user_id
            WHERE uc.cluster = ?
            GROUP BY uf.movie_id
        )
        SELECT movie_id
        FROM UserClusterMovies
        ORDER BY view_count DESC
        LIMIT ?
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(cluster_id, limit))['movie_id'].tolist()

    def _get_cluster_data(self, cluster_id: int) -> pd.DataFrame:
        """Obtiene los datos de ratings para un cluster específico."""
        query = """
        SELECT 
            uf.user_id,
            uf.movie_id,
            1 as rating
        FROM user_film uf
        JOIN user_cluster uc ON uf.user_id = uc.user_id
        WHERE uc.cluster = ?
        AND uf.movie_id IN (
            SELECT movie_id 
            FROM (
                SELECT 
                    movie_id,
                    COUNT(*) as view_count
                FROM user_film uf2
                JOIN user_cluster uc2 ON uf2.user_id = uc2.user_id
                WHERE uc2.cluster = ?
                GROUP BY movie_id
                ORDER BY view_count DESC
                LIMIT 1000
            )
        )
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(cluster_id, cluster_id))

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> Dict:
        """Calcula todas las métricas de evaluación."""
        # MSE y RMSE
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)

        # Logging de diagnóstico
        logger.info(f"\nDiagnóstico de predicciones:")
        logger.info(f"Rango de valores reales: [{np.min(y_true):.2f}, {np.max(y_true):.2f}]")
        logger.info(f"Rango de predicciones: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")
        logger.info(f"Media de valores reales: {np.mean(y_true):.4f}")
        logger.info(f"Media de predicciones: {np.mean(y_pred):.4f}")
        logger.info(f"Desviación estándar de valores reales: {np.std(y_true):.4f}")
        logger.info(f"Desviación estándar de predicciones: {np.std(y_pred):.4f}")

        # Calcular umbral adaptativo basado en la distribución de los datos
        # Usar un umbral más estricto para la clasificación binaria
        threshold = np.percentile(y_true, 75)  # Usar el percentil 75 como umbral
        logger.info(f"Umbral de clasificación (percentil 75): {threshold:.4f}")

        # Métricas de clasificación
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)

        # Logging de distribución de clases
        logger.info(f"\nDistribución de clases:")
        logger.info(f"Positivos reales: {np.sum(y_true_binary)}")
        logger.info(f"Positivos predichos: {np.sum(y_pred_binary)}")
        logger.info(f"Total de muestras: {len(y_true)}")
        logger.info(f"Proporción de positivos reales: {np.mean(y_true_binary):.4f}")
        logger.info(f"Proporción de positivos predichos: {np.mean(y_pred_binary):.4f}")

        # Calcular métricas con manejo de casos especiales
        accuracy = accuracy_score(y_true_binary.flatten(), y_pred_binary.flatten())
        
        # Calcular precision y recall solo si hay predicciones positivas
        if np.sum(y_pred_binary) > 0:
            precision = precision_score(y_true_binary.flatten(), y_pred_binary.flatten(), zero_division=0)
            recall = recall_score(y_true_binary.flatten(), y_pred_binary.flatten(), zero_division=0)
            f1 = f1_score(y_true_binary.flatten(), y_pred_binary.flatten(), zero_division=0)
        else:
            precision = 0
            recall = 0
            f1 = 0
            logger.warning("No se encontraron predicciones positivas")

        logger.info(f"\nMétricas de clasificación:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

        # Reconstruir matrices de usuario-item
        user_ids = np.array([pred.uid for pred in self.test_predictions])
        item_ids = np.array([pred.iid for pred in self.test_predictions])
        ratings_true = np.array([pred.r_ui for pred in self.test_predictions])
        ratings_pred = np.array([pred.est for pred in self.test_predictions])

        # Crear matrices dispersas
        n_users = len(np.unique(user_ids))
        n_items = len(np.unique(item_ids))
        
        # Mapear IDs a índices
        user_map = {uid: i for i, uid in enumerate(np.unique(user_ids))}
        item_map = {iid: i for i, iid in enumerate(np.unique(item_ids))}
        
        # Convertir IDs a índices
        user_indices = np.array([user_map[uid] for uid in user_ids])
        item_indices = np.array([item_map[iid] for iid in item_ids])
        
        # Crear matrices
        matrix_true = csr_matrix((ratings_true, (user_indices, item_indices)), shape=(n_users, n_items))
        matrix_pred = csr_matrix((ratings_pred, (user_indices, item_indices)), shape=(n_users, n_items))

        # Métricas de ranking
        precision_k = 0
        recall_k = 0
        ndcg_k = 0
        map_k = 0
        hit_rate_k = 0
        n_users_with_relevant = 0
        
        for user_idx in range(n_users):
            # Obtener predicciones y valores reales para este usuario
            user_true = matrix_true[user_idx].toarray().flatten()
            user_pred = matrix_pred[user_idx].toarray().flatten()
            
            # Obtener top-K items según predicciones
            top_k_items = np.argsort(user_pred)[-k:]
            
            # Obtener items relevantes (con rating >= threshold)
            relevant_items = np.where(user_true >= threshold)[0]
            
            if len(relevant_items) == 0:
                continue
                
            n_users_with_relevant += 1
            
            # Precision@K
            hits = len(set(top_k_items) & set(relevant_items))
            precision_k += hits / k if k > 0 else 0
            
            # Recall@K
            recall_k += hits / len(relevant_items)
            
            # NDCG@K
            dcg = 0
            idcg = 0
            
            # Calcular DCG
            for i, item in enumerate(top_k_items):
                if item in relevant_items:
                    dcg += 1 / np.log2(i + 2)
            
            # Calcular IDCG (ranking ideal)
            ideal_ranking = np.argsort(user_true)[::-1][:k]
            for i, item in enumerate(ideal_ranking):
                if user_true[item] >= threshold:
                    idcg += 1 / np.log2(i + 2)
            
            ndcg_k += dcg / idcg if idcg > 0 else 0
            
            # MAP@K
            ap = 0
            hits = 0
            for i, item in enumerate(top_k_items):
                if item in relevant_items:
                    hits += 1
                    ap += hits / (i + 1)
            map_k += ap / min(k, len(relevant_items))
            
            # Hit Rate@K
            hit_rate_k += 1 if len(set(top_k_items) & set(relevant_items)) > 0 else 0

        # Normalizar métricas
        if n_users_with_relevant > 0:
            precision_k /= n_users_with_relevant
            recall_k /= n_users_with_relevant
            ndcg_k /= n_users_with_relevant
            map_k /= n_users_with_relevant
            hit_rate_k /= n_users_with_relevant

        logger.info(f"\nMétricas de ranking:")
        logger.info(f"Usuarios con items relevantes: {n_users_with_relevant}")
        logger.info(f"Precision@{k}: {precision_k:.4f}")
        logger.info(f"Recall@{k}: {recall_k:.4f}")
        logger.info(f"NDCG@{k}: {ndcg_k:.4f}")
        logger.info(f"MAP@{k}: {map_k:.4f}")
        logger.info(f"HitRate@{k}: {hit_rate_k:.4f}")

        return {
            'MSE': mse,
            'RMSE': rmse,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            f'Precision@{k}': precision_k,
            f'Recall@{k}': recall_k,
            f'NDCG@{k}': ndcg_k,
            f'MAP@{k}': map_k,
            f'HitRate@{k}': hit_rate_k
        }

    def _plot_variance_analysis(self, max_components: int = 500, cluster_id: int = None):
        """Analiza y grafica la varianza explicada por los componentes."""
        if self.user_movie_matrix is None or self.user_movie_matrix.empty:
            logger.error("No hay matriz de usuario-película disponible para el análisis de varianza")
            return None

        # Asegurar que max_components no exceda el número de columnas
        max_components = min(max_components, self.user_movie_matrix.shape[1])
        
        # Calcular varianza explicada para diferentes números de componentes
        variances = []
        components_range = []
        current_components = 1
        target_variance = 0.95

        while current_components <= max_components:
            svd = TruncatedSVD(n_components=current_components, random_state=42)
            svd.fit(self.user_movie_matrix)
            current_variance = svd.explained_variance_ratio_.sum()
            variances.append(current_variance)
            components_range.append(current_components)

            logger.info(f"Componentes {current_components}: {current_variance*100:.2f}% de varianza explicada")

            if current_variance >= target_variance:
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
        plt.title(f'Análisis de Varianza Explicada por SVD - Cluster {cluster_id}')
        plt.grid(True)
        plt.legend()

        # Guardar gráfica
        plt.savefig(f'models/svd_variance_analysis_cluster_{cluster_id}.png')
        plt.close()

        return n_components_best

    def prepare_data(self):
        """Prepara los datos para el entrenamiento usando views en vez de ratings."""
        try:
            conn = sqlite3.connect('data/tmdb_movies.db')

            # Obtener datos de visualizaciones
            query = """
            SELECT user_id, movie_id
            FROM user_film
            """
            df = pd.read_sql_query(query, conn)

            # Crear columna 'view' con valor 1 (binario: visto/no visto)
            df['view'] = 1

            # Crear matriz de usuario-película binaria
            self.user_movie_matrix = df.pivot_table(
                index='user_id',
                columns='movie_id',
                values='view',
                fill_value=0
            )

            # Crear dataset para Surprise (usando 0-1 como escala)
            df_surprise = df[['user_id', 'movie_id', 'view']]
            reader = Reader(rating_scale=(0, 1))
            data = Dataset.load_from_df(
                df_surprise,
                reader
            )

            # Dividir en train y test
            trainset, testset = train_test_split(data, test_size=0.2)

            logger.info(f"Matriz de usuario-película (views):\n{self.user_movie_matrix}")
            logger.info(f"Dimensiones de la matriz: {self.user_movie_matrix.shape}")

            return trainset, testset

        except Exception as e:
            logger.error(f"Error al preparar datos: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def save_models(self):
        """Guarda los modelos y las películas top por cluster."""
        try:
            # Guardar modelos
            for cluster_id, model in self.models.items():
                model_path = self.models_dir / f'svd_model_cluster_{cluster_id}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Modelo guardado para cluster {cluster_id} en {model_path}")

            # Guardar películas top por cluster
            top_movies_path = self.models_dir / 'top_movies_per_cluster.pkl'
            with open(top_movies_path, 'wb') as f:
                pickle.dump(self.top_movies_per_cluster, f)
            logger.info(f"Películas top guardadas en {top_movies_path}")

        except Exception as e:
            logger.error(f"Error al guardar modelos: {str(e)}")
            raise

    def load_models(self):
        """Carga los modelos y las películas top por cluster."""
        try:
            # Verificar que el directorio existe
            if not self.models_dir.exists():
                logger.error(f"El directorio de modelos {self.models_dir} no existe")
                return False

            # Cargar películas top por cluster
            top_movies_path = self.models_dir / 'top_movies_per_cluster.pkl'
            if top_movies_path.exists():
                try:
                    with open(top_movies_path, 'rb') as f:
                        self.top_movies_per_cluster = pickle.load(f)
                    logger.info(f"Películas top cargadas exitosamente: {len(self.top_movies_per_cluster)} clusters")
                except Exception as e:
                    logger.error(f"Error al cargar películas top: {str(e)}")
                    return False
            else:
                logger.error(f"No se encontró el archivo de películas top en {top_movies_path}")
                return False

            # Cargar modelos
            model_files = list(self.models_dir.glob('svd_model_cluster_*.pkl'))
            if not model_files:
                logger.error("No se encontraron archivos de modelos en el directorio")
                return False

            logger.info(f"Encontrados {len(model_files)} archivos de modelos")
            
            for model_file in model_files:
                try:
                    cluster_id = int(model_file.stem.split('_')[-1])
                    logger.info(f"Intentando cargar modelo para cluster {cluster_id} desde {model_file}")
                    
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                        if model is None:
                            logger.error(f"El modelo cargado para cluster {cluster_id} es None")
                            continue
                        self.models[cluster_id] = model
                        logger.info(f"Modelo cargado exitosamente para cluster {cluster_id}")
                except Exception as e:
                    logger.error(f"Error al cargar modelo para cluster {cluster_id}: {str(e)}")
                    continue

            if not self.models:
                logger.error("No se pudo cargar ningún modelo")
                return False

            logger.info(f"Modelos cargados exitosamente para {len(self.models)} clusters")
            return True

        except Exception as e:
            logger.error(f"Error general al cargar modelos: {str(e)}")
            return False

    def train(self, n_epochs: int = 20, lr_all: float = 0.005, reg_all: float = 0.02, save_models: bool = True):
        """Entrena un modelo SVD para cada cluster usando el número óptimo de componentes."""
        try:
            # Obtener todos los clusters únicos
            with sqlite3.connect(self.db_path) as conn:
                clusters = pd.read_sql_query("SELECT DISTINCT cluster FROM user_cluster", conn)['cluster'].tolist()

            if not clusters:
                logger.error("No se encontraron clusters en la base de datos")
                return

            logger.info(f"Entrenando modelos para {len(clusters)} clusters")
            
            for cluster_id in clusters:
                logger.info(f"\n{'='*50}")
                logger.info(f"Entrenando modelo para cluster {cluster_id}")
                logger.info(f"{'='*50}")
                
                # Obtener películas top para este cluster
                self.top_movies_per_cluster[cluster_id] = self._get_top_movies_per_cluster(cluster_id)
                
                # Obtener datos del cluster
                cluster_data = self._get_cluster_data(cluster_id)
                
                if len(cluster_data) == 0:
                    logger.warning(f"No hay datos para el cluster {cluster_id}")
                    continue
                
                # Crear matriz de usuario-película para el cluster
                self.user_movie_matrix = cluster_data.pivot_table(
                    index='user_id',
                    columns='movie_id',
                    values='rating',
                    fill_value=0
                )
                
                logger.info(f"Dimensiones de la matriz para cluster {cluster_id}: {self.user_movie_matrix.shape}")
                
                # Analizar varianza y obtener número óptimo de componentes
                logger.info(f"\nAnalizando varianza para cluster {cluster_id}...")
                optimal_components = self._plot_variance_analysis(
                    max_components=min(500, self.user_movie_matrix.shape[1]),
                    cluster_id=cluster_id
                )
                
                if optimal_components is None:
                    logger.error(f"No se pudo determinar el número óptimo de componentes para cluster {cluster_id}")
                    continue
                
                logger.info(f"\nNúmero óptimo de componentes para cluster {cluster_id}: {optimal_components}")
                
                # Crear y entrenar el modelo para este cluster usando el número óptimo de componentes
                data = Dataset.load_from_df(cluster_data, self.reader)
                trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
                
                model = SVD(n_factors=optimal_components, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
                model.fit(trainset)
                
                # Guardar el modelo
                self.models[cluster_id] = model
                
                # Evaluar el modelo
                self.test_predictions = model.test(testset)
                y_true = np.array([pred.r_ui for pred in self.test_predictions])
                y_pred = np.array([pred.est for pred in self.test_predictions])
                
                metrics = self._calculate_metrics(y_true, y_pred)
                logger.info(f"\nMétricas para cluster {cluster_id}:")
                for metric_name, value in metrics.items():
                    logger.info(f"{metric_name}: {value:.4f}")

            # Guardar modelos si se solicita
            if save_models:
                self.save_models()
                
            logger.info("\nEntrenamiento completado exitosamente")
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            raise

    def predict(self, user_id: int, movie_id: int) -> float:
        """Realiza una predicción para un usuario y película específicos."""
        # Obtener el cluster del usuario
        with sqlite3.connect(self.db_path) as conn:
            cluster_df = pd.read_sql_query(
                "SELECT cluster FROM user_cluster WHERE user_id = ?",
                conn,
                params=(user_id,)
            )
            
            if len(cluster_df) == 0:
                logger.warning(f"Usuario {user_id} no encontrado en ningún cluster")
                return 0.0
                
            cluster_id = cluster_df['cluster'].iloc[0]
            
            if cluster_id not in self.models:
                logger.warning(f"No hay modelo para el cluster {cluster_id}")
                return 0.0
                
            if movie_id not in self.top_movies_per_cluster[cluster_id]:
                logger.warning(f"Película {movie_id} no está en las top 100 del cluster {cluster_id}")
                return 0.0
        
        # Realizar la predicción usando el modelo del cluster
        model = self.models[cluster_id]
        prediction = model.predict(user_id, movie_id)
        return prediction.est

    def get_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Obtiene recomendaciones para un usuario específico."""
        try:
            # Obtener el cluster del usuario
            with sqlite3.connect(self.db_path) as conn:
                cluster_df = pd.read_sql_query(
                    "SELECT cluster FROM user_cluster WHERE user_id = ?",
                    conn,
                    params=(user_id,)
                )
                
                if len(cluster_df) == 0:
                    logger.warning(f"Usuario {user_id} no encontrado en ningún cluster")
                    return []
                    
                cluster_id = cluster_df['cluster'].iloc[0]
                logger.info(f"Usuario {user_id} pertenece al cluster {cluster_id}")
                
                if cluster_id not in self.models:
                    logger.error(f"No hay modelo para el cluster {cluster_id}. Modelos disponibles: {list(self.models.keys())}")
                    return []
            
            # Obtener películas ya vistas por el usuario
            with sqlite3.connect(self.db_path) as conn:
                viewed_movies = pd.read_sql_query(
                    "SELECT movie_id FROM user_film WHERE user_id = ?",
                    conn,
                    params=(user_id,)
                )['movie_id'].tolist()
            
            logger.info(f"Usuario {user_id} ha visto {len(viewed_movies)} películas")
            
            # Filtrar películas no vistas y que estén en las top 100 del cluster
            available_movies = [
                movie_id for movie_id in self.top_movies_per_cluster[cluster_id]
                if movie_id not in viewed_movies
            ]
            
            logger.info(f"Hay {len(available_movies)} películas disponibles para recomendar")
            
            # Realizar predicciones para todas las películas disponibles
            predictions = []
            model = self.models[cluster_id]
            
            for movie_id in available_movies:
                pred = model.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))
            
            # Ordenar por rating predicho
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Obtener las mejores y peores recomendaciones
            best_recommendations = predictions[:n_recommendations]
            worst_recommendations = predictions[-n_recommendations:]
            
            # Imprimir recomendaciones
            logger.info(f"\nMejores recomendaciones para usuario {user_id}:")
            for movie_id, score in best_recommendations:
                logger.info(f"Película {movie_id}: {score:.4f}")
                
            logger.info(f"\nPeores recomendaciones para usuario {user_id}:")
            for movie_id, score in worst_recommendations:
                logger.info(f"Película {movie_id}: {score:.4f}")
            
            return best_recommendations
            
        except Exception as e:
            logger.error(f"Error al obtener recomendaciones para usuario {user_id}: {str(e)}")
            return []

if __name__ == "__main__":
    # Ejemplo de uso
    recommender = SVDRecommender()
    
    # Ejemplo de recomendaciones
    user_id = 1  # ID de usuario de ejemplo
    recommendations = recommender.get_recommendations(user_id)
    print(f"\nRecomendaciones para usuario {user_id}:")
    for movie_id, score in recommendations:
        print(f"Película {movie_id}: {score:.4f}")