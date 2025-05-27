import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
import pickle
import logging
import sqlite3
from pathlib import Path
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import math
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenreRecommender:
    def __init__(self):
        self.model_path = Path('models/genre_svd_model.pkl')
        self.model_path.parent.mkdir(exist_ok=True)
        self.genre_mapping = {}  # Mapeo de IDs originales a IDs normalizados
        self.reverse_mapping = {}  # Mapeo inverso
        self.user_genre_matrix = None
        self.genre_names = None
        self.U = None  # Left singular vectors
        self.S = None  # Singular values
        self.Vh = None  # Right singular vectors
        self.normalized_matrix = None
        self.n_components = None  # Número óptimo de componentes
        self.explained_variance_ratio = None
        
    def plot_explained_variance(self):
        """Plot individual and cumulative explained variance"""
        if self.S is None:
            logger.error("No SVD model available. Train the model first.")
            return

        # Calculate explained variance
        explained_variance = (self.S ** 2) / (self.S ** 2).sum()
        cumulative_variance = np.cumsum(explained_variance)
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot individual explained variance
        ax1.bar(range(1, len(explained_variance) + 1), explained_variance)
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Explained Variance')
        ax1.set_title('Individual Explained Variance')
        
        # Plot cumulative explained variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        ax2.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} Components')
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        
        # Add grid and adjust layout
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('models/explained_variance.png')
        plt.close()
        
        # Print statistics
        logger.info(f"\nExplained Variance Statistics:")
        logger.info(f"Number of components for 95% variance: {n_components_95}")
        logger.info(f"Total variance explained by {n_components_95} components: {cumulative_variance[n_components_95-1]:.2%}")
        logger.info(f"Variance explained by first component: {explained_variance[0]:.2%}")
        logger.info(f"Variance explained by last component: {explained_variance[n_components_95-1]:.2%}")
        
        return n_components_95

    def prepare_data(self):
        """Prepara los datos de user_film y movie_genres para el entrenamiento."""
        try:
            conn = sqlite3.connect('data/tmdb_movies.db')
            
            # Obtener todos los géneros y crear mapeo
            cursor = conn.cursor()
            cursor.execute('SELECT id, name FROM genres ORDER BY id')
            genres = cursor.fetchall()
            
            # Crear mapeo de IDs originales a IDs normalizados
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
            self.user_genre_matrix = df.pivot(
                index='user_id',
                columns='genre_id',
                values='normalized_rating'
            ).fillna(0)
            
            # Obtener nombres de géneros
            self.genre_names = [genre[1] for genre in genres]
            
            # Dividir en train y test
            train_size = int(len(self.user_genre_matrix) * 0.8)
            train_matrix = self.user_genre_matrix.iloc[:train_size]
            test_matrix = self.user_genre_matrix.iloc[train_size:]
            
            return train_matrix, test_matrix
            
        except Exception as e:
            logger.error(f"Error al preparar datos: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 15) -> Dict:
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
        
        # Métricas de clasificación
        accuracy = 0
        precision = 0
        recall = 0
        f1_score = 0
        
        n_users = len(y_true)
        
        # Calcular estadísticas de los datos
        true_mean = np.mean(y_true)
        pred_mean = np.mean(y_pred)
        true_std = np.std(y_true)
        pred_std = np.std(y_pred)
        
        logger.info(f"\nEstadísticas de los datos:")
        logger.info(f"Media de valores reales: {true_mean:.4f}")
        logger.info(f"Media de predicciones: {pred_mean:.4f}")
        logger.info(f"Desviación estándar de valores reales: {true_std:.4f}")
        logger.info(f"Desviación estándar de predicciones: {pred_std:.4f}")
        
        # Ajustar el umbral basado en la media de los datos
        threshold = true_mean
        logger.info(f"Umbral de clasificación: {threshold:.4f}")
        
        # Contadores para diagnóstico
        total_predictions = 0
        total_positives = 0
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        
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
            
            # Métricas de clasificación
            # Convertir predicciones continuas a binarias
            y_true_binary = (y_true[i] >= threshold).astype(int)
            y_pred_binary = (y_pred[i] >= threshold).astype(int)
            
            # Actualizar contadores
            total_predictions += len(y_true_binary)
            total_positives += np.sum(y_true_binary)
            total_true_positives += np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            total_false_positives += np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            total_false_negatives += np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            # Calcular métricas de clasificación
            true_positives = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            true_negatives = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            
            # Accuracy
            accuracy += (true_positives + true_negatives) / len(y_true_binary)
            
            # Precision
            precision += true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            
            # Recall
            recall += true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Promediar métricas
        n_users = float(n_users)
        
        # Calcular F1-score promedio
        avg_precision = precision / n_users
        avg_recall = recall / n_users
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        # Logging de diagnóstico
        logger.info(f"\nDiagnóstico de clasificación:")
        logger.info(f"Total de predicciones: {total_predictions}")
        logger.info(f"Total de positivos reales: {total_positives}")
        logger.info(f"Total de verdaderos positivos: {total_true_positives}")
        logger.info(f"Total de falsos positivos: {total_false_positives}")
        logger.info(f"Total de falsos negativos: {total_false_negatives}")
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            f'Precision@{k}': precision_k / n_users,
            f'Recall@{k}': recall_k / n_users,
            f'NDCG@{k}': ndcg_k / n_users,
            f'MAP@{k}': map_k / n_users,
            f'HitRate@{k}': hit_rate_k / n_users,
            'Accuracy': accuracy / n_users,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-Score': f1_score
        }
    
    def train(self):
        """Entrena el modelo SVD."""
        try:
            train_matrix, test_matrix = self.prepare_data()
            
            # Convertir a matriz densa para SVD
            train_dense = train_matrix.values
            test_dense = test_matrix.values
            
            # Realizar SVD
            logger.info("Realizando descomposición SVD...")
            self.U, self.S, self.Vh = np.linalg.svd(train_dense, full_matrices=False)
            
            # Encontrar número óptimo de componentes
            self.n_components = self.plot_explained_variance()
            logger.info(f"Usando {self.n_components} componentes para explicar 95% de la varianza")
            
            # Mantener solo n_components
            self.U = self.U[:, :self.n_components]
            self.S = self.S[:self.n_components]
            self.Vh = self.Vh[:self.n_components, :]
            
            # Transformar los datos
            self.transformed_matrix = self.U * self.S
            self.normalized_matrix = normalize(self.transformed_matrix)
            
            # Calcular predicciones para test
            test_transformed = test_dense @ self.Vh.T
            reconstructed = test_transformed @ self.Vh
            
            # Calcular métricas
            train_metrics = self._calculate_metrics(
                train_dense,
                self.transformed_matrix @ self.Vh
            )
            
            test_metrics = self._calculate_metrics(
                test_dense,
                reconstructed
            )
            
            # Logging de métricas
            logger.info("\nMétricas de entrenamiento:")
            for metric, value in train_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            logger.info("\nMétricas de test:")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Guardar modelo
            self.save_model()
            
            return train_metrics, test_metrics
            
        except Exception as e:
            logger.error(f"Error al entrenar modelo: {str(e)}")
            raise
    
    def save_model(self):
        """Guarda el modelo entrenado."""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'U': self.U,
                    'S': self.S,
                    'Vh': self.Vh,
                    'normalized_matrix': self.normalized_matrix,
                    'transformed_matrix': self.transformed_matrix,
                    'genre_mapping': self.genre_mapping,
                    'reverse_mapping': self.reverse_mapping,
                    'n_components': self.n_components,
                    'genre_names': self.genre_names,
                    'user_genre_matrix': self.user_genre_matrix  # Guardar la matriz completa
                }, f)
            logger.info("Modelo guardado exitosamente")
        except Exception as e:
            logger.error(f"Error al guardar modelo: {str(e)}")
            raise
    
    def load_model(self):
        """Carga el modelo guardado."""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.U = data['U']
                self.S = data['S']
                self.Vh = data['Vh']
                self.normalized_matrix = data['normalized_matrix']
                self.transformed_matrix = data['transformed_matrix']
                self.genre_mapping = data['genre_mapping']
                self.reverse_mapping = data['reverse_mapping']
                self.n_components = data['n_components']
                self.genre_names = data['genre_names']
                self.user_genre_matrix = data['user_genre_matrix']  # Cargar la matriz completa
                
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            raise
    
    def get_recommendations(self, user_id, n_recommendations=3):
        """Obtiene recomendaciones de géneros para un usuario."""
        try:
            if self.normalized_matrix is None:
                self.load_model()
            
            if self.user_genre_matrix is None:
                logger.error("No se pudo cargar la matriz de usuario-género")
                return []
            
            # Verificar si el usuario existe
            if user_id not in self.user_genre_matrix.index:
                logger.error(f"Usuario {user_id} no encontrado en la matriz")
                return []
            
            # Obtener índice del usuario
            user_idx = self.user_genre_matrix.index.get_loc(user_id)
            
            # Si el usuario está en la matriz de entrenamiento, usar su vector normalizado
            if user_idx < len(self.normalized_matrix):
                user_vector = self.normalized_matrix[user_idx]
            else:
                # Si el usuario está en test, calcular su vector
                user_data = self.user_genre_matrix.iloc[user_idx].values
                user_transformed = user_data @ self.Vh.T
                user_vector = normalize(user_transformed.reshape(1, -1))[0]
            
            # Calcular similitud con todos los usuarios
            similarity_scores = np.dot(self.normalized_matrix, user_vector)
            
            # Obtener géneros vistos por el usuario
            seen_genres = set(self.user_genre_matrix.columns[self.user_genre_matrix.iloc[user_idx] > 0])
            
            # Obtener géneros preferidos del usuario
            conn = sqlite3.connect('data/tmdb_movies.db')
            cursor = conn.cursor()
            cursor.execute('''
            SELECT g.id
            FROM genres g
            JOIN user_genres ug ON g.id = ug.genre_id
            WHERE ug.user_id = ?
            ''', (user_id,))
            preferred_genres = {row[0] for row in cursor.fetchall()}
            
            # Calcular scores para cada género
            genre_scores = []
            for genre_idx, genre_id in enumerate(self.user_genre_matrix.columns):
                # Convertir ID normalizado a ID original
                original_genre_id = self.reverse_mapping[genre_idx]
                
                # Calcular score base
                base_score = float(similarity_scores[genre_idx])
                
                # Ajustar score basado en si ya ha visto el género o es preferido
                if original_genre_id in seen_genres or original_genre_id in preferred_genres:
                    view_count = self.user_genre_matrix.iloc[user_idx, genre_idx]
                    adjusted_score = base_score * (1 / (1 + 0.5 * (view_count + 1)))
                else:
                    adjusted_score = base_score * 1.5
                
                genre_scores.append({
                    'genre_id': original_genre_id,
                    'name': self.genre_names[genre_idx],
                    'score': adjusted_score
                })
            
            # Ordenar por score y filtrar géneros ya vistos/preferidos
            genre_scores.sort(key=lambda x: x['score'], reverse=True)
            recommended_genres = [
                genre for genre in genre_scores
                if genre['genre_id'] not in seen_genres and genre['genre_id'] not in preferred_genres
            ][:n_recommendations]
            
            if not recommended_genres:
                logger.warning(f"No se encontraron recomendaciones para el usuario {user_id}")
            
            return recommended_genres
            
        except Exception as e:
            logger.error(f"Error al obtener recomendaciones: {str(e)}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()

if __name__ == "__main__":
    recommender = GenreRecommender()
    train_metrics, test_metrics = recommender.train()
    
    # Mostrar métricas de forma más legible
    print("\n=== MÉTRICAS DE EVALUACIÓN ===")
    print("\nMétricas de Entrenamiento:")
    print("-" * 50)
    for metric, value in train_metrics.items():
        print(f"{metric:15}: {value:.4f}")
    
    print("\nMétricas de Test:")
    print("-" * 50)
    for metric, value in test_metrics.items():
        print(f"{metric:15}: {value:.4f}")
    
    # Mostrar número de componentes óptimo
    print(f"\nNúmero óptimo de componentes: {recommender.n_components}")
    print(f"Varianza explicada: {sum((recommender.S ** 2) / (recommender.S ** 2).sum()[:recommender.n_components]):.2%}") 