from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import sqlite3
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MovieDataset(Dataset):
    def __init__(self, user_movie_matrix):
        self.user_movie_matrix = user_movie_matrix
        self.users = []
        self.movies = []
        self.ratings = []

        # Convert sparse matrix to list of (user, movie, rating) tuples
        for user_idx in range(user_movie_matrix.shape[0]):
            for movie_idx in range(user_movie_matrix.shape[1]):
                rating = user_movie_matrix.iloc[user_idx, movie_idx]
                if rating > 0:
                    self.users.append(user_idx)
                    self.movies.append(movie_idx)
                    self.ratings.append(rating)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'movie': torch.tensor(self.movies[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }


class NeuralCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50, layers=[100, 50, 20]):
        super(NeuralCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # Build MLP layers
        self.layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        for layer_size in layers:
            self.layers.append(nn.Linear(input_dim, layer_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(layer_size))
            self.layers.append(nn.Dropout(0.2))
            input_dim = layer_size

        self.output_layer = nn.Linear(layers[-1], 1)

    def forward(self, user_input, movie_input):
        user_embedded = self.user_embedding(user_input)
        movie_embedded = self.movie_embedding(movie_input)

        # Concatenate embeddings
        x = torch.cat([user_embedded, movie_embedded], dim=1)

        # Pass through MLP layers
        for layer in self.layers:
            x = layer(x)

        # Output layer
        output = self.output_layer(x)
        return output.squeeze()


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Calcula el NDCG@K."""
    def dcg_at_k(y_true, y_pred, k):
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(2, k + 2))
        return np.sum(gains / discounts)

    def idcg_at_k(y_true, k):
        y_true = np.sort(y_true)[::-1]
        return dcg_at_k(y_true, y_true, k)

    dcg = dcg_at_k(y_true, y_pred, k)
    idcg = idcg_at_k(y_true, k)
    return dcg / idcg if idcg > 0 else 0


def map_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Calcula el MAP@K."""
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order[:k])
    ap = 0
    hits = 0
    total_relevant = np.sum(y_true >= 4)

    if total_relevant == 0:
        return 0

    for i, pred in enumerate(y_true):
        if pred >= 4:  # Consideramos ratings >= 4 como positivos
            hits += 1
            ap += hits / (i + 1)
    return ap / total_relevant if total_relevant > 0 else 0


def hit_rate_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Calcula el Hit Rate@K."""
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order[:k])
    return 1 if np.any(y_true >= 4) else 0


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Calcula el Recall@K."""
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order[:k])
    relevant = np.sum(y_true >= 4)
    if relevant == 0:
        return 0
    return np.sum(y_true >= 4) / relevant


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Calcula la precisión@k."""
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.sum(y_true >= 4) / k


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> Dict:
    """Calcula métricas de evaluación para el modelo."""
    # Métricas de regresión
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    # Convertir a clasificación binaria (rating >= 4 es positivo)
    y_true_binary = (y_true >= 4).astype(int)
    y_pred_binary = (y_pred >= 4).astype(int)

    # Métricas de clasificación
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    # Métricas de ranking
    precision_k = precision_at_k(y_true, y_pred, k)
    recall_k = recall_at_k(y_true, y_pred, k)
    ndcg_k = ndcg_at_k(y_true, y_pred, k)
    map_k = map_at_k(y_true, y_pred, k)
    hit_rate_k = hit_rate_at_k(y_true, y_pred, k)

    # Logging de diagnóstico
    logger.info(f"\nDiagnóstico de métricas:")
    logger.info(f"Total de predicciones: {len(y_true)}")
    logger.info(f"Predicciones positivas (>=4): {np.sum(y_true >= 4)}")
    logger.info(f"Media de predicciones: {np.mean(y_pred):.4f}")
    logger.info(f"Desviación estándar de predicciones: {np.std(y_pred):.4f}")

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


class MovieRecommenderDL:
    def __init__(self, embedding_dim: int = 50):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.movie_features = None
        self.movie_similarity = None
        self.movies_df = None
        self.user_movie_matrix = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self.model = None
        self.embedding_dim = embedding_dim
        self.metrics = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.movie_data_cache = {}  # Cache for movie data

    def _get_movie_data(self, movie_id: int) -> Dict:
        """Get movie data from SQLite database."""
        # Check cache first
        if movie_id in self.movie_data_cache:
            return self.movie_data_cache[movie_id]

        conn = sqlite3.connect('data/tmdb_movies.db')
        try:
            cursor = conn.cursor()

            # Get basic movie info
            cursor.execute('''
            SELECT * FROM movies WHERE movie_id = ?
            ''', (movie_id,))
            movie = cursor.fetchone()

            if not movie:
                return None

            # Get column names
            columns = [description[0] for description in cursor.description]
            movie_dict = dict(zip(columns, movie))

            # Get genres
            cursor.execute('''
            SELECT g.name 
            FROM genres g
            JOIN movie_genres mg ON g.id = mg.genre_id
            WHERE mg.movie_id = ?
            ''', (movie_id,))
            movie_dict['genres'] = [row[0] for row in cursor.fetchall()]

            # Get production companies
            cursor.execute('''
            SELECT pc.name 
            FROM production_companies pc
            JOIN movie_production_companies mpc ON pc.id = mpc.company_id
            WHERE mpc.movie_id = ?
            ''', (movie_id,))
            movie_dict['production_companies'] = [row[0] for row in cursor.fetchall()]

            # Get production countries
            cursor.execute('''
            SELECT pc.name 
            FROM production_countries pc
            JOIN movie_production_countries mpc ON pc.iso_3166_1 = mpc.country_code
            WHERE mpc.movie_id = ?
            ''', (movie_id,))
            movie_dict['production_countries'] = [row[0] for row in cursor.fetchall()]

            # Get spoken languages
            cursor.execute('''
            SELECT sl.name 
            FROM spoken_languages sl
            JOIN movie_spoken_languages msl ON sl.iso_639_1 = msl.language_code
            WHERE msl.movie_id = ?
            ''', (movie_id,))
            movie_dict['spoken_languages'] = [row[0] for row in cursor.fetchall()]

            # Get collection info if exists
            cursor.execute('''
            SELECT c.name, c.poster_path, c.backdrop_path
            FROM collections c
            JOIN movie_collections mc ON c.id = mc.collection_id
            WHERE mc.movie_id = ?
            ''', (movie_id,))
            collection = cursor.fetchone()
            if collection:
                movie_dict['belongs_to_collection'] = {
                    'name': collection[0],
                    'poster_path': collection[1],
                    'backdrop_path': collection[2]
                }

            # Cache the result
            self.movie_data_cache[movie_id] = movie_dict
            return movie_dict
        finally:
            conn.close()

    def fit(self, data: Dict):
        """Entrena ambos modelos de recomendación."""
        logger.info("Iniciando entrenamiento de modelos...")

        # Get movies data from database
        conn = sqlite3.connect('data/tmdb_movies.db')
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT movie_id, title, overview FROM movies')
            movies_data = cursor.fetchall()

            # Create movies DataFrame
            self.movies_df = pd.DataFrame(movies_data, columns=['movieId', 'title', 'overview'])

            # Get user and rating data from database
            cursor.execute('SELECT user_id, movie_id, rating FROM ratings')
            ratings_data = cursor.fetchall()
            ratings_df = pd.DataFrame(ratings_data, columns=['userId', 'movieId', 'rating'])

            # Create user-movie matrix
            self.user_movie_matrix = ratings_df.pivot(
                index='userId',
                columns='movieId',
                values='rating'
            ).fillna(0)

            # Create user and movie mappings
            self.user_to_idx = {user: idx for idx, user in enumerate(self.user_movie_matrix.index)}
            self.movie_to_idx = {movie: idx for idx, movie in enumerate(self.user_movie_matrix.columns)}
        finally:
            conn.close()

        # Entrenar modelo basado en contenido
        self._fit_content_based()

        # Entrenar modelo colaborativo
        self._fit_collaborative()

        logger.info("Entrenamiento completado exitosamente")

    def _fit_content_based(self):
        """Entrena el modelo basado en contenido."""
        logger.info("Entrenando modelo basado en contenido...")

        # Combine title and overview for content-based features
        self.movies_df['content'] = self.movies_df['title'] + ' ' + self.movies_df['overview'].fillna('')

        # Create TF-IDF matrix
        self.movie_features = self.vectorizer.fit_transform(self.movies_df['content'])

        # Calculate similarity matrix
        self.movie_similarity = cosine_similarity(self.movie_features)

        logger.info("Modelo basado en contenido entrenado exitosamente")

    def _fit_collaborative(self):
        """Entrena el modelo colaborativo con validación cruzada y early stopping."""
        logger.info("Entrenando modelo colaborativo...")

        # Preparar datos
        dataset = MovieDataset(self.user_movie_matrix)

        # Dividir en train y validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

        # Inicializar modelo
        num_users = len(self.user_to_idx)
        num_movies = len(self.movie_to_idx)
        self.model = NeuralCF(num_users, num_movies, self.embedding_dim).to(self.device)

        # Configuración de entrenamiento
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

        # Entrenamiento
        max_epochs = 50
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        # Listas para almacenar pérdidas y métricas
        self.train_losses = []
        self.val_losses = []
        self.epoch_metrics = []

        for epoch in range(max_epochs):
            # Modo entrenamiento
            self.model.train()
            train_loss = 0
            train_predictions = []
            train_targets = []

            for batch in train_loader:
                user_input = batch['user'].to(self.device)
                movie_input = batch['movie'].to(self.device)
                rating = batch['rating'].to(self.device)

                optimizer.zero_grad()
                output = self.model(user_input, movie_input)
                loss = criterion(output, rating)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_predictions.extend(output.cpu().detach().numpy())
                train_targets.extend(rating.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Modo evaluación
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    user_input = batch['user'].to(self.device)
                    movie_input = batch['movie'].to(self.device)
                    rating = batch['rating'].to(self.device)

                    output = self.model(user_input, movie_input)
                    loss = criterion(output, rating)

                    val_loss += loss.item()
                    val_predictions.extend(output.cpu().numpy())
                    val_targets.extend(rating.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)

            # Calcular métricas
            train_metrics = calculate_metrics(np.array(train_targets), np.array(train_predictions))
            val_metrics = calculate_metrics(np.array(val_targets), np.array(val_predictions))

            # Guardar métricas de la época
            self.epoch_metrics.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })

            # Actualizar learning rate
            scheduler.step(avg_val_loss)

            # Logging detallado
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{max_epochs}")
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Val Loss: {avg_val_loss:.4f}")
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            logger.info("\nTrain Metrics:")
            for metric, value in train_metrics.items():
                logger.info(f"{metric}: {value:.4f}")

            logger.info("\nValidation Metrics:")
            for metric, value in val_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            logger.info(f"{'='*50}\n")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Cargar mejor modelo
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Graficar pérdidas
        self._plot_losses()

        # Guardar métricas en un archivo
        self._save_metrics()

        # Analizar y mostrar las mejores métricas
        self._analyze_best_metrics()

        logger.info("Modelo colaborativo entrenado exitosamente")

    def _plot_losses(self):
        """Grafica las pérdidas de entrenamiento y validación."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig('models/training_losses.png')
        plt.close()

    def _save_metrics(self):
        """Guarda las métricas en un archivo CSV."""
        metrics_df = pd.DataFrame(self.epoch_metrics)
        metrics_df.to_csv('models/training_metrics.csv', index=False)
        logger.info("Métricas guardadas en 'models/training_metrics.csv'")

    def evaluate_model(self, test_data: Dict = None):
        """Evalúa el modelo en el conjunto de test."""
        if test_data is None:
            # Usar el conjunto de validación como test
            dataset = MovieDataset(self.user_movie_matrix)
            _, test_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
            test_loader = DataLoader(test_dataset, batch_size=64)
        else:
            # Implementar lógica para datos de test personalizados
            pass

        self.model.eval()
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for batch in test_loader:
                user_input = batch['user'].to(self.device)
                movie_input = batch['movie'].to(self.device)
                rating = batch['rating'].to(self.device)

                output = self.model(user_input, movie_input)
                test_predictions.extend(output.cpu().numpy())
                test_targets.extend(rating.cpu().numpy())

        # Calcular y mostrar métricas
        test_metrics = calculate_metrics(np.array(test_targets), np.array(test_predictions))

        logger.info("\nTest Metrics:")
        for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.4f}")

        return test_metrics

    def get_content_based_recommendations(self, movie_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Obtiene recomendaciones basadas en contenido para una película."""
        try:
            # Get movie index
            movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]

            # Get similarity scores
            similarity_scores = self.movie_similarity[movie_idx]

            # Get top similar movies
            similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations + 1]
            similar_movies = self.movies_df.iloc[similar_indices]

            # Get movie details
            recommendations = []
            for _, movie in similar_movies.iterrows():
                movie_data = self._get_movie_data(movie['movieId'])
                if movie_data:
                    recommendations.append(movie_data)

            return recommendations
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {str(e)}")
            return []

    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Obtiene recomendaciones colaborativas para un usuario."""
        try:
            # Verificar si el usuario existe en la base de datos
            conn = sqlite3.connect('data/tmdb_movies.db')
            cursor = conn.cursor()
            
            # Verificar si el usuario existe en user_film o ratings
            cursor.execute('''
            SELECT COUNT(*) FROM (
                SELECT user_id FROM user_film WHERE user_id = ?
                UNION
                SELECT user_id FROM ratings WHERE user_id = ?
            )
            ''', (user_id, user_id))
            
            if cursor.fetchone()[0] == 0:
                logger.warning(f"Usuario {user_id} no encontrado en la base de datos")
                return []

            # Obtener el índice del usuario o crear uno nuevo
            if user_id not in self.user_to_idx:
                # Asignar un nuevo índice al usuario
                new_idx = len(self.user_to_idx)
                self.user_to_idx[user_id] = new_idx
                logger.info(f"Usuario {user_id} asignado al índice {new_idx}")

            user_idx = self.user_to_idx[user_id]

            # Obtener películas que el usuario ya ha visto
            cursor.execute('''
            SELECT movie_id FROM user_film WHERE user_id = ?
            ''', (user_id,))
            watched_movies = {row[0] for row in cursor.fetchall()}
            
            # Obtener películas calificadas
            cursor.execute('''
            SELECT movie_id FROM ratings WHERE user_id = ?
            ''', (user_id,))
            rated_movies = {row[0] for row in cursor.fetchall()}
            
            # Combinar películas vistas y calificadas
            excluded_movies = watched_movies.union(rated_movies)
            
            # Obtener todas las películas no vistas/no calificadas
            all_movies = set(self.movie_to_idx.keys())
            available_movies = all_movies - excluded_movies
            
            if not available_movies:
                logger.warning(f"No hay películas disponibles para recomendar al usuario {user_id}")
                return []
            
            # Predecir calificaciones para películas disponibles
            predictions = []
            self.model.eval()
            with torch.no_grad():
                for movie_id in available_movies:
                    movie_idx = self.movie_to_idx.get(movie_id)
                    if movie_idx is not None:
                        user_input = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                        movie_input = torch.tensor([movie_idx], dtype=torch.long).to(self.device)
                        prediction = self.model(user_input, movie_input)
                        predictions.append((movie_id, prediction.item()))
            
            if not predictions:
                logger.warning(f"No se pudieron generar predicciones para el usuario {user_id}")
                return []
            
            # Ordenar predicciones y obtener las mejores
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_movies = [movie_id for movie_id, _ in predictions[:n_recommendations]]
            
            # Obtener detalles de las películas
            recommendations = []
            for movie_id in top_movies:
                movie_data = self._get_movie_data(movie_id)
                if movie_data:
                    recommendations.append(movie_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error al obtener recomendaciones colaborativas: {str(e)}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()

    def get_worst_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Obtiene las peores recomendaciones colaborativas para un usuario."""
        try:
            # Verificar si el usuario existe en la base de datos
            conn = sqlite3.connect('data/tmdb_movies.db')
            cursor = conn.cursor()
            
            # Verificar si el usuario existe en user_film o ratings
            cursor.execute('''
            SELECT COUNT(*) FROM (
                SELECT user_id FROM user_film WHERE user_id = ?
                UNION
                SELECT user_id FROM ratings WHERE user_id = ?
            )
            ''', (user_id, user_id))
            
            if cursor.fetchone()[0] == 0:
                logger.warning(f"Usuario {user_id} no encontrado en la base de datos")
                return []

            # Obtener el índice del usuario o crear uno nuevo
            if user_id not in self.user_to_idx:
                # Asignar un nuevo índice al usuario
                new_idx = len(self.user_to_idx)
                self.user_to_idx[user_id] = new_idx
                logger.info(f"Usuario {user_id} asignado al índice {new_idx}")

            user_idx = self.user_to_idx[user_id]

            # Obtener películas que el usuario ya ha visto
            cursor.execute('''
            SELECT movie_id FROM user_film WHERE user_id = ?
            ''', (user_id,))
            watched_movies = {row[0] for row in cursor.fetchall()}
            
            # Obtener películas calificadas
            cursor.execute('''
            SELECT movie_id FROM ratings WHERE user_id = ?
            ''', (user_id,))
            rated_movies = {row[0] for row in cursor.fetchall()}
            
            # Combinar películas vistas y calificadas
            excluded_movies = watched_movies.union(rated_movies)
            
            # Obtener todas las películas no vistas/no calificadas
            all_movies = set(self.movie_to_idx.keys())
            available_movies = all_movies - excluded_movies
            
            if not available_movies:
                logger.warning(f"No hay películas disponibles para recomendar al usuario {user_id}")
                return []
            
            # Predecir calificaciones para películas disponibles
            predictions = []
            self.model.eval()
            with torch.no_grad():
                for movie_id in available_movies:
                    movie_idx = self.movie_to_idx.get(movie_id)
                    if movie_idx is not None:
                        user_input = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                        movie_input = torch.tensor([movie_idx], dtype=torch.long).to(self.device)
                        prediction = self.model(user_input, movie_input)
                        predictions.append((movie_id, prediction.item()))
            
            if not predictions:
                logger.warning(f"No se pudieron generar predicciones para el usuario {user_id}")
                return []
            
            # Ordenar predicciones y obtener las peores
            predictions.sort(key=lambda x: x[1])  # Ordenar por calificación (ascendente)
            worst_movies = [movie_id for movie_id, _ in predictions[:n_recommendations]]
            
            # Obtener detalles de las películas
            recommendations = []
            for movie_id in worst_movies:
                movie_data = self._get_movie_data(movie_id)
                if movie_data:
                    recommendations.append(movie_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error al obtener peores recomendaciones: {str(e)}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()

    def _analyze_best_metrics(self):
        """Analiza y muestra las mejores métricas alcanzadas durante el entrenamiento."""
        if not self.epoch_metrics:
            logger.warning("No hay métricas para analizar")
            return

        # Encontrar la época con mejor val_loss
        best_epoch_idx = np.argmin([m['val_loss'] for m in self.epoch_metrics])
        best_epoch = self.epoch_metrics[best_epoch_idx]

        logger.info("\n" + "="*50)
        logger.info("MEJORES MÉTRICAS ALCANZADAS")
        logger.info("="*50)
        logger.info(f"Época: {best_epoch['epoch']}")
        logger.info(f"Train Loss: {best_epoch['train_loss']:.4f}")
        logger.info(f"Val Loss: {best_epoch['val_loss']:.4f}")

        logger.info("\nMétricas de Entrenamiento:")
        for metric, value in best_epoch['train_metrics'].items():
            logger.info(f"{metric}: {value:.4f}")

        logger.info("\nMétricas de Validación:")
        for metric, value in best_epoch['val_metrics'].items():
            logger.info(f"{metric}: {value:.4f}")

        # Encontrar las mejores métricas individuales
        logger.info("\nMEJORES VALORES POR MÉTRICA:")
        logger.info("-"*30)

        metrics_to_track = [
            'MSE', 'RMSE', 'Accuracy', 'Precision', 'Recall', 'F1-Score',
            'Precision@10', 'Recall@10', 'NDCG@10', 'MAP@10', 'HitRate@10'
        ]

        for metric in metrics_to_track:
            # Para MSE y RMSE, queremos el mínimo
            if metric in ['MSE', 'RMSE']:
                best_value = min(m['val_metrics'][metric] for m in self.epoch_metrics)
                best_epoch = next(m['epoch'] for m in self.epoch_metrics
                                if m['val_metrics'][metric] == best_value)
                logger.info(f"{metric}: {best_value:.4f} (Época {best_epoch})")
            # Para el resto, queremos el máximo
            else:
                best_value = max(m['val_metrics'][metric] for m in self.epoch_metrics)
                best_epoch = next(m['epoch'] for m in self.epoch_metrics
                                if m['val_metrics'][metric] == best_value)
                logger.info(f"{metric}: {best_value:.4f} (Época {best_epoch})")

        logger.info("="*50 + "\n")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        torch.cuda.empty_cache()