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
from torch.utils.data import Dataset, DataLoader
import sqlite3

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

def precision_at_k(y_true, y_pred, k=10):
    """Calcula la precisión@k."""
    top_k_idx = np.argsort(y_pred)[-k:]
    return np.mean(y_true[top_k_idx] >= 4)

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
        finally:
            conn.close()
        
        # Get user and rating data from CSV
        self.user_movie_matrix = data['user_movie_matrix']
        self.user_to_idx = data['user_to_idx']
        self.movie_to_idx = data['movie_to_idx']

        # Entrenar modelo basado en contenido
        self._fit_content_based()

        # Entrenar modelo colaborativo
        self._fit_collaborative()

        logger.info("Entrenamiento completado exitosamente")

    def _fit_content_based(self):
        """Entrena el modelo basado en contenido."""
        logger.info("Entrenando modelo basado en contenido...")

        # Preparar características de texto
        text_features = []
        for _, row in self.movies_df.iterrows():
            # Combinar título y descripción
            features = f"{row['title']} {row['overview']}"
            text_features.append(features)

        # Convertir características de texto a matriz TF-IDF
        self.movie_features = self.vectorizer.fit_transform(text_features)
        self.movie_similarity = cosine_similarity(self.movie_features)

        logger.info(f"Modelo basado en contenido entrenado con {len(text_features)} películas")

    def _fit_collaborative(self):
        """Entrena el modelo colaborativo usando una red neuronal."""
        logger.info("Entrenando modelo colaborativo...")

        # Dividir datos en entrenamiento y prueba (80-20)
        np.random.seed(42)
        mask = np.random.rand(self.user_movie_matrix.shape[0], self.user_movie_matrix.shape[1]) < 0.8
        train_matrix = self.user_movie_matrix.copy()
        test_matrix = self.user_movie_matrix.copy()
        train_matrix[~mask] = 0
        test_matrix[mask] = 0

        # Crear datasets
        train_dataset = MovieDataset(train_matrix)
        test_dataset = MovieDataset(test_matrix)

        # Crear dataloaders
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1024)

        # Inicializar modelo
        num_users = len(self.user_to_idx)
        num_movies = len(self.movie_to_idx)
        self.model = NeuralCF(num_users, num_movies, self.embedding_dim).to(self.device)
        
        # Definir optimizador y función de pérdida
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Entrenamiento
        best_rmse = float('inf')
        best_metrics = None
        patience = 5
        patience_counter = 0

        for epoch in range(10):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                user = batch['user'].to(self.device)
                movie = batch['movie'].to(self.device)
                rating = batch['rating'].to(self.device)

                optimizer.zero_grad()
                prediction = self.model(user, movie)
                loss = criterion(prediction, rating)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluación
            self.model.eval()
            predictions = []
            actuals = []
            with torch.no_grad():
                for batch in test_loader:
                    user = batch['user'].to(self.device)
                    movie = batch['movie'].to(self.device)
                    rating = batch['rating'].to(self.device)
                    
                    prediction = self.model(user, movie)
                    predictions.extend(prediction.cpu().numpy())
                    actuals.extend(rating.cpu().numpy())

            predictions = np.array(predictions)
            actuals = np.array(actuals)

            # Calcular métricas
            metrics = {
                'mse': mean_squared_error(actuals, predictions),
                'rmse': np.sqrt(mean_squared_error(actuals, predictions))
            }

            # Métricas de clasificación
            actuals_binary = (actuals >= 4).astype(int)
            predictions_binary = (predictions >= 4).astype(int)

            if np.sum(actuals_binary) > 0:
                metrics.update({
                    'accuracy': accuracy_score(actuals_binary, predictions_binary),
                    'precision': precision_score(actuals_binary, predictions_binary, zero_division=0),
                    'recall': recall_score(actuals_binary, predictions_binary, zero_division=0),
                    'f1_score': f1_score(actuals_binary, predictions_binary, zero_division=0)
                })

            # Precision@k
            if len(actuals) >= 10:
                metrics['precision@10'] = precision_at_k(actuals, predictions, k=10)
            if len(actuals) >= 5:
                metrics['precision@5'] = precision_at_k(actuals, predictions, k=5)

            logger.info(f"Epoch {epoch + 1}/50 - Loss: {total_loss/len(train_loader):.4f} - RMSE: {metrics['rmse']:.4f}")

            # Early stopping
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_metrics = metrics
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break

        self.metrics = best_metrics
        logger.info("Modelo colaborativo entrenado exitosamente")
        logger.info("Métricas finales:")
        for metric, value in best_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    def get_content_based_recommendations(self, movie_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Obtiene recomendaciones basadas en contenido para una película específica."""
        if self.movie_similarity is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer recomendaciones")

        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        similarity_scores = self.movie_similarity[movie_idx]
        similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations + 1]

        recommendations = []
        for idx in similar_indices:
            movie_id = self.movies_df.iloc[idx]['movieId']
            movie_data = self._get_movie_data(movie_id)
            if movie_data:
                movie_data['similarity_score'] = float(similarity_scores[idx])
                recommendations.append(movie_data)

        return recommendations

    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Obtiene recomendaciones colaborativas para un usuario específico."""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer recomendaciones")

        if user_id not in self.user_to_idx:
            raise ValueError(f"Usuario {user_id} no encontrado en el sistema")

        user_idx = self.user_to_idx[user_id]
        
        # Obtener películas que el usuario no ha calificado
        user_movies = set(self.user_movie_matrix.columns[self.user_movie_matrix.iloc[user_idx] > 0])
        all_movies = set(self.user_movie_matrix.columns)
        unwatched_movies = list(all_movies - user_movies)

        # Preparar datos para predicción
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for movie_id in unwatched_movies:
                movie_idx = self.movie_to_idx[movie_id]
                user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                movie_tensor = torch.tensor([movie_idx], dtype=torch.long).to(self.device)
                
                prediction = self.model(user_tensor, movie_tensor)
                predictions.append(prediction.item())

        # Obtener las mejores predicciones
        predictions = np.array(predictions)
        top_indices = predictions.argsort()[::-1][:n_recommendations]

        # Crear lista de recomendaciones
        recommendations = []
        for idx in top_indices:
            movie_id = unwatched_movies[idx]
            movie_data = self._get_movie_data(movie_id)
            if movie_data:
                movie_data['predicted_rating'] = float(predictions[idx])
                recommendations.append(movie_data)

        return recommendations

    def get_worst_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Obtiene las peores recomendaciones colaborativas para un usuario específico."""
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer recomendaciones")

        if user_id not in self.user_to_idx:
            raise ValueError(f"Usuario {user_id} no encontrado en el sistema")

        user_idx = self.user_to_idx[user_id]

        # Obtener películas no vistas por el usuario
        user_movies = set(self.user_movie_matrix.columns[self.user_movie_matrix.iloc[user_idx] > 0])
        all_movies = set(self.user_movie_matrix.columns)
        unwatched_movies = list(all_movies - user_movies)

        # Predecir ratings para películas no vistas
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for movie_id in unwatched_movies:
                movie_idx = self.movie_to_idx[movie_id]
                user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                movie_tensor = torch.tensor([movie_idx], dtype=torch.long).to(self.device)

                prediction = self.model(user_tensor, movie_tensor)
                predictions.append(prediction.item())

        # Obtener las peores predicciones (menor rating)
        predictions = np.array(predictions)
        worst_indices = predictions.argsort()[:n_recommendations]  # orden ascendente

        # Crear lista de peores recomendaciones
        worst_recommendations = []
        for idx in worst_indices:
            movie_id = unwatched_movies[idx]
            movie_data = self._get_movie_data(movie_id)
            if movie_data:
                movie_data['predicted_rating'] = float(predictions[idx])
                worst_recommendations.append(movie_data)

        return worst_recommendations

    def __del__(self):
        """Cerrar la conexión a la base de datos al destruir el objeto."""
        if self.movie_data_cache:
            self.movie_data_cache.clear()