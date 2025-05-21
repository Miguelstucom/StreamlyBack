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
        """Entrena el modelo colaborativo."""
        logger.info("Entrenando modelo colaborativo...")

        # Create dataset
        dataset = MovieDataset(self.user_movie_matrix)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Initialize model
        num_users = len(self.user_to_idx)
        num_movies = len(self.movie_to_idx)
        self.model = NeuralCF(num_users, num_movies, self.embedding_dim).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        self.model.train()
        for epoch in range(10):
            total_loss = 0
            for batch in dataloader:
                user_input = batch['user'].to(self.device)
                movie_input = batch['movie'].to(self.device)
                rating = batch['rating'].to(self.device)

                optimizer.zero_grad()
                output = self.model(user_input, movie_input)
                loss = criterion(output, rating)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

        logger.info("Modelo colaborativo entrenado exitosamente")

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
            # Get user index
            user_idx = self.user_to_idx.get(user_id)
            if user_idx is None:
                return []

            # Get user ratings
            user_ratings = self.user_movie_matrix.iloc[user_idx]

            # Get unrated movies
            unrated_movies = user_ratings[user_ratings == 0].index

            # Predict ratings for unrated movies
            predictions = []
            self.model.eval()
            with torch.no_grad():
                for movie_id in unrated_movies:
                    movie_idx = self.movie_to_idx.get(movie_id)
                    if movie_idx is not None:
                        user_input = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                        movie_input = torch.tensor([movie_idx], dtype=torch.long).to(self.device)
                        prediction = self.model(user_input, movie_input)
                        predictions.append((movie_id, prediction.item()))

            # Sort predictions and get top recommendations
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_movies = [movie_id for movie_id, _ in predictions[:n_recommendations]]

            # Get movie details
            recommendations = []
            for movie_id in top_movies:
                movie_data = self._get_movie_data(movie_id)
                if movie_data:
                    recommendations.append(movie_data)

            return recommendations
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {str(e)}")
            return []

    def get_worst_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Obtiene las peores recomendaciones colaborativas para un usuario."""
        try:
            # Get user index
            user_idx = self.user_to_idx.get(user_id)
            if user_idx is None:
                return []

            # Get user ratings
            user_ratings = self.user_movie_matrix.iloc[user_idx]

            # Get unrated movies
            unrated_movies = user_ratings[user_ratings == 0].index

            # Predict ratings for unrated movies
            predictions = []
            self.model.eval()
            with torch.no_grad():
                for movie_id in unrated_movies:
                    movie_idx = self.movie_to_idx.get(movie_id)
                    if movie_idx is not None:
                        user_input = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                        movie_input = torch.tensor([movie_idx], dtype=torch.long).to(self.device)
                        prediction = self.model(user_input, movie_input)
                        predictions.append((movie_id, prediction.item()))

            # Sort predictions and get worst recommendations
            predictions.sort(key=lambda x: x[1])
            worst_movies = [movie_id for movie_id, _ in predictions[:n_recommendations]]

            # Get movie details
            recommendations = []
            for movie_id in worst_movies:
                movie_data = self._get_movie_data(movie_id)
                if movie_data:
                    recommendations.append(movie_data)

            return recommendations
        except Exception as e:
            logger.error(f"Error getting worst collaborative recommendations: {str(e)}")
            return []

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        torch.cuda.empty_cache()