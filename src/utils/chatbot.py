import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from googletrans import Translator
import re

logger = logging.getLogger(__name__)

class MovieChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.translator = Translator()
        
        # Load model and tokenizer
        logger.info("Loading language model...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium",
            device_map=self.device,
            use_safetensors=True
        )
        
        # Load movie data
        self._load_movie_data()
        
    def _load_movie_data(self):
        """Load all movie data from SQLite database."""
        conn = sqlite3.connect('data/tmdb_movies.db')
        try:
            # Load movies
            self.movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
            
            # Load genres
            genres_df = pd.read_sql_query("""
                SELECT m.movie_id, GROUP_CONCAT(g.name) as genres
                FROM movies m
                JOIN movie_genres mg ON m.movie_id = mg.movie_id
                JOIN genres g ON mg.genre_id = g.id
                GROUP BY m.movie_id
            """, conn)
            
            # Load production companies
            companies_df = pd.read_sql_query("""
                SELECT m.movie_id, GROUP_CONCAT(pc.name) as companies
                FROM movies m
                JOIN movie_production_companies mpc ON m.movie_id = mpc.movie_id
                JOIN production_companies pc ON mpc.company_id = pc.id
                GROUP BY m.movie_id
            """, conn)
            
            # Load production countries
            countries_df = pd.read_sql_query("""
                SELECT m.movie_id, GROUP_CONCAT(pc.name) as countries
                FROM movies m
                JOIN movie_production_countries mpc ON m.movie_id = mpc.movie_id
                JOIN production_countries pc ON mpc.country_code = pc.iso_3166_1
                GROUP BY m.movie_id
            """, conn)
            
            # Load spoken languages
            languages_df = pd.read_sql_query("""
                SELECT m.movie_id, GROUP_CONCAT(sl.name) as languages
                FROM movies m
                JOIN movie_spoken_languages msl ON m.movie_id = msl.movie_id
                JOIN spoken_languages sl ON msl.language_code = sl.iso_639_1
                GROUP BY m.movie_id
            """, conn)
            
            # Merge all data
            self.movies_df = self.movies_df.merge(genres_df, on='movie_id', how='left')
            self.movies_df = self.movies_df.merge(companies_df, on='movie_id', how='left')
            self.movies_df = self.movies_df.merge(countries_df, on='movie_id', how='left')
            self.movies_df = self.movies_df.merge(languages_df, on='movie_id', how='left')
            
            # Load ratings for user reviews
            self.ratings_df = pd.read_csv('data/ratings.csv')
            
            # Rename TMDB vote columns to match our filter names
            self.movies_df = self.movies_df.rename(columns={
                'vote_count': 'vote_count',
                'vote_average': 'average_rating'
            })
            
        finally:
            conn.close()
    
    def _translate_to_english(self, text: str) -> str:
        """Translate text to English."""
        try:
            return self.translator.translate(text, dest='en').text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text
    
    def _translate_to_spanish(self, text: str) -> str:
        """Translate text to Spanish."""
        try:
            return self.translator.translate(text, dest='es').text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text
    
    def _extract_filters(self, query: str) -> Dict:
        """Extract filters from the query."""
        filters = {
            'min_votes': 0,
            'max_votes': None,
            'min_rating': 0,
            'max_rating': None,
            'year': None,
            'genre': None,
            'sort_by': None,  # 'best', 'worst', 'most_votes', 'least_votes'
            'limit': 3  # Default to 3 movies
        }
        
        # Extract minimum votes
        min_votes_match = re.search(r'más de (\d+)\s*(?:votos|ratings)', query.lower())
        if min_votes_match:
            filters['min_votes'] = int(min_votes_match.group(1))
            filters['sort_by'] = 'most_votes'
        
        # Extract maximum votes
        max_votes_match = re.search(r'menos de (\d+)\s*(?:votos|ratings)', query.lower())
        if max_votes_match:
            filters['max_votes'] = int(max_votes_match.group(1))
            filters['sort_by'] = 'least_votes'
        
        # Extract minimum rating
        min_rating_match = re.search(r'más de (\d+(?:\.\d+)?)\s*estrellas', query.lower())
        if min_rating_match:
            filters['min_rating'] = float(min_rating_match.group(1))
        
        # Extract maximum rating
        max_rating_match = re.search(r'menos de (\d+(?:\.\d+)?)\s*estrellas', query.lower())
        if max_rating_match:
            filters['max_rating'] = float(max_rating_match.group(1))
        
        # Extract year
        year_match = re.search(r'año\s*(\d{4})', query.lower())
        if year_match:
            filters['year'] = int(year_match.group(1))
        
        # Extract genre - look for various patterns
        genre_patterns = [
            r'género\s+(\w+)(?:\s+con|\s+que|$)',  # "género Drama"
            r'película\s+de\s+(\w+)(?:\s+con|\s+que|$)',  # "película de Drama"
            r'películas\s+de\s+(\w+)(?:\s+con|\s+que|$)',  # "películas de Drama"
            r'de\s+(\w+)(?:\s+con|\s+que|$)',  # "de Drama"
            r'qué\s+películas\s+hay\s+del\s+género\s+(\w+)',  # "qué películas hay del género Drama"
            r'películas\s+del\s+género\s+(\w+)',  # "películas del género Drama"
            r'películas\s+de\s+género\s+(\w+)',  # "películas de género Drama"
            r'género\s+(\w+)'  # "género Drama"
        ]
        
        for pattern in genre_patterns:
            genre_match = re.search(pattern, query.lower())
            if genre_match:
                filters['genre'] = genre_match.group(1).strip()
                break
        
        # Check for best/worst rated movies
        if 'mejor valoradas' in query.lower() or 'mejor valorados' in query.lower():
            filters['sort_by'] = 'best'
        elif 'peor valoradas' in query.lower() or 'peor valorados' in query.lower():
            filters['sort_by'] = 'worst'
        
        return filters
    
    def _get_movies_by_filters(self, filters: Dict) -> pd.DataFrame:
        """Get movies based on filters."""
        filtered_movies = self.movies_df.copy()
        
        if filters['min_votes'] > 0:
            filtered_movies = filtered_movies[filtered_movies['vote_count'] >= filters['min_votes']]
        
        if filters['max_votes'] is not None:
            filtered_movies = filtered_movies[filtered_movies['vote_count'] <= filters['max_votes']]
        
        if filters['min_rating'] > 0:
            filtered_movies = filtered_movies[filtered_movies['average_rating'] >= filters['min_rating']]
        
        if filters['max_rating'] is not None:
            filtered_movies = filtered_movies[filtered_movies['average_rating'] <= filters['max_rating']]
        
        if filters['year']:
            filtered_movies = filtered_movies[filtered_movies['release_date'].str.startswith(str(filters['year']))]
        
        if filters['genre']:
            # Convert genre to title case for better matching
            genre = filters['genre'].title()
            # Split genres string and check if the genre is in the list
            filtered_movies = filtered_movies[filtered_movies['genres'].apply(
                lambda x: genre in x.split(', ') if isinstance(x, str) else False
            )]
        
        # Apply sorting based on sort_by
        if filters['sort_by'] == 'best':
            filtered_movies = filtered_movies.sort_values('average_rating', ascending=False)
        elif filters['sort_by'] == 'worst':
            filtered_movies = filtered_movies.sort_values('average_rating', ascending=True)
        elif filters['sort_by'] == 'most_votes':
            filtered_movies = filtered_movies.sort_values('vote_count', ascending=False)
        elif filters['sort_by'] == 'least_votes':
            filtered_movies = filtered_movies.sort_values('vote_count', ascending=True)
        
        return filtered_movies
    
    def _format_movie_info(self, movie: pd.Series) -> str:
        """Format movie information into a readable string."""
        # Get the original title from the database
        original_title = movie['title']
        
        info = [
            f"Título: {original_title}",
            f"Año: {movie['release_date'][:4]}",
            f"Géneros: {movie['genres']}",
            f"Calificación TMDB: {movie['average_rating']:.1f}/10",
            f"Número de votos TMDB: {movie['vote_count']}",
            f"Descripción: {movie['overview']}"
        ]
        return "\n".join(info)
    
    def _is_greeting(self, query: str) -> bool:
        """Check if the query is a greeting."""
        greetings = ['hola', 'buenos días', 'buenas tardes', 'buenas noches', 'saludos', 'hey', 'hi', 'hello']
        return any(greeting in query.lower() for greeting in greetings)
    
    def _is_thanks(self, query: str) -> bool:
        """Check if the query is a thank you message."""
        thanks = ['gracias', 'muchas gracias', 'thank you', 'thanks']
        return any(thank in query.lower() for thank in thanks)
    
    def _is_farewell(self, query: str) -> bool:
        """Check if the query is a farewell."""
        farewells = ['adiós', 'hasta luego', 'hasta pronto', 'bye', 'goodbye']
        return any(farewell in query.lower() for farewell in farewells)
    
    def _extract_movie_title(self, query: str) -> Optional[str]:
        """Extract movie title from query using multiple patterns."""
        # Try different patterns to extract the movie title
        patterns = [
            r'película\s+(.+?)(?:\s+con|\s+que|$)',  # "película X"
            r'sobre\s+(.+?)(?:\s+con|\s+que|$)',     # "sobre X"
            r'información\s+(?:de|sobre)\s+(.+?)(?:\s+con|\s+que|$)',  # "información de X"
            r'info\s+(?:de|sobre)\s+(.+?)(?:\s+con|\s+que|$)',        # "info de X"
            r'buscar\s+(.+?)(?:\s+con|\s+que|$)',    # "buscar X"
            r'(.+?)(?:\s+película|\s+film|\s+movie)',  # "X película"
            r'dime toda la información de (.+?)(?:\s+con|\s+que|$)',  # "dime toda la información de X"
            r'de que año es (.+?)(?:\s+con|\s+que|$)',  # "de que año es X"
            r'cuál es el año de (.+?)(?:\s+con|\s+que|$)',  # "cuál es el año de X"
            r'año de (.+?)(?:\s+con|\s+que|$)'  # "año de X"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, try to extract the last word or phrase
        words = query.lower().split()
        if len(words) >= 2:
            # Try to find the movie title in the last 1-3 words
            for i in range(min(3, len(words))):
                potential_title = ' '.join(words[-(i+1):])
                if potential_title and len(potential_title) > 2:  # Avoid single letters
                    return potential_title
        
        return None

    def _get_suggestions(self) -> str:
        """Get a list of example queries."""
        suggestions = [
            "¿Cuáles son las películas mejor valoradas?",
            "¿Dime las películas con menos de 1000 ratings?",
            "¿Qué películas hay del género Drama?",
            "¿De qué año es Jumanji?",
            "¿Dime toda la información de The Godfather?"
        ]
        return "\n".join([f"- {s}" for s in suggestions])

    def process_query(self, query: str) -> str:
        """Process a natural language query about movies."""
        try:
            # Handle greetings
            if self._is_greeting(query):
                return "¡Hola! Soy tu asistente de películas. ¿En qué puedo ayudarte hoy?"
            
            # Handle thank you messages
            if self._is_thanks(query):
                return "¡De nada! Estoy aquí para ayudarte. ¿Hay algo más en lo que pueda asistirte?"
            
            # Handle farewells
            if self._is_farewell(query):
                return "¡Hasta luego! Que disfrutes de tus películas."
            
            # First, try to extract a movie title
            movie_title = self._extract_movie_title(query)
            if movie_title:
                # Search for the movie
                movie = self.movies_df[
                    self.movies_df['title'].str.contains(movie_title, case=False, na=False, regex=False)
                ]
                
                if not movie.empty:
                    # If the query is specifically about the year
                    if 'año' in query.lower():
                        return f"La película '{movie.iloc[0]['title']}' es del año {movie.iloc[0]['release_date'][:4]}."
                    # Otherwise return full information
                    return self._format_movie_info(movie.iloc[0])
            
            # If no movie was found or no title was extracted, continue with other query types
            filters = self._extract_filters(query)
            
            # Handle rating-based queries
            if filters['sort_by'] in ['best', 'worst', 'most_votes', 'least_votes']:
                filtered_movies = self._get_movies_by_filters(filters)
                if filtered_movies.empty:
                    return "Lo siento, no encontré películas que coincidan con tus criterios."
                
                # Get top 3 movies
                top_movies = filtered_movies.head(3)
                
                if filters['sort_by'] == 'best':
                    response = "Las 3 películas mejor valoradas:\n\n"
                elif filters['sort_by'] == 'worst':
                    response = "Las 3 películas peor valoradas:\n\n"
                elif filters['sort_by'] == 'most_votes':
                    response = f"Las 3 películas con más votos (más de {filters['min_votes']}):\n\n"
                elif filters['sort_by'] == 'least_votes':
                    response = f"Las 3 películas con menos votos (menos de {filters['max_votes']}):\n\n"
                
                for _, movie in top_movies.iterrows():
                    response += self._format_movie_info(movie) + "\n\n"
                return response
            
            # Check if this is a genre query
            elif filters['genre']:
                filtered_movies = self._get_movies_by_filters(filters)
                if filtered_movies.empty:
                    return f"No encontré películas del género {filters['genre']}."
                
                # Sort by rating and get top 3
                top_movies = filtered_movies.sort_values('average_rating', ascending=False).head(3)
                
                response = f"Las 3 mejores películas de {filters['genre']}:\n\n"
                for _, movie in top_movies.iterrows():
                    response += self._format_movie_info(movie) + "\n\n"
                return response
            
            # Process different types of queries
            elif 'recomendar' in query.lower() or 'recomiend' in query.lower():
                # Handle recommendation queries
                filtered_movies = self._get_movies_by_filters(filters)
                if filtered_movies.empty:
                    return "Lo siento, no encontré películas que coincidan con tus criterios."
                
                # Sort by rating and get top 3
                top_movies = filtered_movies.sort_values('average_rating', ascending=False).head(3)
                
                response = "Aquí tienes algunas recomendaciones:\n\n"
                for _, movie in top_movies.iterrows():
                    response += self._format_movie_info(movie) + "\n\n"
                return response
                
            elif 'año' in query.lower() and filters['year']:
                # Handle year queries
                filtered_movies = self._get_movies_by_filters(filters)
                if filtered_movies.empty:
                    return f"No encontré películas del año {filters['year']}."
                
                # Sort by rating and get top 3
                top_movies = filtered_movies.sort_values('average_rating', ascending=False).head(3)
                
                response = f"Las 3 mejores películas del año {filters['year']}:\n\n"
                for _, movie in top_movies.iterrows():
                    response += self._format_movie_info(movie) + "\n\n"
                return response
            
            else:
                # If no specific query type was matched, return suggestions
                return f"Lo siento, no he entendido tu pregunta. Quizás quieras decir:\n{self._get_suggestions()}"
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Lo siento, hubo un error procesando tu consulta. Quizás quieras decir:\n{self._get_suggestions()}" 