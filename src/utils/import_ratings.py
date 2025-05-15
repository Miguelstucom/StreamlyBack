import pandas as pd
import sqlite3

# Leer ratings
ratings = pd.read_csv('data/ratings.csv')

# Conectar a la base de datos
conn = sqlite3.connect('data/tmdb_movies.db')
cursor = conn.cursor()

# Crear tabla ratings si no existe
cursor.execute('''
CREATE TABLE IF NOT EXISTS ratings (
    user_id INTEGER,
    movie_id INTEGER,
    rating REAL,
    timestamp INTEGER,
    description TEXT,
    PRIMARY KEY (user_id, movie_id),
    FOREIGN KEY (user_id) REFERENCES users(userId),
    FOREIGN KEY (movie_id) REFERENCES movies(id)
)
''')

# Insertar ratings
for _, row in ratings.iterrows():
    cursor.execute('''
    INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, timestamp, description)
    VALUES (?, ?, ?, ?, ?)
    ''', (row['userId'], row['movieId'], row['rating'], row['timestamp'], row['description']))

conn.commit()
conn.close()
print('Ratings importados correctamente a tmdb_movies.db.') 