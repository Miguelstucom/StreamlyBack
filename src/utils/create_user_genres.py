import pandas as pd
import sqlite3

# Leer usuarios
users = pd.read_csv('data/users.csv')

# Conectar a la base de datos
conn = sqlite3.connect('data/tmdb_movies.db')
cursor = conn.cursor()

# Crear tabla intermedia user_genres (borrar si existe)
cursor.execute('DROP TABLE IF EXISTS user_genres')
cursor.execute('''
CREATE TABLE user_genres (
    user_id INTEGER,
    genre_id INTEGER,
    PRIMARY KEY (user_id, genre_id),
    FOREIGN KEY (user_id) REFERENCES users(userId),
    FOREIGN KEY (genre_id) REFERENCES genres(id)
)
''')

# Obtener todos los géneros existentes en la tabla genres
cursor.execute('SELECT id, name FROM genres')
genre_map = {name.lower(): gid for gid, name in cursor.fetchall()}

# Mapeo de géneros no encontrados a sus equivalentes
genre_aliases = {
    'sci-fi': 'Science Fiction',
    'children': 'Family',
    'musical': 'Music'
}

# Insertar relaciones usuario-genero
missing_genres = set()
for _, row in users.iterrows():
    user_id = row['userId']
    if pd.isna(row['preferred_genres']):
        continue
    genres = [g.strip().lower() for g in str(row['preferred_genres']).split('|') if g.strip()]
    for genre in genres:
        # Usar el alias si existe, de lo contrario usar el género original
        genre_name = genre_aliases.get(genre, genre)
        genre_id = genre_map.get(genre_name.lower())
        if genre_id:
            cursor.execute('INSERT OR IGNORE INTO user_genres (user_id, genre_id) VALUES (?, ?)', (user_id, genre_id))
        else:
            missing_genres.add(genre)

conn.commit()
conn.close()
print('Tabla user_genres creada y poblada correctamente.')
if missing_genres:
    print('Géneros no encontrados en la tabla genres:', missing_genres) 