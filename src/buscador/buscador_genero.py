#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import sqlite3
import time
import pickle
from pathlib import Path
from typing import List, Tuple, Set

# --- Configuración ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = PROJECT_ROOT / "src" / "data"
DATA_FILE = DB_DIR / "tmdb_movies.db"

OUTPUT_DIR = Path(__file__).resolve().parent
PKL_FILE = OUTPUT_DIR / "genre_engine.pkl" # Nombre original del PKL
METRICS_TXT = OUTPUT_DIR / "genre_engine_metrics.txt"

# --- Utilidades (normaliza no es necesaria aquí si solo se guardan datos crudos) ---

# --- Carga de datos de la base de datos ---
SQL = """
SELECT m.title, GROUP_CONCAT(g.name, ', ')
FROM movies m
LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
LEFT JOIN genres g ON mg.genre_id = g.id
GROUP BY m.movie_id, m.title 
ORDER BY m.movie_id 
""" # Añadido m.title al GROUP BY y ORDER BY para consistencia

def leer_bd(db_path: Path) -> Tuple[List[str], List[str], Set[str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(SQL)
    rows = cur.fetchall()
    titles = [r[0] for r in rows if r[0] is not None] # Filtrar títulos None
    # Asegurar que genres tiene la misma longitud que titles
    genres_map = {r[0]: (r[1] or "") for r in rows if r[0] is not None}
    genres = [genres_map[title] for title in titles]


    cur.execute("SELECT DISTINCT name FROM genres WHERE name IS NOT NULL")
    catalogo = {r[0].lower() for r in cur.fetchall() if r[0]}
    conn.close()
    return titles, genres, catalogo

# --- Construcción del Motor ---
def construir_motor():
    print(f"⚙️ Construyendo motor de GÉNEROS desde: {DATA_FILE}")
    print(f"   Guardando PKL en: {PKL_FILE}")
    t0 = time.perf_counter()

    if not DATA_FILE.exists():
        print(f"❌ Error: El archivo de base de datos '{DATA_FILE}' no se encontró.")
        return

    titles, genres, catalogo = leer_bd(DATA_FILE)
    
    if not titles:
        print(f"❌ No se encontraron títulos en la base de datos.")
        return

    engine_data = dict(
        titles=titles, # Se guardará como 'titles_orig' en el agente
        genres=genres, # Se guardará como 'genres_orig' en el agente
        catalogo=list(catalogo) # Se guardará como 'all_genres' en el agente
    )
    with open(PKL_FILE, "wb") as f:
        pickle.dump(engine_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    dt = time.perf_counter() - t0
    Path(METRICS_TXT).write_text(
        f"Películas: {len(titles)}\n"
        f"Géneros únicos en catálogo: {len(catalogo)}\n"
        f"Tiempo de construcción: {dt:.2f}s\n"
        f"Tamaño {PKL_FILE.name}: {PKL_FILE.stat().st_size/1e6:.1f} MB\n",
        encoding="utf-8")
    print(f"✅ Motor de GÉNEROS guardado en {PKL_FILE} ({dt:.1f}s). Métricas en {METRICS_TXT}")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Asegúrate que '{DATA_FILE.name}' existe en '{DB_DIR}'.")
    construir_motor()