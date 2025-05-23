#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import sqlite3
import time
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Configuración ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = PROJECT_ROOT / "src" / "data"
DATA_FILE = DB_DIR / "tmdb_movies.db"

OUTPUT_DIR = Path(__file__).resolve().parent
ENGINE_PKL_FILE = OUTPUT_DIR / "st_overview_engine.pkl" # Nombre original del PKL
METRICS_TXT_FILE = OUTPUT_DIR / "st_overview_metrics.txt"

DB_TABLE_NAME = "movies"
ID_COLUMN = "movie_id"
TITLE_COLUMN = "title"
OVERVIEW_COLUMN = "overview"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # O el modelo que desees

# --- Normalización de Texto ---
def normalize_text_for_embedding(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return text.strip().lower() # Simple normalización para ST

# --- Construcción del Motor ---
def build_engine():
    print(f"⚙️ Construyendo motor de SINOPSIS (basado en '{OVERVIEW_COLUMN}') desde: {DATA_FILE}")
    print(f"   Guardando PKL en: {ENGINE_PKL_FILE}")
    t_start = time.perf_counter()

    if not DATA_FILE.exists():
        print(f"❌ Error: El archivo de base de datos '{DATA_FILE}' no se encontró.")
        return

    movie_ids_list = []
    titles_for_display_list = []
    overviews_for_embedding_list = []

    try:
        with sqlite3.connect(DATA_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = f"SELECT {ID_COLUMN}, {TITLE_COLUMN}, {OVERVIEW_COLUMN} FROM {DB_TABLE_NAME} WHERE {OVERVIEW_COLUMN} IS NOT NULL AND {OVERVIEW_COLUMN} != ''"
            print(f"Ejecutando consulta: {query}")
            cursor.execute(query)
            rows = cursor.fetchall()

            if not rows:
                print(f"❌ No se encontraron filas con overviews válidos en la tabla '{DB_TABLE_NAME}'.")
                return

            for row in rows:
                movie_id = row[ID_COLUMN]
                title = str(row[TITLE_COLUMN]) if row[TITLE_COLUMN] else "Título Desconocido"
                overview = str(row[OVERVIEW_COLUMN]) if row[OVERVIEW_COLUMN] else ""

                if overview.strip(): # Doble chequeo, aunque la query ya filtra
                    movie_ids_list.append(movie_id)
                    titles_for_display_list.append(title)
                    overviews_for_embedding_list.append(normalize_text_for_embedding(overview))
    except sqlite3.Error as e:
        print(f"❌ Error de SQLite al leer datos: {e}")
        return

    if not overviews_for_embedding_list:
        print("❌ No se encontraron overviews válidos para procesar. El motor no puede ser construido.")
        return

    print(f"Procesando {len(overviews_for_embedding_list)} overviews para embeddings...")
    print(f"Cargando modelo SentenceTransformer: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generando embeddings para los overviews (esto puede tardar)...")
    embeddings = model.encode(
        overviews_for_embedding_list,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Creando índice FAISS (IndexFlatIP)...")
    index_dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(index_dimension)
    faiss_index.add(embeddings)
    print(f"Índice FAISS creado con {faiss_index.ntotal} vectores.")

    engine_data = {
        "movie_ids": movie_ids_list,
        "titles_for_display": titles_for_display_list, # Se mapeará a 'titles_orig' en el agente
        "model_name": MODEL_NAME, # Guardar el nombre del modelo usado
        "embeddings_for_rebuild": embeddings, # Clave usada por el agente
    }
    with open(ENGINE_PKL_FILE, "wb") as f_out:
        pickle.dump(engine_data, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    dt_build = time.perf_counter() - t_start
    metrics_content = (
        f"Motor de Búsqueda por Overview (Sentence-Transformers)\n"
        f"Películas con overview indexadas: {len(titles_for_display_list)}\n"
        f"Modelo de embedding: {MODEL_NAME}\n"
        f"Dimensión del vector: {index_dimension}\n"
        f"Tiempo total de construcción: {dt_build:.2f} s\n"
        f"Tamaño {ENGINE_PKL_FILE.name}: {ENGINE_PKL_FILE.stat().st_size / 1e6:.2f} MB\n"
    )
    Path(METRICS_TXT_FILE).write_text(metrics_content, encoding="utf-8")
    print(f"✅ Motor de SINOPSIS guardado en {ENGINE_PKL_FILE} ({dt_build:.1f}s). Métricas en {METRICS_TXT_FILE}")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Asegúrate que '{DATA_FILE.name}' existe en '{DB_DIR}'.")
    build_engine()