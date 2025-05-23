#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import re
import sqlite3
import time
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Configuración ---
# Ruta base del proyecto (StreamlyBack) relativa a este script (src/buscador/buscador_por_titulo.py)
# Suponiendo que la DB está en StreamlyBack/src/data/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
DB_DIR = PROJECT_ROOT / "src" / "data"
DATA_FILE = DB_DIR / "tmdb_movies.db"

# Los PKL se guardarán en el mismo directorio que este script (src/buscador/)
OUTPUT_DIR = Path(__file__).resolve().parent
PKL_FILE = OUTPUT_DIR / "model_db_title_only.pkl" # Nombre original del PKL
METRICS_TXT = OUTPUT_DIR / "metrics_db_title_only.txt"

DB_TABLE_NAME = "movies"
DB_TITLE_COLUMN = "title"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # O el modelo que desees usar para construir

# --- Utilidades ---
def normaliza(t: str) -> str:
    text_str = str(t)
    text_str = re.sub(r"\s*\(\d{4}\)$", "", text_str).strip().lower()
    return text_str

# --- Construcción del Motor ---
def build_engine() -> None:
    print(f"⚙️ Construyendo motor de búsqueda por TÍTULO desde: {DATA_FILE}")
    print(f"   Guardando PKL en: {PKL_FILE}")
    t0 = time.perf_counter()

    if not DATA_FILE.exists():
        print(f"❌ Error: El archivo de base de datos '{DATA_FILE}' no se encontró.")
        return

    titles_orig_list = []
    try:
        conn = sqlite3.connect(DATA_FILE)
        cursor = conn.cursor()
        query = f"SELECT {DB_TITLE_COLUMN} FROM {DB_TABLE_NAME} WHERE {DB_TITLE_COLUMN} IS NOT NULL"
        print(f"Ejecutando consulta: {query}")
        cursor.execute(query)
        titles_orig_list = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
        conn.close()

        if not titles_orig_list:
            print(f"❌ No se encontraron títulos en la columna '{DB_TITLE_COLUMN}' de la tabla '{DB_TABLE_NAME}'.")
            return
    except sqlite3.Error as e:
        print(f"❌ Error al leer la base de datos SQLite: {e}")
        if "conn" in locals() and conn:
            conn.close()
        return

    print(f"Cargados {len(titles_orig_list)} títulos de la base de datos.")
    titles_norm_list = [normaliza(t) for t in titles_orig_list]

    print(f"Cargando modelo SentenceTransformer: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Generando embeddings para los títulos normalizados...")
    emb = model.encode(
        titles_norm_list,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    print("Creando índice FAISS (IndexFlatIP)...")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    print(f"Guardando motor en {PKL_FILE}...")
    engine_data = dict(
        titles_orig=titles_orig_list,
        titles_norm=titles_norm_list,
        embeddings=emb,
        model_name=MODEL_NAME, # Guardar el nombre del modelo usado para construir
    )
    with open(PKL_FILE, "wb") as f:
        pickle.dump(engine_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    dt = time.perf_counter() - t0
    Path(METRICS_TXT).write_text(
        f"Películas (desde DB): {len(titles_orig_list)}\n"
        f"Modelo de embedding: {MODEL_NAME}\n"
        f"Dim vector: {emb.shape[1]}\n"
        f"Tiempo total: {dt:.2f} s\n"
        f"Tamaño {PKL_FILE.name}: {PKL_FILE.stat().st_size/1e6:.1f} MB\n",
        encoding="utf-8",
    )
    print(f"✅ Motor de TÍTULO guardado en {PKL_FILE} ({dt:.1f}s). Métricas en {METRICS_TXT}")

if __name__ == "__main__":
    # Crear directorio de salida para PKL si no existe (aunque __file__.parent ya debería existir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Crear directorio de datos si no existe (para que el usuario sepa dónde poner la DB)
    DB_DIR.mkdir(parents=True, exist_ok=True) 
    print(f"Asegúrate que '{DATA_FILE.name}' existe en '{DB_DIR}'.")
    build_engine()