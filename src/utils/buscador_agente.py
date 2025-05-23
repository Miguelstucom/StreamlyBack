#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agente de búsqueda de películas para la API StreamlyBack.
Clasifica la consulta del usuario y la dirige al motor apropiado.
"""
import os
import re
import difflib
import pickle
import time
import logging
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any, Union

import faiss
from sentence_transformers import SentenceTransformer
import torch

# --- Configuración ---
# Rutas relativas al directorio de este script (src/utils/buscador_agente.py)
# Asumiendo que los PKL están en src/buscador/
CURRENT_DIR = Path(__file__).resolve().parent
PKL_DIR = CURRENT_DIR.parent / "buscador" # Sube un nivel a src/ y luego entra a buscador/

PKL_TITLE    = PKL_DIR / "model_db_title_only.pkl"
PKL_GENRE    = PKL_DIR / "genre_engine.pkl" # Este es el que generaste (simple, sin FAISS)
PKL_OVERVIEW = PKL_DIR / "st_overview_engine.pkl"
TOP_K        = 5

# Configuración de Logging (más apropiado para un backend)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Estado Global de los Motores (Cargados una vez) ---
engines_loaded = False
data_title, model_title, index_title = None, None, None
data_genre_content, all_genres_list, title_to_genre_map_dict = None, None, None # Renombrado para claridad
data_over, model_over, index_over = None, None, None

# --- Utilidades ---
def normaliza(txt: str) -> str:
    return re.sub(r"\s*\(\d{4}\)$", "", str(txt)).strip().lower()

# --- Carga de Motores ---
def _load_faiss_engine_internal(pkl_path: Path, is_overview_engine: bool = False):
    logger.info(f"Cargando motor FAISS desde: {pkl_path}")
    data = pickle.load(open(pkl_path, "rb"))
    
    if "embeddings" in data:
        emb = data["embeddings"]
    elif "embeddings_for_rebuild" in data: # Usado por st_overview_engine.pkl
        emb = data["embeddings_for_rebuild"]
    else:
        logger.error(f"Clave 'embeddings' o 'embeddings_for_rebuild' no encontrada en {pkl_path}")
        raise ValueError(f"Clave de embeddings faltante en {pkl_path}")

    # Usar el nombre del modelo guardado en el PKL.
    model_name_from_pkl = data.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") # Fallback por si acaso
    logger.info(f"  Cargando SentenceTransformer: {model_name_from_pkl} para {pkl_path.name}")
    
    # Configurar el modelo con opciones básicas
    model = SentenceTransformer(model_name_from_pkl)
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    if is_overview_engine:
        if "titles_for_display" in data and "titles_orig" not in data:
            data["titles_orig"] = data["titles_for_display"]
    
    if "titles_orig" not in data:
        logger.error(f"Clave 'titles_orig' no encontrada en {pkl_path}")
        raise ValueError(f"Clave 'titles_orig' (o 'titles_for_display') no encontrada en {pkl_path}")

    if "titles_norm" not in data:
        logger.info(f"  Generando 'titles_norm' para {pkl_path.name} a partir de 'titles_orig'.")
        data["titles_norm"] = [normaliza(t) for t in data["titles_orig"]]
    
    logger.info(f"  Motor {pkl_path.name} cargado. {len(data['titles_orig'])} títulos.")
    return data, model, index

def _load_genre_engine_internal(pkl_path: Path) -> Dict[str, Any]:
    logger.info(f"Cargando motor de géneros desde: {pkl_path}")
    with open(pkl_path, "rb") as f:
        genre_data_raw = pickle.load(f)
    
    processed_data = {
        "titles_orig": list(genre_data_raw.get("titles", [])),
        "genres_orig": list(genre_data_raw.get("genres", [])),
        "all_genres": set(map(str.lower, genre_data_raw.get("catalogo", [])))
    }
    logger.info(f"  Motor {pkl_path.name} cargado. {len(processed_data['titles_orig'])} títulos, {len(processed_data['all_genres'])} géneros.")
    return processed_data

def inicializar_motores():
    global engines_loaded, data_title, model_title, index_title, \
           data_genre_content, all_genres_list, title_to_genre_map_dict, \
           data_over, model_over, index_over

    if engines_loaded:
        logger.info("Motores de búsqueda ya inicializados.")
        return

    logger.info("Inicializando motores de búsqueda por primera vez...")
    try:
        data_title, model_title, index_title = _load_faiss_engine_internal(PKL_TITLE)
        
        data_genre_loaded = _load_genre_engine_internal(PKL_GENRE)
        data_genre_content = data_genre_loaded 
        all_genres_list = data_genre_loaded["all_genres"]
        title_to_genre_map_dict = {
            title: genres_str for title, genres_str in zip(data_genre_loaded["titles_orig"], data_genre_loaded["genres_orig"])
        }
        
        data_over, model_over, index_over = _load_faiss_engine_internal(PKL_OVERVIEW, is_overview_engine=True)
        
        engines_loaded = True
        logger.info("Motores de búsqueda listos.")
    except Exception as e:
        logger.error(f"Error durante la inicialización de motores: {e}", exc_info=True)
        # Podrías querer que la aplicación falle si los motores no cargan, o manejarlo de otra forma.
        raise RuntimeError(f"No se pudieron inicializar los motores de búsqueda: {e}")


# --- Funciones de Búsqueda (Internas) ---
def _buscar_faiss_internal(q:str, model, index, titles_orig: List[str], titles_norm: List[str])->List[str]:
    qn = normaliza(q)
    sug = difflib.get_close_matches(qn, titles_norm, n=1, cutoff=0.7)
    corrected_query_for_embedding = qn
    suggestion_info = None

    if sug and sug[0] != qn:
        try:
            original_suggested_title = titles_orig[titles_norm.index(sug[0])]
            suggestion_info = f"Consulta original: '{q}', corregida a: '{original_suggested_title}' (embedding con '{sug[0]}')"
            logger.info(f"Sugerencia de typo: {suggestion_info}")
        except (ValueError, IndexError):
            suggestion_info = f"Consulta original: '{q}', corregida a: '{sug[0]}'"
            logger.info(f"Sugerencia de typo (sin título original mapeado): {suggestion_info}")
        corrected_query_for_embedding = sug[0]
    
    vec = model.encode([corrected_query_for_embedding], convert_to_numpy=True,
                       normalize_embeddings=True).astype("float32")
    _distances, faiss_indices = index.search(vec, TOP_K) 
    
    results = []
    for i in faiss_indices[0]:
        if i >= 0 and i < len(titles_orig):
             results.append(titles_orig[i])
    return results


def _parse_genres_query_internal(user_q:str)->List[str]:
    q = user_q.lower()
    q = re.sub(r"[,&]", " ", q)
    q = re.sub(r"\band\b", " ", q, flags=re.IGNORECASE)
    q = " ".join(q.split())
    found=[]
    tmp=q
    if not all_genres_list: 
        logger.warning("all_genres_list no está inicializado para _parse_genres_query_internal")
        return []
    for g in sorted(all_genres_list, key=len, reverse=True):
        pat=r"\b"+re.escape(g)+r"\b"
        if re.search(pat,tmp):
            found.append(g)
            tmp=re.sub(pat," ",tmp,1)
    return found

def _buscar_por_generos_internal(user_q:str)->List[Tuple[str,str]]:
    if not data_genre_content:
        logger.warning("data_genre_content no está inicializado para _buscar_por_generos_internal")
        return []

    wanted_norm = set(_parse_genres_query_internal(user_q))
    if not wanted_norm: return []
    
    exac, super_, part = [],[],[]
    for title, genres_str in zip(data_genre_content["titles_orig"], data_genre_content["genres_orig"]):
        movie_genres_set_norm = {normaliza(x) for x in genres_str.split(",") if x.strip()}
        if not movie_genres_set_norm: continue

        if movie_genres_set_norm == wanted_norm:
            exac.append((title, genres_str))
        elif wanted_norm.issubset(movie_genres_set_norm):
            super_.append((title, genres_str))
        elif movie_genres_set_norm & wanted_norm:
            part.append((title, genres_str))
            
    return (exac+super_+part)[:TOP_K]

# --- Clasificación de Consulta (Interna) ---
def _classify_query_internal(q:str)->str:
    lq = q.lower()
    words = q.strip().split()
    num_words = len(words)

    if lq.startswith("genero:"):
        return "genre"

    overview_keywords = [
        "película sobre", "pelicula sobre", "peli sobre", "película de un", "peli de un",
        "películas parecidas a", "peliculas parecidas a", "historia de", "trama sobre",
        "busco algo que trate de", "donde sale un", "persona que", "grupo de", "un tío que", "una tía que"
    ]
    if any(kw in lq for kw in overview_keywords) or q.endswith("?"):
        return "overview"

    if re.search(r"\(\d{4}\)$", q.strip()):
        return "title"

    if not all_genres_list: 
        logger.warning("all_genres_list no disponible para _classify_query_internal (géneros)")
        extracted_genres = []
    else:
        extracted_genres = _parse_genres_query_internal(q) 
    
    if extracted_genres:
        temp_q = q
        for g_norm in extracted_genres: 
            temp_q = re.sub(r'\b' + re.escape(g_norm) + r'\b', '', temp_q, flags=re.IGNORECASE)
        remaining_text = temp_q.replace("genero:", "").strip() 
        remaining_words_list = [w for w in remaining_text.split() if w]
        if not remaining_words_list: 
            return "genre"
        if len(remaining_words_list) <= 2 and all(len(word) <=3 for word in remaining_words_list):
             return "genre"

    qn = normaliza(q)

    if not data_title: 
        logger.warning("data_title no disponible para _classify_query_internal (títulos)")
    elif 1 <= num_words <= 3 and not extracted_genres: 
        short_query_title_match = difflib.get_close_matches(qn, data_title["titles_norm"], n=1, cutoff=0.72)
        if short_query_title_match:
            return "title"

    if not data_title:
        pass 
    elif difflib.get_close_matches(qn, data_title["titles_norm"], n=1, cutoff=0.88):
        return "title"

    if num_words > 6: 
        return "overview"
        
    if extracted_genres and num_words > len(extracted_genres) + 1 : 
        return "ambiguous"

    return "ambiguous"

# --- Interfaz Pública del Agente ---
def buscar_peliculas_agente(query_usuario: str, tipo_forzado: str = None) -> Dict[str, Any]:
    if not engines_loaded:
        logger.warning("Motores no inicializados. Llamando a inicializar_motores(). Esto debería ocurrir en el startup de la API.")
        inicializar_motores() # Asegurar que se cargan, aunque idealmente se hace una vez al inicio de la app

    if not query_usuario or not query_usuario.strip():
        return {"tipo_busqueda_efectuado": "error", "resultados": [], "info_adicional": {"mensaje": "Consulta vacía."}}

    start_time = time.perf_counter()
    tipo_detectado = _classify_query_internal(query_usuario)
    
    tipo_busqueda_final = tipo_forzado if tipo_forzado in ["title", "genre", "overview"] else tipo_detectado
    
    logger.info(f"Consulta: '{query_usuario}', Tipo Forzado: {tipo_forzado}, Tipo Detectado: {tipo_detectado}, Tipo Final: {tipo_busqueda_final}")
    
    resultados_finales = []
    info_adicional = {"clasificacion_original": tipo_detectado}

    if tipo_busqueda_final == "ambiguous":
        info_adicional["nota_ambiguedad"] = "Clasificación ambigua. Intentando como título, luego como descripción."
        tipo_busqueda_final = "title" # Default para ambigüedad

    if tipo_busqueda_final == "title":
        raw_results = _buscar_faiss_internal(query_usuario, model_title, index_title,
                                           data_title["titles_orig"], data_title["titles_norm"])
        for title in raw_results:
            resultados_finales.append({
                "titulo": title,
                "generos": title_to_genre_map_dict.get(title, "N/A")
            })

    elif tipo_busqueda_final == "genre":
        gq = query_usuario
        if query_usuario.lower().startswith("genero:"):
            gq = query_usuario[7:].strip()
        
        raw_results = _buscar_por_generos_internal(gq)
        for title, genres_str in raw_results:
            resultados_finales.append({
                "titulo": title,
                "generos": genres_str 
            })
            
    elif tipo_busqueda_final == "overview":
        raw_results = _buscar_faiss_internal(query_usuario, model_over, index_over,
                                           data_over["titles_orig"], data_over["titles_norm"])
        for title in raw_results:
            resultados_finales.append({
                "titulo": title,
                "generos": title_to_genre_map_dict.get(title, "N/A")
            })
    
    # Si la primera opción para ambigüedad (título) no dio resultados, intentar por descripción.
    if info_adicional.get("nota_ambiguedad") and tipo_busqueda_final == "title" and not resultados_finales:
        logger.info("Búsqueda ambigua por título sin resultados, intentando por descripción...")
        info_adicional["nota_ambiguedad"] = "Clasificación ambigua. Título sin éxito, intentando como descripción."
        tipo_busqueda_final = "overview" # Cambiamos el tipo efectuado para el log final
        raw_results_overview = _buscar_faiss_internal(query_usuario, model_over, index_over,
                                           data_over["titles_orig"], data_over["titles_norm"])
        for title in raw_results_overview:
            resultados_finales.append({
                "titulo": title,
                "generos": title_to_genre_map_dict.get(title, "N/A")
            })

    end_time = time.perf_counter()
    processing_time = end_time - start_time
    logger.info(f"Búsqueda para '{query_usuario}' completada en {processing_time:.4f}s. Resultados: {len(resultados_finales)}")

    return {
        "tipo_busqueda_efectuado": tipo_busqueda_final,
        "resultados": resultados_finales,
        "info_adicional": info_adicional,
        "tiempo_procesamiento_seg": round(processing_time, 4)
    }

# --- Bloque de prueba ---
if __name__ == '__main__':
    # Crear directorio PKL si no existe (para que el desarrollador sepa dónde poner los PKL)
    if not PKL_DIR.exists():
        logger.warning(f"El directorio para los archivos PKL ({PKL_DIR}) no existe. Creándolo.")
        PKL_DIR.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Por favor, asegúrate de que los archivos .pkl están en {PKL_DIR} después de generarlos.")

    # Verificar existencia de PKLs antes de inicializar
    missing_pkls = []
    for pkl_f in [PKL_TITLE, PKL_GENRE, PKL_OVERVIEW]:
        if not pkl_f.exists():
            missing_pkls.append(str(pkl_f))
    
    if missing_pkls:
        logger.error("Faltan los siguientes archivos PKL necesarios:")
        for mp in missing_pkls:
            logger.error(f" - {mp}")
        logger.error("Por favor, genera estos archivos usando los scripts correspondientes en src/buscador/ y colócalos en ese directorio.")
        exit(1) # Salir si faltan PKLs críticos

    logger.info("Llamando a inicializar_motores() (se cargarán si es la primera vez)...")
    inicializar_motores() # Esto cargará los modelos.
    
    logger.info("\n--- Ejemplo de Búsqueda desde el Agente ---")
    
    queries_test = [
        "toy story",
        "genero: action comedy",
        "película sobre un muñeco que cobra vida",
        "toip storip", # Debería ser clasificado como título por la mejora
        "accion aventura cars", # Probablemente ambiguo
        "un drama intenso sobre la guerra"
    ]

    for q_test in queries_test:
        logger.info(f"\n--- Buscando: '{q_test}' ---")
        resultados_dict = buscar_peliculas_agente(q_test)
        logger.info(f"  Tipo Buscado: {resultados_dict['tipo_busqueda_efectuado']}")
        logger.info(f"  Info Adicional: {resultados_dict['info_adicional']}")
        logger.info(f"  Tiempo: {resultados_dict['tiempo_procesamiento_seg']}s")
        if resultados_dict["resultados"]:
            logger.info(f"  Resultados ({len(resultados_dict['resultados'])}):")
            for i, r in enumerate(resultados_dict["resultados"][:3]): # Mostrar solo los 3 primeros
                logger.info(f"    {i+1}. {r['titulo']} (Generos: {r['generos']})")
        else:
            logger.info("    No se encontraron resultados.")