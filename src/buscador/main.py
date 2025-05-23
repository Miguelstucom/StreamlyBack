#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agente jefe que enruta la consulta del usuario al motor adecuado
(título, géneros o descripción).

Requisitos:
  • Los tres .pkl generados en el mismo directorio:
      - model_db_title_only.pkl
      - model_db_title_and_genres.pkl
      - st_overview_engine.pkl
  • tmdb_movies.db (solo lo lee el motor-géneros si necesita reconstruirse)
  • pip install sentence-transformers faiss-cpu (o faiss-gpu)
"""

# --------------------------------------------------------------------------- #
import os, re, difflib, pickle, time, sqlite3
from pathlib import Path
from typing import List, Tuple, Set

import faiss
from sentence_transformers import SentenceTransformer

# ---------- constantes ----------------------------------------------------- #
PKL_TITLE      = "model_db_title_only.pkl"
PKL_GENRE      = "model_db_title_and_genres.pkl"
PKL_OVERVIEW   = "st_overview_engine.pkl"
DATA_FILE      = "tmdb_movies.db"      # por si hay que reconstruir motor-géneros
TOP_K          = 5

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"

# ---------- utilidades genéricas ------------------------------------------- #
def normaliza(txt:str)->str:
    return re.sub(r"\s*\(\d{4}\)$","",str(txt)).strip().lower()

# ----------------- Carga de los tres motores ------------------------------- #
def load_faiss(pkl_path:str):
    data  = pickle.load(open(pkl_path,"rb"))
    emb   = data["embeddings"]
    model = SentenceTransformer(data["model_name"])
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
    return data, model, index                 # devolvemos el dict completo tb

## 1) TÍTULO
data_title, model_title, index_title = load_faiss(PKL_TITLE)

## 2) GÉNERO  («data_genre» contiene también ‘all_genres’ salvado al construir)
data_genre, model_genre, index_genre = load_faiss(PKL_GENRE)
all_genres: Set[str] = set(data_genre["all_genres"])   # minúsculas

## 3) DESCRIPCIÓN / SINOPSIS
data_over, model_over, index_over   = load_faiss(PKL_OVERVIEW)

# ---------------------- funciones de búsqueda ------------------------------ #
def buscar_faiss(q:str, model, index, titles_orig, titles_norm)->List[str]:
    qn = normaliza(q)
    # corrección de typos
    sug = difflib.get_close_matches(qn, titles_norm, n=1, cutoff=0.7)
    if sug and sug[0]!=qn:
        qn = sug[0]
    vec = model.encode([qn], convert_to_numpy=True,
                       normalize_embeddings=True).astype("float32")
    _, idx = index.search(vec, TOP_K)
    return [titles_orig[i] for i in idx[0]]

def parse_genres_query(user_q:str)->List[str]:
    q = user_q.lower()
    q = re.sub(r"[,&]", " ", q)
    q = re.sub(r"\band\b", " ", q)
    q = " ".join(q.split())
    found=[]
    tmp=q
    for g in sorted(all_genres, key=len, reverse=True):
        pat=r"\b"+re.escape(g)+r"\b"
        if re.search(pat,tmp):
            found.append(g); tmp=re.sub(pat," ",tmp,1)
    return found

def format_genres_output(movie_g:str, wanted:List[str])->str:
    wset={normaliza(x) for x in wanted}
    out=[]
    for g in [s.strip() for s in movie_g.split(",") if s.strip()]:
        out.append(f"{GREEN if normaliza(g) in wset else RED}{g}{RESET}")
    return ", ".join(out)

def buscar_por_generos(user_q:str)->List[Tuple[str,str]]:
    wanted=set(parse_genres_query(user_q))
    if not wanted: return []
    exac, super_, part = [],[],[]
    for t,g in zip(data_genre["titles_orig"], data_genre["genres_orig"]):
        mset={normaliza(x) for x in g.split(",")}
        if not mset: continue
        if mset==wanted:            exac.append((t,g))
        elif wanted<=mset:          super_.append((t,g))
        elif mset & wanted:         part.append((t,g))
    return (exac+super_+part)[:TOP_K]

# ------------------------ clasificación de la consulta --------------------- #
def classify_query(q:str)->str:
    """
    Devuelve 'genre' | 'title' | 'overview' | 'ambiguous'
    """
    if q.lower().startswith("genero:"):
        return "genre"

    words = q.strip().split()
    # si contiene keywords típicas de pregunta larga -> overview
    if len(words) > 8 or q.endswith("?"):
        return "overview"

    # ¿solo géneros?
    if parse_genres_query(q):
        return "genre"

    # heurística: si tiene año entre paréntesis, parece título
    if re.search(r"\(\d{4}\)$", q.strip()):
        return "title"

    # si la similitud textual contra títulos es muy alta -> título
    qn=normaliza(q)
    close = difflib.get_close_matches(qn, data_title["titles_norm"], n=1, cutoff=0.85)
    if close: return "title"

    # ambiguous: podría ser descripción corta
    return "ambiguous"

def pedir_confirmacion(tipo:str)->bool:
    resp=input(f"¿Quieres buscar por {tipo}? (s/n): ").strip().lower()
    return resp.startswith("s")

# --------------------------- CLI principal --------------------------------- #
def main():
    print("🤖  Buscador agente listo.\nEscribe tu consulta o 'exit'.\n")
    while True:
        q = input("Consulta: ").strip()
        if not q: continue
        if q.lower() in {"exit","salir","apagar sistema"}:
            print("👋  Hasta luego"); break

        tipo = classify_query(q)
        if tipo=="ambiguous":
            print("No tengo claro si es título o descripción.")
            if pedir_confirmacion("título"):
                tipo="title"
            elif pedir_confirmacion("descripción"):
                tipo="overview"
            else:
                print("Abortado."); continue

        if tipo=="title":
            res=buscar_faiss(q, model_title, index_title,
                             data_title["titles_orig"], data_title["titles_norm"])
            if not res: print("Sin resultados.");
            else:
                print(f"Top {len(res)} por TÍTULO:")
                for r in res:
                    idx = data_genre["titles_orig"].index(r) \
                          if r in data_genre["titles_orig"] else None
                    genres = data_genre["genres_orig"][idx] if idx is not None else "?"
                    print(f" • {r}  (Géneros: {genres})")

        elif tipo=="genre":
            gq = q[7:].strip() if q.lower().startswith("genero:") else q
            res=buscar_por_generos(gq)
            if not res: print("Sin resultados para esos géneros.")
            else:
                wl=parse_genres_query(gq)
                print(f"Top {len(res)} por GÉNERO:")
                for t,g in res:
                    print(f" • {t}  ({format_genres_output(g, wl)})")

        elif tipo=="overview":
            res=buscar_faiss(q, model_over, index_over,
                             data_over["titles_orig"], data_over["titles_norm"])
            if not res: print("Sin resultados.");
            else:
                print(f"Top {len(res)} por DESCRIPCIÓN:")
                for r in res:
                    print(" •", r)

# --------------------------------------------------------------------------- #
if __name__=="__main__":
    # chequeo rápido de ficheros
    for f in (PKL_TITLE, PKL_GENRE, PKL_OVERVIEW):
        if not Path(f).exists():
            print(f"❌ Falta {f}.  Asegúrate de haber generado los tres modelos."); exit(1)
    main()
