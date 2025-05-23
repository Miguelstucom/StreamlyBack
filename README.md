# Sistema de Recomendación de Películas

Este proyecto implementa un sistema de recomendación de películas utilizando datos de archivos CSV y técnicas de machine learning.

## Estructura del Proyecto

```
movie_recommender/
├── data/                   # Directorio para archivos CSV
├── src/                    # Código fuente
│   ├── data/              # Módulos para procesamiento de datos
│   ├── models/            # Modelos de machine learning
│   ├── api/               # Endpoints de la API
│   └── utils/             # Utilidades y helpers
├── tests/                 # Tests unitarios
└── notebooks/             # Jupyter notebooks para análisis
```

## Instalación

1. Crear un entorno virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Colocar los archivos CSV en el directorio `data/`
2. Ejecutar el servidor:
```bash
uvicorn src.api.main:app --reload
```

## Características

- Procesamiento de datos de películas desde CSV
- Sistema de recomendación basado en contenido
- API REST para obtener recomendaciones
- Análisis exploratorio de datos #   s t r e a m l y B a c k  
 