import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_rf_scores():
    """Calcula los scores RF (Recency-Frequency) para cada usuario."""
    try:
        # Conectar a la base de datos
        conn = sqlite3.connect('data/tmdb_movies.db')
        
        # Obtener datos de visualización
        query = '''
        SELECT 
            uf.user_id,
            uf.movie_id,
            uf.view_date
        FROM user_film uf
        '''
        df = pd.read_sql_query(query, conn)
        
        # Convertir view_date a datetime
        df['view_date'] = pd.to_datetime(df['view_date'])
        
        # Calcular fecha máxima para referencia
        max_date = df['view_date'].max()
        
        # Calcular métricas RF por usuario
        rf = pd.DataFrame()
        
        # Recency: días desde la última visualización
        recency = df.groupby('user_id')['view_date'].max()
        rf['Recency'] = (max_date - recency).dt.days
        
        # Frequency: número total de películas vistas
        rf['Frequency'] = df.groupby('user_id')['movie_id'].count()
        
        # Normalizar los valores
        scaler = StandardScaler()
        rf_normalized = pd.DataFrame(
            scaler.fit_transform(rf),
            columns=['Recency', 'Frequency'],
            index=rf.index
        )
        
        # Calcular el método del codo
        inertias = []
        K = range(1, 11)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(rf_normalized)
            inertias.append(kmeans.inertia_)
        
        # Crear gráfica del método del codo
        plt.figure(figsize=(10, 6))
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Método del Codo para Selección Óptima de k')
        plt.savefig('elbow_method.png')
        plt.close()
        
        # Crear visualización de los clusters (usando k=4 como ejemplo)
        optimal_k = 3  # Podemos ajustar esto basado en el método del codo
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        rf_normalized['Cluster'] = kmeans.fit_predict(rf_normalized)
        
        # Crear gráfica de dispersión
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=rf_normalized,
            x='Recency',
            y='Frequency',
            hue='Cluster',
            palette='deep'
        )
        plt.title('Clusters de Usuarios basados en Recency y Frequency')
        plt.xlabel('Recency (días desde última visualización)')
        plt.ylabel('Frequency (número de películas vistas)')
        plt.savefig('rf_clusters.png')
        plt.close()
        
        # Analizar características de cada cluster
        cluster_analysis = rf.join(pd.DataFrame({'Cluster': rf_normalized['Cluster']}))
        cluster_means = cluster_analysis.groupby('Cluster').mean()
        
        print("\nCaracterísticas promedio de cada cluster:")
        print(cluster_means)
        
        # Añadir interpretación de los clusters
        print("\nInterpretación de los clusters:")
        for cluster in cluster_means.index:
            recency = cluster_means.loc[cluster, 'Recency']
            frequency = cluster_means.loc[cluster, 'Frequency']
            
            print(f"\nCluster {cluster}:")
            print(f"- Recency: {recency:.2f} días desde la última visualización")
            print(f"- Frequency: {frequency:.2f} películas vistas en total")
            
            if recency < 30:
                recency_status = "muy reciente"
            elif recency < 90:
                recency_status = "reciente"
            else:
                recency_status = "antiguo"
                
            if frequency > 100:
                frequency_status = "muy activo"
            elif frequency > 50:
                frequency_status = "moderadamente activo"
            else:
                frequency_status = "poco activo"
                
            print(f"Este cluster representa usuarios con actividad {recency_status} y {frequency_status}")
        
        return rf_normalized, cluster_means
        
    except Exception as e:
        print(f"Error en el análisis RF: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    rf_data, cluster_means = calculate_rf_scores()
    print("\nAnálisis RF completado. Se han generado las gráficas 'elbow_method.png' y 'rf_clusters.png'") 