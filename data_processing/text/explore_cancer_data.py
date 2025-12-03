#!/usr/bin/env python3
"""
Script para explorar el dataset Cancer_Data.csv
Analiza las columnas disponibles y sus características
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def explore_dataset(file_path):
    """
    Explora el dataset de cáncer y muestra información detallada
    """
    print("=" * 60)
    print("ANÁLISIS EXPLORATORIO DEL DATASET DE CÁNCER")
    print("=" * 60)
    
    # Cargar el dataset
    df = pd.read_csv(file_path)
    
    print(f"\n📊 INFORMACIÓN GENERAL:")
    print(f"   • Forma del dataset: {df.shape}")
    print(f"   • Número de muestras: {df.shape[0]}")
    print(f"   • Número de características: {df.shape[1]}")
    
    print(f"\n📋 COLUMNAS DISPONIBLES:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\n🎯 VARIABLE OBJETIVO:")
    print(f"   • Columna: 'diagnosis'")
    print(f"   • Valores únicos: {df['diagnosis'].unique()}")
    print(f"   • Distribución:")
    print(df['diagnosis'].value_counts())
    
    print(f"\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())
    
    print(f"\n🔍 INFORMACIÓN DE TIPOS DE DATOS:")
    print(df.dtypes)
    
    print(f"\n❓ VALORES FALTANTES:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("   ✅ No hay valores faltantes")
    else:
        print(missing_values[missing_values > 0])
    
    return df

def analyze_feature_importance(df):
    """
    Analiza la importancia de las características
    """
    print(f"\n🔬 ANÁLISIS DE CARACTERÍSTICAS:")
    
    # Separar características y variable objetivo
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    
    print(f"   • Características numéricas: {X.shape[1]}")
    print(f"   • Características eliminadas: ['id', 'diagnosis']")
    
    # Calcular correlaciones con la variable objetivo
    y_numeric = (y == 'M').astype(int)  # M=1, B=0
    correlations = X.corrwith(y_numeric).abs().sort_values(ascending=False)
    
    print(f"\n📊 TOP 10 CARACTERÍSTICAS MÁS CORRELACIONADAS:")
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"   {i:2d}. {feature}: {corr:.4f}")
    
    return X, y

def perform_pca_analysis(X, y):
    """
    Realiza análisis PCA del dataset
    """
    print(f"\n🎯 ANÁLISIS PCA:")
    
    # Estandarizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Calcular varianza explicada
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"   • Varianza explicada por los primeros 5 componentes:")
    for i in range(5):
        print(f"     PC{i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)")
    
    print(f"\n   • Varianza acumulada:")
    for i in [2, 5, 10, 15, 20]:
        if i <= len(cumulative_variance):
            print(f"     Primeros {i} componentes: {cumulative_variance[i-1]:.4f} ({cumulative_variance[i-1]*100:.2f}%)")
    
    # Encontrar número de componentes para 95% de varianza
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\n   • Componentes necesarios para 95% de varianza: {n_components_95}")
    
    return X_pca, pca, explained_variance_ratio

def plot_pca_visualizations(X_pca, y, explained_variance_ratio):
    """
    Crea visualizaciones del análisis PCA
    """
    print(f"\n📊 CREANDO VISUALIZACIONES...")
    
    # Configurar el estilo
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Varianza explicada
    axes[0, 0].plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    axes[0, 0].set_title('Varianza Explicada por Componente')
    axes[0, 0].set_xlabel('Componente Principal')
    axes[0, 0].set_ylabel('Varianza Explicada')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Varianza acumulada
    cumulative_variance = np.cumsum(explained_variance_ratio)
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    axes[0, 1].axhline(y=0.95, color='g', linestyle='--', label='95%')
    axes[0, 1].set_title('Varianza Acumulada')
    axes[0, 1].set_xlabel('Número de Componentes')
    axes[0, 1].set_ylabel('Varianza Acumulada')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter plot PC1 vs PC2
    colors = ['red' if label == 'M' else 'blue' for label in y]
    axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
    axes[1, 0].set_title('PC1 vs PC2')
    axes[1, 0].set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
    axes[1, 0].set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scatter plot PC1 vs PC3
    axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 2], c=colors, alpha=0.6)
    axes[1, 1].set_title('PC1 vs PC3')
    axes[1, 1].set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
    axes[1, 1].set_ylabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/cancer_pca_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Gráficos guardados en: visualization/plots/cancer_pca_analysis.png")

def recommend_features_for_svm(X, y):
    """
    Recomienda qué características usar para el modelo SVM
    """
    print(f"\n🤖 RECOMENDACIONES PARA MODELO SVM:")
    
    # Calcular correlaciones
    y_numeric = (y == 'M').astype(int)
    correlations = X.corrwith(y_numeric).abs().sort_values(ascending=False)
    
    print(f"   • Características recomendadas (top 10):")
    top_features = correlations.head(10).index.tolist()
    for i, feature in enumerate(top_features, 1):
        corr = correlations[feature]
        print(f"     {i:2d}. {feature} (correlación: {corr:.4f})")
    
    print(f"\n   • Características por categoría:")
    mean_features = [col for col in X.columns if '_mean' in col]
    se_features = [col for col in X.columns if '_se' in col]
    worst_features = [col for col in X.columns if '_worst' in col]
    
    print(f"     • Características _mean: {len(mean_features)}")
    print(f"     • Características _se: {len(se_features)}")
    print(f"     • Características _worst: {len(worst_features)}")
    
    return top_features

def main():
    """
    Función principal
    """
    file_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/raw/text/Cancer_Data.csv'
    
    try:
        # Explorar dataset
        df = explore_dataset(file_path)
        
        # Analizar características
        X, y = analyze_feature_importance(df)
        
        # Realizar PCA
        X_pca, pca, explained_variance_ratio = perform_pca_analysis(X, y)
        
        # Crear visualizaciones
        plot_pca_visualizations(X_pca, y, explained_variance_ratio)
        
        # Recomendar características para SVM
        top_features = recommend_features_for_svm(X, y)
        
        print(f"\n✅ ANÁLISIS COMPLETADO")
        print(f"   • Dataset cargado exitosamente")
        print(f"   • PCA realizado con {X.shape[1]} características")
        print(f"   • Visualizaciones guardadas")
        print(f"   • Características recomendadas para SVM identificadas")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()
