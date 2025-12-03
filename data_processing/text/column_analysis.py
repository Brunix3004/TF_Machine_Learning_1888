#!/usr/bin/env python3
"""
Script para analizar y seleccionar las mejores columnas para el modelo SVM
Basado en correlación, importancia y análisis estadístico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ColumnAnalyzer:
    """
    Clase para analizar y seleccionar las mejores columnas del dataset
    """
    
    def __init__(self, file_path):
        """
        Inicializa el analizador de columnas
        """
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_importance = None
        self.correlations = None
        
    def load_data(self):
        """
        Carga los datos
        """
        print("📊 Cargando dataset...")
        self.df = pd.read_csv(self.file_path)
        
        # Separar características y variable objetivo
        self.X = self.df.drop(['id', 'diagnosis'], axis=1)
        self.y = self.df['diagnosis']
        
        # Convertir variable objetivo a numérica
        self.y_numeric = (self.y == 'M').astype(int)  # M=1, B=0
        
        print(f"   ✅ Dataset cargado: {self.X.shape[0]} muestras, {self.X.shape[1]} características")
        
        return self.X, self.y
    
    def analyze_correlations(self):
        """
        Analiza las correlaciones entre características y la variable objetivo
        """
        print("\n🔍 ANÁLISIS DE CORRELACIONES:")
        
        # Calcular correlaciones
        self.correlations = self.X.corrwith(self.y_numeric).abs().sort_values(ascending=False)
        
        print(f"   📊 Top 15 características más correlacionadas:")
        for i, (feature, corr) in enumerate(self.correlations.head(15).items(), 1):
            print(f"     {i:2d}. {feature}: {corr:.4f}")
        
        return self.correlations
    
    def analyze_feature_importance(self):
        """
        Analiza la importancia de las características usando Random Forest
        """
        print("\n🌲 ANÁLISIS DE IMPORTANCIA CON RANDOM FOREST:")
        
        # Entrenar Random Forest para obtener importancia
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y_numeric)
        
        # Obtener importancia de características
        self.feature_importance = pd.Series(
            rf.feature_importances_, 
            index=self.X.columns
        ).sort_values(ascending=False)
        
        print(f"   📊 Top 15 características más importantes:")
        for i, (feature, importance) in enumerate(self.feature_importance.head(15).items(), 1):
            print(f"     {i:2d}. {feature}: {importance:.4f}")
        
        return self.feature_importance
    
    def analyze_statistical_tests(self):
        """
        Realiza pruebas estadísticas para selección de características
        """
        print("\n📈 ANÁLISIS ESTADÍSTICO:")
        
        # F-test
        f_selector = SelectKBest(score_func=f_classif, k='all')
        f_selector.fit(self.X, self.y_numeric)
        f_scores = pd.Series(f_selector.scores_, index=self.X.columns).sort_values(ascending=False)
        
        print(f"   📊 Top 15 características (F-test):")
        for i, (feature, score) in enumerate(f_scores.head(15).items(), 1):
            print(f"     {i:2d}. {feature}: {score:.2f}")
        
        # Mutual Information
        mi_scores = mutual_info_classif(self.X, self.y_numeric, random_state=42)
        mi_scores = pd.Series(mi_scores, index=self.X.columns).sort_values(ascending=False)
        
        print(f"\n   📊 Top 15 características (Mutual Information):")
        for i, (feature, score) in enumerate(mi_scores.head(15).items(), 1):
            print(f"     {i:2d}. {feature}: {score:.4f}")
        
        return f_scores, mi_scores
    
    def create_feature_ranking(self):
        """
        Crea un ranking combinado de características
        """
        print("\n🏆 CREANDO RANKING COMBINADO:")
        
        # Normalizar scores (0-1)
        corr_norm = (self.correlations - self.correlations.min()) / (self.correlations.max() - self.correlations.min())
        importance_norm = (self.feature_importance - self.feature_importance.min()) / (self.feature_importance.max() - self.feature_importance.min())
        
        # Crear ranking combinado
        combined_score = (corr_norm + importance_norm) / 2
        
        ranking_df = pd.DataFrame({
            'Característica': combined_score.index,
            'Correlación': self.correlations,
            'Importancia_RF': self.feature_importance,
            'Score_Combinado': combined_score
        }).sort_values('Score_Combinado', ascending=False)
        
        print(f"   📊 Top 20 características (ranking combinado):")
        for i, row in ranking_df.head(20).iterrows():
            print(f"     {ranking_df.index.get_loc(i)+1:2d}. {row['Característica']}: {row['Score_Combinado']:.4f}")
        
        return ranking_df
    
    def recommend_features_by_category(self, ranking_df):
        """
        Recomienda características por categoría
        """
        print("\n📋 RECOMENDACIONES POR CATEGORÍA:")
        
        # Categorizar características
        mean_features = [col for col in self.X.columns if '_mean' in col]
        se_features = [col for col in self.X.columns if '_se' in col]
        worst_features = [col for col in self.X.columns if '_worst' in col]
        
        print(f"   📊 Características _mean (promedio):")
        mean_ranking = ranking_df[ranking_df['Característica'].isin(mean_features)].head(5)
        for i, row in mean_ranking.iterrows():
            print(f"     • {row['Característica']}: {row['Score_Combinado']:.4f}")
        
        print(f"\n   📊 Características _se (error estándar):")
        se_ranking = ranking_df[ranking_df['Característica'].isin(se_features)].head(5)
        for i, row in se_ranking.iterrows():
            print(f"     • {row['Característica']}: {row['Score_Combinado']:.4f}")
        
        print(f"\n   📊 Características _worst (peor):")
        worst_ranking = ranking_df[ranking_df['Característica'].isin(worst_features)].head(5)
        for i, row in worst_ranking.iterrows():
            print(f"     • {row['Característica']}: {row['Score_Combinado']:.4f}")
        
        return mean_ranking, se_ranking, worst_ranking
    
    def create_visualizations(self, ranking_df):
        """
        Crea visualizaciones del análisis de características
        """
        print("\n📊 Creando visualizaciones...")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Top 15 características por correlación
        top_corr = self.correlations.head(15)
        axes[0, 0].barh(range(len(top_corr)), top_corr.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_yticks(range(len(top_corr)))
        axes[0, 0].set_yticklabels(top_corr.index, fontsize=10)
        axes[0, 0].set_title('Top 15 Características por Correlación', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Correlación Absoluta')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Top 15 características por importancia
        top_importance = self.feature_importance.head(15)
        axes[0, 1].barh(range(len(top_importance)), top_importance.values, color='lightcoral', alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_importance)))
        axes[0, 1].set_yticklabels(top_importance.index, fontsize=10)
        axes[0, 1].set_title('Top 15 Características por Importancia (RF)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Importancia')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Ranking combinado
        top_combined = ranking_df.head(15)
        axes[1, 0].barh(range(len(top_combined)), top_combined['Score_Combinado'].values, 
                       color='lightgreen', alpha=0.7)
        axes[1, 0].set_yticks(range(len(top_combined)))
        axes[1, 0].set_yticklabels(top_combined['Característica'], fontsize=10)
        axes[1, 0].set_title('Top 15 Características (Ranking Combinado)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Score Combinado')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Distribución por categoría
        categories = ['_mean', '_se', '_worst']
        category_counts = [len([col for col in self.X.columns if cat in col]) for cat in categories]
        category_names = ['Mean', 'SE', 'Worst']
        
        axes[1, 1].pie(category_counts, labels=category_names, autopct='%1.1f%%', 
                       colors=['lightblue', 'lightcoral', 'lightgreen'], startangle=90)
        axes[1, 1].set_title('Distribución de Características por Categoría', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar gráficos
        output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/feature_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Gráficos guardados en: {output_path}")
    
    def recommend_final_features(self, ranking_df, n_features=15):
        """
        Recomienda las características finales para el modelo
        """
        print(f"\n🎯 RECOMENDACIÓN FINAL DE CARACTERÍSTICAS ({n_features} características):")
        
        # Seleccionar top características
        top_features = ranking_df.head(n_features)['Característica'].tolist()
        
        print(f"   📊 Características recomendadas:")
        for i, feature in enumerate(top_features, 1):
            score = ranking_df[ranking_df['Característica'] == feature]['Score_Combinado'].iloc[0]
            print(f"     {i:2d}. {feature} (score: {score:.4f})")
        
        # Análisis por categoría
        mean_count = len([f for f in top_features if '_mean' in f])
        se_count = len([f for f in top_features if '_se' in f])
        worst_count = len([f for f in top_features if '_worst' in f])
        
        print(f"\n   📈 Distribución por categoría:")
        print(f"     • _mean: {mean_count} características")
        print(f"     • _se: {se_count} características")
        print(f"     • _worst: {worst_count} características")
        
        # Guardar recomendaciones
        recommendations_df = ranking_df.head(n_features)
        output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/training/results/feature_recommendations.csv'
        recommendations_df.to_csv(output_path, index=False)
        
        print(f"\n   ✅ Recomendaciones guardadas en: {output_path}")
        
        return top_features, recommendations_df

def main():
    """
    Función principal para analizar las columnas
    """
    print("🔍 INICIANDO ANÁLISIS DE COLUMNAS")
    print("="*60)
    
    # Inicializar analizador
    file_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/raw/text/Cancer_Data.csv'
    analyzer = ColumnAnalyzer(file_path)
    
    try:
        # Cargar datos
        X, y = analyzer.load_data()
        
        # Analizar correlaciones
        correlations = analyzer.analyze_correlations()
        
        # Analizar importancia
        feature_importance = analyzer.analyze_feature_importance()
        
        # Análisis estadístico
        f_scores, mi_scores = analyzer.analyze_statistical_tests()
        
        # Crear ranking combinado
        ranking_df = analyzer.create_feature_ranking()
        
        # Recomendaciones por categoría
        mean_ranking, se_ranking, worst_ranking = analyzer.recommend_features_by_category(ranking_df)
        
        # Crear visualizaciones
        analyzer.create_visualizations(ranking_df)
        
        # Recomendación final
        top_features, recommendations_df = analyzer.recommend_final_features(ranking_df, n_features=15)
        
        print(f"\n✅ ANÁLISIS DE COLUMNAS COMPLETADO")
        print(f"   • {len(X.columns)} características analizadas")
        print(f"   • {len(top_features)} características recomendadas")
        print(f"   • Archivos generados: visualizaciones y recomendaciones CSV")
        
        return analyzer, top_features, recommendations_df
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    analyzer, top_features, recommendations = main()
