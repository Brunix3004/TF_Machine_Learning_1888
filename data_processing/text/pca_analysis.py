#!/usr/bin/env python3
"""
Script específico para análisis PCA del dataset Cancer_Data.csv
Analiza la reducción de dimensionalidad y selección de características
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class PCAAnalyzer:
    """
    Clase para realizar análisis PCA completo del dataset de cáncer
    """
    
    def __init__(self, file_path):
        """
        Inicializa el analizador PCA
        """
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.pca = None
        self.X_pca = None
        
    def load_data(self):
        """
        Carga y prepara los datos
        """
        print("📊 Cargando dataset...")
        self.df = pd.read_csv(self.file_path)
        
        # Eliminar columnas con valores NaN (como 'Unnamed: 32')
        self.df = self.df.dropna(axis=1)
        
        # Separar características y variable objetivo
        self.X = self.df.drop(['id', 'diagnosis'], axis=1)
        self.y = self.df['diagnosis']
        
        print(f"   ✅ Dataset cargado: {self.X.shape[0]} muestras, {self.X.shape[1]} características")
        print(f"   🔍 Columnas eliminadas con NaN: {self.df.shape[1] - self.X.shape[1] - 2}")
        return self.X, self.y
    
    def preprocess_data(self):
        """
        Preprocesa los datos (estandarización)
        """
        print("🔧 Preprocesando datos...")
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        print("   ✅ Datos estandarizados")
        return self.X_scaled
    
    def perform_pca(self, n_components=None):
        """
        Realiza análisis PCA
        """
        print("🎯 Realizando análisis PCA...")
        
        if n_components is None:
            # PCA completo para análisis
            self.pca = PCA()
        else:
            # PCA con número específico de componentes
            self.pca = PCA(n_components=n_components)
        
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        print(f"   ✅ PCA completado: {self.X_pca.shape[1]} componentes")
        return self.X_pca, self.pca
    
    def analyze_variance_explained(self):
        """
        Analiza la varianza explicada por cada componente
        """
        print("\n📈 ANÁLISIS DE VARIANZA EXPLICADA:")
        
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"   • Varianza explicada por los primeros 10 componentes:")
        for i in range(min(10, len(explained_variance_ratio))):
            print(f"     PC{i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)")
        
        # Encontrar componentes para diferentes niveles de varianza
        thresholds = [0.80, 0.90, 0.95, 0.99]
        print(f"\n   • Componentes necesarios para diferentes niveles de varianza:")
        for threshold in thresholds:
            n_components = np.argmax(cumulative_variance >= threshold) + 1
            print(f"     {threshold*100:.0f}%: {n_components} componentes ({cumulative_variance[n_components-1]:.4f})")
        
        return explained_variance_ratio, cumulative_variance
    
    def plot_pca_results(self, explained_variance_ratio, cumulative_variance):
        """
        Crea visualizaciones del análisis PCA
        """
        print("\n📊 Creando visualizaciones...")
        
        # Configurar el estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Varianza explicada individual
        axes[0, 0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 
                       color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Varianza Explicada por Componente', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Componente Principal')
        axes[0, 0].set_ylabel('Varianza Explicada')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Varianza acumulada
        axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                       'ro-', linewidth=2, markersize=4)
        axes[0, 1].axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95%')
        axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90%')
        axes[0, 1].set_title('Varianza Acumulada', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Número de Componentes')
        axes[0, 1].set_ylabel('Varianza Acumulada')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scree plot
        axes[0, 2].plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 
                       'bo-', linewidth=2, markersize=4)
        axes[0, 2].set_title('Scree Plot', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Componente Principal')
        axes[0, 2].set_ylabel('Varianza Explicada')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. PC1 vs PC2
        colors = ['red' if label == 'M' else 'blue' for label in self.y]
        scatter = axes[1, 0].scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=colors, alpha=0.6, s=30)
        axes[1, 0].set_title(f'PC1 vs PC2\n({explained_variance_ratio[0]*100:.1f}% + {explained_variance_ratio[1]*100:.1f}% = {(explained_variance_ratio[0]+explained_variance_ratio[1])*100:.1f}%)', 
                           fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
        axes[1, 0].set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. PC1 vs PC3
        axes[1, 1].scatter(self.X_pca[:, 0], self.X_pca[:, 2], c=colors, alpha=0.6, s=30)
        axes[1, 1].set_title(f'PC1 vs PC3\n({explained_variance_ratio[0]*100:.1f}% + {explained_variance_ratio[2]*100:.1f}% = {(explained_variance_ratio[0]+explained_variance_ratio[2])*100:.1f}%)', 
                           fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
        axes[1, 1].set_ylabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. PC2 vs PC3
        axes[1, 2].scatter(self.X_pca[:, 1], self.X_pca[:, 2], c=colors, alpha=0.6, s=30)
        axes[1, 2].set_title(f'PC2 vs PC3\n({explained_variance_ratio[1]*100:.1f}% + {explained_variance_ratio[2]*100:.1f}% = {(explained_variance_ratio[1]+explained_variance_ratio[2])*100:.1f}%)', 
                           fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
        axes[1, 2].set_ylabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Agregar leyenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Maligno (M)'),
                          Patch(facecolor='blue', label='Benigno (B)')]
        axes[1, 0].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Guardar gráficos
        output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/cancer_pca_detailed_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Gráficos guardados en: {output_path}")
    
    def analyze_feature_contributions(self):
        """
        Analiza las contribuciones de las características originales a los componentes principales
        """
        print("\n🔍 ANÁLISIS DE CONTRIBUCIONES DE CARACTERÍSTICAS:")
        
        # Obtener los componentes principales
        components = self.pca.components_
        feature_names = self.X.columns
        
        # Analizar las contribuciones a los primeros 3 componentes
        for i in range(min(3, components.shape[0])):
            print(f"\n   PC{i+1} - Top 10 características más importantes:")
            pc_contributions = np.abs(components[i])
            top_indices = np.argsort(pc_contributions)[-10:][::-1]
            
            for j, idx in enumerate(top_indices, 1):
                feature = feature_names[idx]
                contribution = pc_contributions[idx]
                print(f"     {j:2d}. {feature}: {contribution:.4f}")
    
    def recommend_components_for_svm(self, cumulative_variance):
        """
        Recomienda el número de componentes para usar en SVM
        """
        print("\n🤖 RECOMENDACIONES PARA MODELO SVM:")
        
        # Encontrar número de componentes para diferentes niveles
        n_90 = np.argmax(cumulative_variance >= 0.90) + 1
        n_95 = np.argmax(cumulative_variance >= 0.95) + 1
        n_99 = np.argmax(cumulative_variance >= 0.99) + 1
        
        print(f"   • Para 90% de varianza: {n_90} componentes")
        print(f"   • Para 95% de varianza: {n_95} componentes")
        print(f"   • Para 99% de varianza: {n_99} componentes")
        
        # Recomendación basada en el análisis
        if n_95 <= 10:
            recommended = n_95
            reason = "95% de varianza con pocos componentes"
        elif n_90 <= 5:
            recommended = n_90
            reason = "90% de varianza con muy pocos componentes"
        else:
            recommended = min(10, n_95)
            reason = "Balance entre varianza y complejidad"
        
        print(f"\n   🎯 RECOMENDACIÓN: {recommended} componentes")
        print(f"   📝 Razón: {reason}")
        print(f"   📊 Varianza explicada: {cumulative_variance[recommended-1]:.4f} ({cumulative_variance[recommended-1]*100:.2f}%)")
        
        return recommended
    
    def generate_pca_report(self, explained_variance_ratio, cumulative_variance):
        """
        Genera un reporte completo del análisis PCA
        """
        print("\n" + "="*60)
        print("REPORTE COMPLETO DE ANÁLISIS PCA")
        print("="*60)
        
        print(f"\n📊 RESUMEN ESTADÍSTICO:")
        print(f"   • Total de características originales: {len(self.X.columns)}")
        print(f"   • Total de muestras: {self.X.shape[0]}")
        print(f"   • Componentes principales generados: {len(explained_variance_ratio)}")
        
        print(f"\n📈 VARIANZA EXPLICADA:")
        print(f"   • Primer componente: {explained_variance_ratio[0]:.4f} ({explained_variance_ratio[0]*100:.2f}%)")
        print(f"   • Primeros 2 componentes: {cumulative_variance[1]:.4f} ({cumulative_variance[1]*100:.2f}%)")
        print(f"   • Primeros 3 componentes: {cumulative_variance[2]:.4f} ({cumulative_variance[2]*100:.2f}%)")
        print(f"   • Primeros 5 componentes: {cumulative_variance[4]:.4f} ({cumulative_variance[4]*100:.2f}%)")
        
        # Guardar resultados en CSV
        results_df = pd.DataFrame({
            'Componente': range(1, len(explained_variance_ratio) + 1),
            'Varianza_Explicada': explained_variance_ratio,
            'Varianza_Acumulada': cumulative_variance
        })
        
        output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/pca_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n   ✅ Resultados guardados en: {output_path}")
        
        return results_df

def main():
    """
    Función principal para ejecutar el análisis PCA
    """
    print("🎯 INICIANDO ANÁLISIS PCA DEL DATASET DE CÁNCER")
    print("="*60)
    
    # Inicializar analizador
    file_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/raw/text/Cancer_Data.csv'
    analyzer = PCAAnalyzer(file_path)
    
    try:
        # Cargar datos
        X, y = analyzer.load_data()
        
        # Preprocesar
        X_scaled = analyzer.preprocess_data()
        
        # Realizar PCA completo
        X_pca, pca = analyzer.perform_pca()
        
        # Analizar varianza
        explained_variance_ratio, cumulative_variance = analyzer.analyze_variance_explained()
        
        # Crear visualizaciones
        analyzer.plot_pca_results(explained_variance_ratio, cumulative_variance)
        
        # Analizar contribuciones
        analyzer.analyze_feature_contributions()
        
        # Recomendar componentes para SVM
        recommended_components = analyzer.recommend_components_for_svm(cumulative_variance)
        
        # Generar reporte
        results_df = analyzer.generate_pca_report(explained_variance_ratio, cumulative_variance)
        
        print(f"\n✅ ANÁLISIS PCA COMPLETADO EXITOSAMENTE")
        print(f"   • Componentes recomendados para SVM: {recommended_components}")
        print(f"   • Varianza explicada: {cumulative_variance[recommended_components-1]:.4f}")
        print(f"   • Archivos generados: visualizaciones y resultados CSV")
        
        return analyzer, recommended_components
        
    except Exception as e:
        print(f"❌ Error durante el análisis PCA: {str(e)}")
        return None, None

if __name__ == "__main__":
    analyzer, recommended_components = main()
