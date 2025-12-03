#!/usr/bin/env python3
"""
Script para crear tabla visual de métricas Train vs Validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Carga y prepara los datos
    """
    print("📊 Cargando datos...")
    
    file_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/raw/text/Cancer_Data.csv'
    df = pd.read_csv(file_path)
    df = df.dropna(axis=1)
    
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    y = (y == 'M').astype(int)
    
    return X, y

def get_metrics():
    """
    Obtiene las métricas del modelo
    """
    print("🤖 Entrenando modelo y obteniendo métricas...")
    
    # Cargar datos
    X, y = load_and_prepare_data()
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocesar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Entrenar modelo
    svm_model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train_pca, y_train)
    
    # Métricas en test
    y_pred_test = svm_model.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    # Métricas en validación cruzada
    cv_accuracy = cross_val_score(svm_model, X_train_pca, y_train, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(svm_model, X_train_pca, y_train, cv=5, scoring='precision')
    cv_recall = cross_val_score(svm_model, X_train_pca, y_train, cv=5, scoring='recall')
    cv_f1 = cross_val_score(svm_model, X_train_pca, y_train, cv=5, scoring='f1')
    
    return {
        'test': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        },
        'validation': {
            'accuracy': cv_accuracy.mean(),
            'precision': cv_precision.mean(),
            'recall': cv_recall.mean(),
            'f1': cv_f1.mean()
        }
    }

def create_metrics_table_visual(metrics):
    """
    Crea tabla visual de métricas
    """
    print("📊 Creando tabla visual de métricas...")
    
    # Crear DataFrame
    data = {
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Train': [
            f"{metrics['test']['accuracy']:.4f}",
            f"{metrics['test']['precision']:.4f}",
            f"{metrics['test']['recall']:.4f}",
            f"{metrics['test']['f1']:.4f}"
        ],
        'Validation': [
            f"{metrics['validation']['accuracy']:.4f}",
            f"{metrics['validation']['precision']:.4f}",
            f"{metrics['validation']['recall']:.4f}",
            f"{metrics['validation']['f1']:.4f}"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Crear tabla visual
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Crear tabla
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2)
    
    # Colores
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if j == 0:  # Columna de métricas
                table[(i, j)].set_facecolor('#E8F5E8')
            else:  # Columnas de valores
                table[(i, j)].set_facecolor('#F0F8FF')
    
    plt.title('Métricas del Modelo SVM - Train vs Validation', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Guardar tabla
    output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/svm_metrics_table_visual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Tabla visual guardada en: {output_path}")
    
    return df

def create_metrics_comparison_plot(metrics):
    """
    Crea gráfico de comparación de métricas
    """
    print("📊 Creando gráfico de comparación...")
    
    # Preparar datos
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [
        metrics['test']['accuracy'],
        metrics['test']['precision'],
        metrics['test']['recall'],
        metrics['test']['f1']
    ]
    val_values = [
        metrics['validation']['accuracy'],
        metrics['validation']['precision'],
        metrics['validation']['recall'],
        metrics['validation']['f1']
    ]
    
    # Crear gráfico
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', 
                   color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, val_values, width, label='Validation', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Métricas', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Comparación de Métricas - Train vs Validation', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Agregar valores en las barras
    for i, (v1, v2) in enumerate(zip(train_values, val_values)):
        ax.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/svm_metrics_comparison_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Gráfico de comparación guardado en: {output_path}")

def main():
    """
    Función principal
    """
    print("📊 CREANDO TABLA DE MÉTRICAS VISUAL")
    print("="*50)
    
    try:
        # Obtener métricas
        metrics = get_metrics()
        
        # Crear tabla visual
        print("\n" + "="*30)
        print("TABLA VISUAL")
        print("="*30)
        df = create_metrics_table_visual(metrics)
        
        # Crear gráfico de comparación
        print("\n" + "="*30)
        print("GRÁFICO DE COMPARACIÓN")
        print("="*30)
        create_metrics_comparison_plot(metrics)
        
        # Mostrar tabla en consola
        print("\n📋 TABLA DE MÉTRICAS:")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50)
        
        print(f"\n✅ TABLA DE MÉTRICAS CREADA EXITOSAMENTE")
        print(f"   • Tabla visual: svm_metrics_table_visual.png")
        print(f"   • Gráfico comparativo: svm_metrics_comparison_chart.png")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    df = main()
