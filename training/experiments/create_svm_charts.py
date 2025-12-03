#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para crear gráficos comparativos de métricas SVM
Lee los datos de los archivos CSV y genera visualizaciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'training', 'experiments', 'results')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualization', 'plots')

os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

def load_data():
    """Carga los datos de resultados"""
    csv_path = os.path.join(RESULTS_DIR, 'svm_metrics_comparison_real.csv')
    
    if not os.path.exists(csv_path):
        print("Error: No se encontró el archivo de resultados")
        print("Ejecuta primero: compare_svm_metrics.py")
        return None
    
    df = pd.read_csv(csv_path)
    print("Datos cargados: {} estrategias".format(len(df)))
    return df

def create_detailed_charts(df):
    """Crea gráficos detallados comparativos"""
    print("\nCreando gráficos detallados...")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 12))
    
    estrategias = df['Estrategia'].str.replace('\n', ' ').values
    
    # 1. Métricas principales (Accuracy, Precision, Recall, F1)
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(estrategias))
    width = 0.2
    ax1.bar(x - 1.5*width, df['Accuracy'], width, label='Accuracy', alpha=0.8, color='#2196F3')
    ax1.bar(x - 0.5*width, df['Precision'], width, label='Precision', alpha=0.8, color='#4CAF50')
    ax1.bar(x + 0.5*width, df['Recall'], width, label='Recall', alpha=0.8, color='#FF9800')
    ax1.bar(x + 1.5*width, df['F1-Score'], width, label='F1-Score', alpha=0.8, color='#9C27B0')
    ax1.set_xlabel('Estrategia', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Porcentaje (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Métricas Principales por Estrategia', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(estrategias, rotation=15, ha='right', fontsize=9)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([85, 105])
    
    # 2. FNR (Tasa de Falsos Negativos) - CRÍTICO
    ax2 = plt.subplot(2, 3, 2)
    colors = ['#d32f2f' if fnr > 5 else '#ff9800' if fnr > 3 else '#4caf50' for fnr in df['FNR']]
    bars = ax2.bar(estrategias, df['FNR'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Estrategia', fontsize=11, fontweight='bold')
    ax2.set_ylabel('FNR (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Tasa de Falsos Negativos (FNR) - CRÍTICO', fontsize=13, fontweight='bold', color='red')
    ax2.set_xticklabels(estrategias, rotation=15, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, df['FNR']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                '{:.2f}%'.format(value), ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # 3. Recall vs FNR (Trade-off)
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(df['FNR'], df['Recall'], s=400, alpha=0.7, 
                         c=range(len(df)), cmap='viridis', edgecolors='black', linewidth=2)
    for i, (fnr, recall) in enumerate(zip(df['FNR'], df['Recall'])):
        ax3.annotate('E{}'.format(i+1), (fnr, recall), 
                    fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    ax3.set_xlabel('FNR (%) - Menor es mejor', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Recall (%) - Mayor es mejor', fontsize=11, fontweight='bold')
    ax3.set_title('Trade-off: Recall vs FNR', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    # Leyenda
    for i, estrategia in enumerate(estrategias):
        ax3.text(df.iloc[i]['FNR'], df.iloc[i]['Recall'] + 1.5, 
                'E{}: {}'.format(i+1, estrategia.split('(')[0].strip()), 
                fontsize=8, ha='center', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', alpha=0.7))
    
    # 4. Falsos Negativos y Falsos Positivos
    ax4 = plt.subplot(2, 3, 4)
    x = np.arange(len(estrategias))
    width = 0.35
    bars1 = ax4.bar(x - width/2, df['FN'], width, label='Falsos Negativos', 
                    color='#f44336', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, df['FP'], width, label='Falsos Positivos', 
                    color='#ff9800', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Estrategia', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cantidad de Casos', fontsize=11, fontweight='bold')
    ax4.set_title('Falsos Negativos vs Falsos Positivos', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(estrategias, rotation=15, ha='right', fontsize=9)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        '{:.0f}'.format(height), ha='center', va='bottom', 
                        fontweight='bold', fontsize=9)
    
    # 5. Matriz de métricas (Heatmap)
    ax5 = plt.subplot(2, 3, 5)
    metrics_for_heatmap = df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'FNR', 'Specificity']].T
    metrics_for_heatmap.columns = ['E{}'.format(i+1) for i in range(len(estrategias))]
    metrics_for_heatmap.index = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'FNR', 'Specificity']
    sns.heatmap(metrics_for_heatmap, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=95, vmin=0, vmax=100, ax=ax5, 
                cbar_kws={'label': 'Porcentaje (%)'}, linewidths=0.5)
    ax5.set_title('Heatmap de Métricas por Estrategia', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Estrategia', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Métrica', fontsize=11, fontweight='bold')
    
    # 6. Comparación de mejoras respecto al baseline
    ax6 = plt.subplot(2, 3, 6)
    baseline = df.iloc[0]
    improvements = {
        'FNR': ((baseline['FNR'] - df['FNR']) / baseline['FNR'] * 100).fillna(0),
        'Recall': ((df['Recall'] - baseline['Recall']) / baseline['Recall'] * 100).fillna(0),
        'FN': ((baseline['FN'] - df['FN']) / baseline['FN'] * 100).fillna(0) if baseline['FN'] > 0 else pd.Series([0]*len(df))
    }
    x = np.arange(len(estrategias))
    width = 0.25
    ax6.bar(x - width, improvements['FNR'], width, label='Mejora FNR (%)', 
           alpha=0.8, color='#4CAF50')
    ax6.bar(x, improvements['Recall'], width, label='Mejora Recall (%)', 
           alpha=0.8, color='#2196F3')
    ax6.bar(x + width, improvements['FN'], width, label='Reducción FN (%)', 
           alpha=0.8, color='#FF9800')
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax6.set_xlabel('Estrategia', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Mejora (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Mejoras Respecto al Modelo Original', fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(estrategias, rotation=15, ha='right', fontsize=9)
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(VISUALIZATIONS_DIR, 'svm_metrics_comparison_detailed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Gráfico detallado guardado: {}".format(output_path))
    return output_path

def create_simple_charts(df):
    """Crea gráficos simplificados"""
    print("Creando gráficos simplificados...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    estrategias = df['Estrategia'].str.replace('\n', ' ').values
    
    # 1. FNR (lo más importante)
    ax1 = axes[0]
    colors = ['#d32f2f' if fnr > 5 else '#ff9800' if fnr > 3 else '#4caf50' for fnr in df['FNR']]
    bars = ax1.bar(estrategias, df['FNR'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('FNR (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Tasa de Falsos Negativos (FNR)', fontsize=13, fontweight='bold')
    ax1.set_xticklabels(estrategias, rotation=15, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, df['FNR']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                '{:.2f}%'.format(value), ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # 2. Recall
    ax2 = axes[1]
    bars = ax2.bar(estrategias, df['Recall'], color='#2196f3', alpha=0.8, 
                  edgecolor='black', linewidth=2)
    ax2.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Recall (Sensibilidad)', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(estrategias, rotation=15, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([85, 100])
    for bar, value in zip(bars, df['Recall']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                '{:.2f}%'.format(value), ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # 3. Falsos Negativos absolutos
    ax3 = axes[2]
    bars = ax3.bar(estrategias, df['FN'], color='#f44336', alpha=0.8, 
                  edgecolor='black', linewidth=2)
    ax3.set_ylabel('Cantidad de FN', fontsize=12, fontweight='bold')
    ax3.set_title('Falsos Negativos (Cantidad)', fontsize=13, fontweight='bold')
    ax3.set_xticklabels(estrategias, rotation=15, ha='right', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, df['FN']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                '{:.0f}'.format(value), ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(VISUALIZATIONS_DIR, 'svm_metrics_comparison_simple.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Gráfico simplificado guardado: {}".format(output_path))
    return output_path

def create_radar_chart(df):
    """Crea gráfico tipo radar/spider para comparar estrategias"""
    print("Creando gráfico tipo radar...")
    
    # Seleccionar métricas clave
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity']
    # Invertir FNR para que mayor sea mejor (100 - FNR)
    df_radar = df.copy()
    df_radar['FNR_inverted'] = 100 - df_radar['FNR']
    metrics.append('FNR_inverted')
    
    # Normalizar a escala 0-100
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Ángulos para cada métrica
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo
    
    # Colores para cada estrategia
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    for idx, row in df_radar.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]  # Cerrar el círculo
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Estrategia'].replace('\n', ' '), 
               color=colors[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Etiquetas
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 
                    'Specificity', '100-FNR']
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
    ax.set_title('Comparación de Estrategias - Gráfico Radar', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(VISUALIZATIONS_DIR, 'svm_metrics_radar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Gráfico radar guardado: {}".format(output_path))
    return output_path

def create_improvement_chart(df):
    """Crea gráfico de mejoras respecto al baseline"""
    print("Creando gráfico de mejoras...")
    
    baseline = df.iloc[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    estrategias = df['Estrategia'].str.replace('\n', ' ').values
    
    # 1. Reducción de FNR
    ax1 = axes[0, 0]
    fnr_reduction = baseline['FNR'] - df['FNR']
    colors = ['#4caf50' if x > 0 else '#f44336' for x in fnr_reduction]
    bars = ax1.bar(estrategias, fnr_reduction, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel('Reducción de FNR (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Reducción de Tasa de Falsos Negativos', fontsize=12, fontweight='bold')
    ax1.set_xticklabels(estrategias, rotation=15, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, fnr_reduction):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                '{:.2f}%'.format(value), ha='center', 
                va='bottom' if height > 0 else 'top', fontweight='bold', fontsize=9)
    
    # 2. Mejora de Recall
    ax2 = axes[0, 1]
    recall_improvement = df['Recall'] - baseline['Recall']
    bars = ax2.bar(estrategias, recall_improvement, color='#2196f3', alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Mejora de Recall (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Mejora de Recall (Sensibilidad)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(estrategias, rotation=15, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, recall_improvement):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                '{:.2f}%'.format(value), ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # 3. Reducción de Falsos Negativos (absolutos)
    ax3 = axes[1, 0]
    fn_reduction = baseline['FN'] - df['FN']
    colors = ['#4caf50' if x > 0 else '#f44336' for x in fn_reduction]
    bars = ax3.bar(estrategias, fn_reduction, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_ylabel('Reducción de FN (casos)', fontsize=11, fontweight='bold')
    ax3.set_title('Reducción de Falsos Negativos (Cantidad)', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(estrategias, rotation=15, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, fn_reduction):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.1),
                '{:.0f}'.format(value), ha='center', 
                va='bottom' if height > 0 else 'top', fontweight='bold', fontsize=9)
    
    # 4. Comparación Accuracy vs FNR (burbujas)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['FNR'], df['Accuracy'], s=df['FN']*200+100, 
                         alpha=0.6, c=range(len(df)), cmap='viridis', 
                         edgecolors='black', linewidth=2)
    for i, (fnr, acc, fn) in enumerate(zip(df['FNR'], df['Accuracy'], df['FN'])):
        ax4.annotate('E{}'.format(i+1), (fnr, acc), fontsize=10, 
                    fontweight='bold', ha='center', va='center', color='white')
    ax4.set_xlabel('FNR (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Accuracy vs FNR (tamaño = cantidad de FN)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(VISUALIZATIONS_DIR, 'svm_improvements_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Gráfico de mejoras guardado: {}".format(output_path))
    return output_path

def main():
    """Función principal"""
    print("="*70)
    print("GENERADOR DE GRÁFICOS COMPARATIVOS SVM")
    print("="*70)
    
    # Cargar datos
    df = load_data()
    if df is None:
        return
    
    print("\nEstrategias encontradas:")
    for i, estrategia in enumerate(df['Estrategia'], 1):
        print("  {}: {}".format(i, estrategia.replace('\n', ' ')))
    
    # Crear gráficos
    try:
        create_detailed_charts(df)
        create_simple_charts(df)
        create_radar_chart(df)
        create_improvement_chart(df)
        
        print("\n" + "="*70)
        print("TODOS LOS GRÁFICOS GENERADOS EXITOSAMENTE")
        print("="*70)
        print("\nArchivos creados en: {}".format(VISUALIZATIONS_DIR))
        print("  - svm_metrics_comparison_detailed.png (6 subgráficos)")
        print("  - svm_metrics_comparison_simple.png (3 gráficos principales)")
        print("  - svm_metrics_radar_chart.png (gráfico tipo radar)")
        print("  - svm_improvements_chart.png (análisis de mejoras)")
        
    except Exception as e:
        print("Error al crear gráficos: {}".format(str(e)))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
