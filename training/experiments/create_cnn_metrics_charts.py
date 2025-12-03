#!/usr/bin/env python3
"""
Script para generar gráficos comparativos de métricas de modelos CNN
con y sin data augmentation, incluyendo análisis de falsos negativos.
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets" / "breast_cancer" / "CNN_data" / "BreaKHis 400X"
MODELS_DIR = BASE_DIR / "models" / "saved"
RESULTS_DIR = BASE_DIR / "training" / "experiments" / "results"
VISUALIZATION_DIR = BASE_DIR / "visualization" / "plots"

# Crear directorios si no existen
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
SEED = 42

# Configurar estilo de matplotlib
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

print("=" * 70)
print("GENERACIÓN DE GRÁFICOS DE MÉTRICAS CNN")
print("=" * 70)
print(f"\nDirectorio de resultados: {RESULTS_DIR}")
print(f"Directorio de visualización: {VISUALIZATION_DIR}\n")

# ============================================================
# 1. CARGAR DATOS Y MODELOS
# ============================================================

def load_model_and_predictions(model_path, test_ds):
    """Carga un modelo y genera predicciones"""
    if not model_path.exists():
        return None, None, None
    
    try:
        model = keras.models.load_model(str(model_path))
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for images, labels in test_ds:
            y_true.extend(labels.numpy())
            proba = model.predict(images, verbose=0)
            y_pred_proba.extend(proba.flatten())
            y_pred.extend((proba >= 0.5).astype(int).flatten())
        
        return np.array(y_true), np.array(y_pred), np.array(y_pred_proba)
    except Exception as e:
        print(f"Error al cargar modelo {model_path}: {e}")
        return None, None, None

# Cargar dataset de test
print("Cargando dataset de test...")
test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

# Cargar resultados del JSON si existe
comparison_file = RESULTS_DIR / "cnn_augmentation_comparison.json"
comparison_data = None
if comparison_file.exists():
    with open(comparison_file, 'r') as f:
        comparison_data = json.load(f)
    print(f"✓ Resultados cargados desde: {comparison_file}")

# Cargar predicciones de los modelos
print("\nCargando modelos y generando predicciones...")
model_no_aug_path = MODELS_DIR / "cnn_no_augmentation_final.keras"
model_with_aug_path = MODELS_DIR / "cnn_with_augmentation_final.keras"

y_true_no_aug, y_pred_no_aug, y_pred_proba_no_aug = load_model_and_predictions(
    model_no_aug_path, test_ds
)
y_true_with_aug, y_pred_with_aug, y_pred_proba_with_aug = load_model_and_predictions(
    model_with_aug_path, test_ds
)

# Si no hay modelos, usar datos del JSON
if y_pred_no_aug is None and comparison_data:
    print("⚠️  Modelos no encontrados, usando datos del JSON...")
    # Recrear predicciones aproximadas desde el JSON (solo para visualización)
    pass

# ============================================================
# 2. CALCULAR MÉTRICAS
# ============================================================

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calcula todas las métricas relevantes"""
    if y_true is None or y_pred is None:
        return None
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        return None
    
    total = len(y_true)
    total_malignos = TP + FN
    total_benignos = TN + FP
    
    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    fnr = FN / total_malignos if total_malignos > 0 else 0
    specificity = TN / total_benignos if total_benignos > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba) if y_pred_proba is not None else (None, None, None)
    roc_auc = auc(fpr, tpr) if fpr is not None and tpr is not None else None
    
    return {
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'fnr': fnr, 'specificity': specificity, 'f1': f1,
        'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
        'total_malignos': total_malignos, 'total_benignos': total_benignos
    }

metrics_no_aug = calculate_metrics(y_true_no_aug, y_pred_no_aug, y_pred_proba_no_aug)
metrics_with_aug = calculate_metrics(y_true_with_aug, y_pred_with_aug, y_pred_proba_with_aug)

# ============================================================
# 3. CREAR GRÁFICOS
# ============================================================

print("\nGenerando gráficos...")

# ========== GRÁFICO 1: Matrices de Confusión ==========
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

if metrics_no_aug:
    cm_no_aug = np.array([
        [metrics_no_aug['TN'], metrics_no_aug['FP']],
        [metrics_no_aug['FN'], metrics_no_aug['TP']]
    ])
    sns.heatmap(cm_no_aug, annot=True, fmt='d', cmap='Blues', ax=axes[0],
               xticklabels=['Benigno', 'Maligno'],
               yticklabels=['Benigno', 'Maligno'],
               cbar_kws={'label': 'Cantidad'})
    axes[0].set_title(f'Matriz de Confusión - SIN Augmentation\n' + 
                     f"FNR: {metrics_no_aug['fnr']*100:.2f}% | Recall: {metrics_no_aug['recall']*100:.2f}%",
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Real', fontsize=12)
    axes[0].set_xlabel('Predicción', fontsize=12)

if metrics_with_aug:
    cm_with_aug = np.array([
        [metrics_with_aug['TN'], metrics_with_aug['FP']],
        [metrics_with_aug['FN'], metrics_with_aug['TP']]
    ])
    sns.heatmap(cm_with_aug, annot=True, fmt='d', cmap='Greens', ax=axes[1],
               xticklabels=['Benigno', 'Maligno'],
               yticklabels=['Benigno', 'Maligno'],
               cbar_kws={'label': 'Cantidad'})
    axes[1].set_title(f'Matriz de Confusión - CON Augmentation\n' + 
                     f"FNR: {metrics_with_aug['fnr']*100:.2f}% | Recall: {metrics_with_aug['recall']*100:.2f}%",
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Real', fontsize=12)
    axes[1].set_xlabel('Predicción', fontsize=12)

plt.tight_layout()
plot_file = VISUALIZATION_DIR / "cnn_confusion_matrices.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Gráfico guardado: {plot_file}")
plt.close()

# ========== GRÁFICO 2: Comparación de Métricas Principales ==========
if metrics_no_aug and metrics_with_aug:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Subplot 1: Métricas principales
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    no_aug_vals = [
        metrics_no_aug['accuracy'],
        metrics_no_aug['precision'],
        metrics_no_aug['recall'],
        metrics_no_aug['f1']
    ]
    with_aug_vals = [
        metrics_with_aug['accuracy'],
        metrics_with_aug['precision'],
        metrics_with_aug['recall'],
        metrics_with_aug['f1']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, no_aug_vals, width, label='Sin Augmentation', 
                          alpha=0.8, color='#3498db')
    bars2 = axes[0, 0].bar(x + width/2, with_aug_vals, width, label='Con Augmentation', 
                          alpha=0.8, color='#2ecc71')
    
    # Añadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=9)
    
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_title('Comparación de Métricas Principales', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_names)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 1.1])
    
    # Subplot 2: Mejoras
    improvements = [
        metrics_with_aug['accuracy'] - metrics_no_aug['accuracy'],
        metrics_with_aug['precision'] - metrics_no_aug['precision'],
        metrics_with_aug['recall'] - metrics_no_aug['recall'],
        metrics_with_aug['f1'] - metrics_no_aug['f1']
    ]
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = axes[0, 1].bar(metrics_names, improvements, color=colors, alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:+.4f}',
                      ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 1].set_ylabel('Mejora', fontsize=12)
    axes[0, 1].set_title('Mejora con Data Augmentation', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, axis='y', alpha=0.3)
    
    # Subplot 3: Análisis de Falsos Negativos
    fn_metrics = ['FNR\n(Falsos Negativos)', 'Recall\n(Sensibilidad)', 'Specificity']
    no_aug_fn = [
        metrics_no_aug['fnr'],
        metrics_no_aug['recall'],
        metrics_no_aug['specificity']
    ]
    with_aug_fn = [
        metrics_with_aug['fnr'],
        metrics_with_aug['recall'],
        metrics_with_aug['specificity']
    ]
    
    x2 = np.arange(len(fn_metrics))
    bars1 = axes[1, 0].bar(x2 - width/2, no_aug_fn, width, label='Sin Augmentation', 
                          alpha=0.8, color='#e74c3c')
    bars2 = axes[1, 0].bar(x2 + width/2, with_aug_fn, width, label='Con Augmentation', 
                          alpha=0.8, color='#27ae60')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=9)
    
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Análisis de Falsos Negativos (Crítico en Diagnóstico)', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x2)
    axes[1, 0].set_xticklabels(fn_metrics)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])
    
    # Subplot 4: Reducción de FNR
    fnr_improvement = metrics_no_aug['fnr'] - metrics_with_aug['fnr']
    recall_improvement = metrics_with_aug['recall'] - metrics_no_aug['recall']
    
    fnr_data = {
        'Sin Augmentation': metrics_no_aug['fnr'],
        'Con Augmentation': metrics_with_aug['fnr']
    }
    
    bars = axes[1, 1].bar(fnr_data.keys(), fnr_data.values(), 
                         color=['#e74c3c', '#27ae60'], alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.4f}\n({height*100:.2f}%)',
                      ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes[1, 1].set_ylabel('False Negative Rate (FNR)', fontsize=12)
    axes[1, 1].set_title(f'Reducción de Falsos Negativos\nMejora: {fnr_improvement*100:.2f}%', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, axis='y', alpha=0.3)
    axes[1, 1].set_ylim([0, max(fnr_data.values()) * 1.3])
    
    # Añadir anotación de mejora
    axes[1, 1].annotate(f'Mejora: {fnr_improvement*100:.2f}%',
                       xy=(1, metrics_with_aug['fnr']),
                       xytext=(0.5, max(fnr_data.values()) * 1.1),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=12, fontweight='bold', color='green',
                       ha='center')
    
    plt.tight_layout()
    plot_file = VISUALIZATION_DIR / "cnn_metrics_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {plot_file}")
    plt.close()

# ========== GRÁFICO 3: Curvas ROC ==========
if metrics_no_aug and metrics_with_aug and metrics_no_aug['fpr'] is not None:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(metrics_no_aug['fpr'], metrics_no_aug['tpr'], 
           label=f'Sin Augmentation (AUC = {metrics_no_aug["roc_auc"]:.4f})',
           linewidth=2, color='#3498db')
    ax.plot(metrics_with_aug['fpr'], metrics_with_aug['tpr'],
           label=f'Con Augmentation (AUC = {metrics_with_aug["roc_auc"]:.4f})',
           linewidth=2, color='#2ecc71')
    ax.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio', linewidth=1)
    
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR / Recall)', fontsize=12)
    ax.set_title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = VISUALIZATION_DIR / "cnn_roc_curves.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {plot_file}")
    plt.close()

# ========== GRÁFICO 4: Radar Chart de Métricas ==========
if metrics_no_aug and metrics_with_aug:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Métricas para el radar chart
    categories = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    N = len(categories)
    
    # Ángulos para cada categoría
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Cerrar el círculo
    
    # Valores
    no_aug_radar = [
        metrics_no_aug['accuracy'],
        metrics_no_aug['precision'],
        metrics_no_aug['recall'],
        metrics_no_aug['specificity'],
        metrics_no_aug['f1']
    ]
    no_aug_radar += no_aug_radar[:1]
    
    with_aug_radar = [
        metrics_with_aug['accuracy'],
        metrics_with_aug['precision'],
        metrics_with_aug['recall'],
        metrics_with_aug['specificity'],
        metrics_with_aug['f1']
    ]
    with_aug_radar += with_aug_radar[:1]
    
    # Plot
    ax.plot(angles, no_aug_radar, 'o-', linewidth=2, label='Sin Augmentation', color='#3498db')
    ax.fill(angles, no_aug_radar, alpha=0.25, color='#3498db')
    
    ax.plot(angles, with_aug_radar, 'o-', linewidth=2, label='Con Augmentation', color='#2ecc71')
    ax.fill(angles, with_aug_radar, alpha=0.25, color='#2ecc71')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    
    ax.set_title('Comparación de Métricas - Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    plot_file = VISUALIZATION_DIR / "cnn_radar_chart.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {plot_file}")
    plt.close()

# ========== GRÁFICO 5: Tabla Resumen de Métricas ==========
if metrics_no_aug and metrics_with_aug:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Crear tabla
    data = [
        ['Accuracy', f"{metrics_no_aug['accuracy']:.4f}", f"{metrics_with_aug['accuracy']:.4f}", 
         f"{metrics_with_aug['accuracy'] - metrics_no_aug['accuracy']:+.4f}"],
        ['Precision', f"{metrics_no_aug['precision']:.4f}", f"{metrics_with_aug['precision']:.4f}",
         f"{metrics_with_aug['precision'] - metrics_no_aug['precision']:+.4f}"],
        ['Recall (Sensibilidad)', f"{metrics_no_aug['recall']:.4f}", f"{metrics_with_aug['recall']:.4f}",
         f"{metrics_with_aug['recall'] - metrics_no_aug['recall']:+.4f}"],
        ['Specificity', f"{metrics_no_aug['specificity']:.4f}", f"{metrics_with_aug['specificity']:.4f}",
         f"{metrics_with_aug['specificity'] - metrics_no_aug['specificity']:+.4f}"],
        ['F1-Score', f"{metrics_no_aug['f1']:.4f}", f"{metrics_with_aug['f1']:.4f}",
         f"{metrics_with_aug['f1'] - metrics_no_aug['f1']:+.4f}"],
        ['FNR (Falsos Negativos)', f"{metrics_no_aug['fnr']:.4f}", f"{metrics_with_aug['fnr']:.4f}",
         f"{metrics_no_aug['fnr'] - metrics_with_aug['fnr']:+.4f}"],
        ['AUC-ROC', f"{metrics_no_aug['roc_auc']:.4f}" if metrics_no_aug['roc_auc'] else 'N/A',
         f"{metrics_with_aug['roc_auc']:.4f}" if metrics_with_aug['roc_auc'] else 'N/A',
         f"{metrics_with_aug['roc_auc'] - metrics_no_aug['roc_auc']:+.4f}" if (metrics_no_aug['roc_auc'] and metrics_with_aug['roc_auc']) else 'N/A'],
    ]
    
    table = ax.table(cellText=data,
                    colLabels=['Métrica', 'Sin Augmentation', 'Con Augmentation', 'Mejora'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Colorear encabezados
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear fila de FNR (importante)
    for i in range(4):
        table[(6, i)].set_facecolor('#f39c12')
        table[(6, i)].set_text_props(weight='bold')
    
    # Colorear mejoras positivas en verde
    for i in range(1, len(data) + 1):
        if i < len(data) + 1:
            improvement = data[i-1][3]
            if improvement.startswith('+'):
                table[(i, 3)].set_facecolor('#d5f4e6')
            elif improvement.startswith('-'):
                table[(i, 3)].set_facecolor('#fadbd8')
    
    ax.set_title('Tabla Comparativa de Métricas - Modelos CNN', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_file = VISUALIZATION_DIR / "cnn_metrics_table.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {plot_file}")
    plt.close()

# ========== GRÁFICO 6: Análisis Detallado de Falsos Negativos ==========
if metrics_no_aug and metrics_with_aug:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Comparación de FNR
    models = ['Sin Augmentation', 'Con Augmentation']
    fnr_values = [metrics_no_aug['fnr'], metrics_with_aug['fnr']]
    colors_fnr = ['#e74c3c', '#27ae60']
    
    bars = axes[0, 0].bar(models, fnr_values, color=colors_fnr, alpha=0.7, width=0.6)
    for i, (bar, val) in enumerate(zip(bars, fnr_values)):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}\n({val*100:.2f}%)',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[0, 0].set_ylabel('False Negative Rate (FNR)', fontsize=12)
    axes[0, 0].set_title('Probabilidad de Falsos Negativos', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, max(fnr_values) * 1.4])
    
    # Subplot 2: Comparación de Recall
    recall_values = [metrics_no_aug['recall'], metrics_with_aug['recall']]
    colors_recall = ['#3498db', '#2ecc71']
    
    bars = axes[0, 1].bar(models, recall_values, color=colors_recall, alpha=0.7, width=0.6)
    for i, (bar, val) in enumerate(zip(bars, recall_values)):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}\n({val*100:.2f}%)',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[0, 1].set_ylabel('Recall (Sensibilidad)', fontsize=12)
    axes[0, 1].set_title('Capacidad de Detectar Casos Malignos', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 1.1])
    
    # Subplot 3: Desglose de Matriz de Confusión - Sin Aug
    cm_no_aug_data = {
        'TN': metrics_no_aug['TN'],
        'FP': metrics_no_aug['FP'],
        'FN': metrics_no_aug['FN'],
        'TP': metrics_no_aug['TP']
    }
    
    bars = axes[1, 0].bar(cm_no_aug_data.keys(), cm_no_aug_data.values(),
                         color=['#3498db', '#f39c12', '#e74c3c', '#27ae60'], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[1, 0].set_ylabel('Cantidad', fontsize=12)
    axes[1, 0].set_title('Matriz de Confusión - Sin Augmentation', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    
    # Subplot 4: Desglose de Matriz de Confusión - Con Aug
    cm_with_aug_data = {
        'TN': metrics_with_aug['TN'],
        'FP': metrics_with_aug['FP'],
        'FN': metrics_with_aug['FN'],
        'TP': metrics_with_aug['TP']
    }
    
    bars = axes[1, 1].bar(cm_with_aug_data.keys(), cm_with_aug_data.values(),
                         color=['#3498db', '#f39c12', '#e74c3c', '#27ae60'], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[1, 1].set_ylabel('Cantidad', fontsize=12)
    axes[1, 1].set_title('Matriz de Confusión - Con Augmentation', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = VISUALIZATION_DIR / "cnn_false_negatives_detailed.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {plot_file}")
    plt.close()

print(f"\n{'='*70}")
print("GENERACIÓN DE GRÁFICOS COMPLETADA")
print(f"{'='*70}")
print(f"\nGráficos guardados en: {VISUALIZATION_DIR}")
print("\nGráficos generados:")
print("  1. cnn_confusion_matrices.png - Matrices de confusión")
print("  2. cnn_metrics_comparison.png - Comparación de métricas")
print("  3. cnn_roc_curves.png - Curvas ROC")
print("  4. cnn_radar_chart.png - Radar chart de métricas")
print("  5. cnn_metrics_table.png - Tabla resumen")
print("  6. cnn_false_negatives_detailed.png - Análisis de falsos negativos")
print()
