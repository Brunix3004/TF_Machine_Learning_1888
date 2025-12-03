#!/usr/bin/env python3
"""
Script para evaluar modelos CNN y calcular la probabilidad de falsos negativos.
Esto es crítico en diagnóstico médico ya que un falso negativo significa
que un caso maligno es clasificado incorrectamente como benigno.
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets" / "breast_cancer" / "CNN_data" / "BreaKHis 400X"
MODELS_DIR = BASE_DIR / "models" / "saved"
RESULTS_DIR = BASE_DIR / "training" / "experiments" / "results"

# Parámetros
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
SEED = 42

print("=" * 70)
print("EVALUACIÓN DE PROBABILIDAD DE FALSOS NEGATIVOS EN MODELOS CNN")
print("=" * 70)
print(f"\nDirectorio de datos: {DATA_DIR}")
print(f"Directorio de modelos: {MODELS_DIR}\n")

# Verificar que existe el directorio de datos
if not DATA_DIR.exists():
    print(f"ERROR: No se encuentra el directorio {DATA_DIR}")
    sys.exit(1)

# ============================================================
# 1. CARGAR DATOS DE TEST
# ============================================================

print("Cargando dataset de test...")
test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False  # Importante: no mezclar para mantener orden
)

class_names = test_ds.class_names
print(f"Clases detectadas: {class_names}")
print(f"Total de imágenes en test: {len(test_ds) * BATCH_SIZE}\n")

# Convertir dataset a arrays para evaluación
print("Convirtiendo dataset a arrays...")
y_true = []
y_pred_no_aug = []
y_pred_with_aug = []
y_pred_proba_no_aug = []
y_pred_proba_with_aug = []

# Obtener etiquetas verdaderas
for images, labels in test_ds:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)

# ============================================================
# 2. CARGAR Y EVALUAR MODELOS
# ============================================================

def evaluate_model(model_path, model_name):
    """Evalúa un modelo y retorna predicciones"""
    print(f"\n{'='*70}")
    print(f"EVALUANDO: {model_name}")
    print(f"{'='*70}")
    
    if not model_path.exists():
        print(f"⚠️  Modelo no encontrado: {model_path}")
        return None, None
    
    try:
        model = keras.models.load_model(str(model_path))
        print(f"✓ Modelo cargado exitosamente")
        
        # Obtener predicciones
        print("Generando predicciones...")
        y_pred_proba = []
        y_pred = []
        
        for images, _ in test_ds:
            proba = model.predict(images, verbose=0)
            y_pred_proba.extend(proba.flatten())
            y_pred.extend((proba >= 0.5).astype(int).flatten())
        
        y_pred_proba = np.array(y_pred_proba)
        y_pred = np.array(y_pred)
        
        return y_pred, y_pred_proba
    except Exception as e:
        print(f"✗ Error al cargar modelo: {e}")
        return None, None

# Evaluar modelo sin augmentation
model_no_aug_path = MODELS_DIR / "cnn_no_augmentation_final.keras"
y_pred_no_aug, y_pred_proba_no_aug = evaluate_model(model_no_aug_path, "Modelo SIN Data Augmentation")

# Evaluar modelo con augmentation
model_with_aug_path = MODELS_DIR / "cnn_with_augmentation_final.keras"
y_pred_with_aug, y_pred_proba_with_aug = evaluate_model(model_with_aug_path, "Modelo CON Data Augmentation")

# ============================================================
# 3. CALCULAR MÉTRICAS DE FALSOS NEGATIVOS
# ============================================================

def analyze_false_negatives(y_true, y_pred, y_pred_proba, model_name):
    """Analiza en detalle los falsos negativos"""
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE FALSOS NEGATIVOS: {model_name}")
    print(f"{'='*70}\n")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Para clasificación binaria: [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        print("⚠️  Error: Matriz de confusión no tiene forma esperada")
        return None
    
    # Calcular métricas
    total = len(y_true)
    total_malignos = TP + FN  # Casos realmente malignos
    total_benignos = TN + FP  # Casos realmente benignos
    
    # Probabilidad de falso negativo (FNR - False Negative Rate)
    fnr = FN / total_malignos if total_malignos > 0 else 0
    
    # Recall (Sensibilidad) - complemento de FNR
    recall = TP / total_malignos if total_malignos > 0 else 0
    
    # Especificidad
    specificity = TN / total_benignos if total_benignos > 0 else 0
    
    # Precisión
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Accuracy
    accuracy = (TP + TN) / total if total > 0 else 0
    
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("MATRIZ DE CONFUSIÓN:")
    print("-" * 70)
    print("                Predicción")
    print("              Benigno  Maligno")
    print(f"Real Benigno    {TN:4d}     {FP:4d}")
    print(f"Real Maligno    {FN:4d}     {TP:4d}")
    print()
    
    print("DESGLOSE:")
    print("-" * 70)
    print(f"  TN (True Negative):  {TN:4d} - Correctamente identificados como benignos")
    print(f"  FP (False Positive): {FP:4d} - Incorrectamente identificados como malignos")
    print(f"  FN (False Negative): {FN:4d} - ⚠️  INCORRECTAMENTE identificados como benignos (¡CRÍTICO!)")
    print(f"  TP (True Positive):  {TP:4d} - Correctamente identificados como malignos")
    print()
    
    print("ANÁLISIS DE FALSOS NEGATIVOS (CRÍTICO EN DIAGNÓSTICO MÉDICO):")
    print("-" * 70)
    print(f"Total de casos malignos reales: {total_malignos}")
    print(f"Falsos negativos (FN): {FN}")
    print()
    
    print("PROBABILIDAD DE FALSO NEGATIVO (False Negative Rate - FNR):")
    print("-" * 70)
    print(f"  FNR = FN / (FN + TP) = {FN} / {total_malignos} = {fnr:.4f} ({fnr*100:.2f}%)")
    print()
    print(f"  ⚠️  Esto significa que hay un {fnr*100:.2f}% de probabilidad de que")
    print(f"     un caso maligno sea clasificado incorrectamente como benigno.")
    print()
    
    print("RECALL (Sensibilidad) - Complemento de FNR:")
    print("-" * 70)
    print(f"  Recall = TP / (TP + FN) = {TP} / {total_malignos} = {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Recall = 1 - FNR = 1 - {fnr:.4f} = {1-fnr:.4f}")
    print()
    print(f"  ✓ Esto significa que el modelo detecta correctamente el {recall*100:.2f}%")
    print(f"    de los casos malignos.")
    print()
    
    print("OTRAS MÉTRICAS:")
    print("-" * 70)
    print(f"  Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:    {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Specificity:  {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  F1-Score:     {f1:.4f}")
    print()
    
    # Análisis de probabilidades de los falsos negativos
    if FN > 0:
        fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
        fn_probas = y_pred_proba[fn_indices]
        
        print("ANÁLISIS DE PROBABILIDADES DE LOS FALSOS NEGATIVOS:")
        print("-" * 70)
        print(f"  Número de falsos negativos: {FN}")
        print(f"  Probabilidad promedio de los FN: {fn_probas.mean():.4f}")
        print(f"  Probabilidad mínima de los FN: {fn_probas.min():.4f}")
        print(f"  Probabilidad máxima de los FN: {fn_probas.max():.4f}")
        print(f"  Desviación estándar: {fn_probas.std():.4f}")
        print()
        print(f"  Nota: Valores cercanos a 0.5 indican casos 'borderline'")
        print(f"        que podrían requerir revisión manual.")
        print()
    
    return {
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
        'FNR': fnr, 'Recall': recall, 'Accuracy': accuracy,
        'Precision': precision, 'Specificity': specificity, 'F1': f1,
        'total_malignos': total_malignos, 'total_benignos': total_benignos
    }

# Analizar ambos modelos
results = {}

if y_pred_no_aug is not None:
    results['no_aug'] = analyze_false_negatives(
        y_true, y_pred_no_aug, y_pred_proba_no_aug, 
        "Modelo SIN Data Augmentation"
    )

if y_pred_with_aug is not None:
    results['with_aug'] = analyze_false_negatives(
        y_true, y_pred_with_aug, y_pred_proba_with_aug,
        "Modelo CON Data Augmentation"
    )

# ============================================================
# 4. COMPARACIÓN DE MODELOS
# ============================================================

if len(results) == 2:
    print(f"\n{'='*70}")
    print("COMPARACIÓN DE MODELOS")
    print(f"{'='*70}\n")
    
    no_aug = results['no_aug']
    with_aug = results['with_aug']
    
    print(f"{'Métrica':<25} {'Sin Aug':<20} {'Con Aug':<20} {'Mejora':<15}")
    print("-" * 80)
    
    # FNR
    fnr_improvement = no_aug['FNR'] - with_aug['FNR']
    print(f"{'FNR (Falsos Negativos)':<25} {no_aug['FNR']:<20.4f} {with_aug['FNR']:<20.4f} {fnr_improvement:+.4f}")
    
    # Recall
    recall_improvement = with_aug['Recall'] - no_aug['Recall']
    print(f"{'Recall (Sensibilidad)':<25} {no_aug['Recall']:<20.4f} {with_aug['Recall']:<20.4f} {recall_improvement:+.4f}")
    
    # Accuracy
    acc_improvement = with_aug['Accuracy'] - no_aug['Accuracy']
    print(f"{'Accuracy':<25} {no_aug['Accuracy']:<20.4f} {with_aug['Accuracy']:<20.4f} {acc_improvement:+.4f}")
    
    # Precision
    prec_improvement = with_aug['Precision'] - no_aug['Precision']
    print(f"{'Precision':<25} {no_aug['Precision']:<20.4f} {with_aug['Precision']:<20.4f} {prec_improvement:+.4f}")
    
    # F1
    f1_improvement = with_aug['F1'] - no_aug['F1']
    print(f"{'F1-Score':<25} {no_aug['F1']:<20.4f} {with_aug['F1']:<20.4f} {f1_improvement:+.4f}")
    
    print("-" * 80)
    print()
    
    print("RESUMEN:")
    print("-" * 70)
    if fnr_improvement > 0:
        print(f"✓ El modelo CON augmentation reduce los falsos negativos en {fnr_improvement*100:.2f}%")
        print(f"  (Mejor para diagnóstico médico)")
    elif fnr_improvement < 0:
        print(f"⚠️  El modelo SIN augmentation tiene {abs(fnr_improvement)*100:.2f}% menos falsos negativos")
    else:
        print("  Ambos modelos tienen la misma tasa de falsos negativos")
    print()

# ============================================================
# 5. VISUALIZACIÓN
# ============================================================

if len(results) >= 1:
    print("Generando visualizaciones...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Matriz de confusión - Modelo sin augmentation
    if 'no_aug' in results:
        cm_no_aug = np.array([
            [results['no_aug']['TN'], results['no_aug']['FP']],
            [results['no_aug']['FN'], results['no_aug']['TP']]
        ])
        sns.heatmap(cm_no_aug, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Benigno', 'Maligno'],
                   yticklabels=['Benigno', 'Maligno'])
        axes[0, 0].set_title('Matriz de Confusión - SIN Augmentation\n' + 
                           f"FNR: {results['no_aug']['FNR']*100:.2f}% | Recall: {results['no_aug']['Recall']*100:.2f}%")
        axes[0, 0].set_ylabel('Real')
        axes[0, 0].set_xlabel('Predicción')
    
    # Matriz de confusión - Modelo con augmentation
    if 'with_aug' in results:
        cm_with_aug = np.array([
            [results['with_aug']['TN'], results['with_aug']['FP']],
            [results['with_aug']['FN'], results['with_aug']['TP']]
        ])
        sns.heatmap(cm_with_aug, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
                   xticklabels=['Benigno', 'Maligno'],
                   yticklabels=['Benigno', 'Maligno'])
        axes[0, 1].set_title('Matriz de Confusión - CON Augmentation\n' + 
                           f"FNR: {results['with_aug']['FNR']*100:.2f}% | Recall: {results['with_aug']['Recall']*100:.2f}%")
        axes[0, 1].set_ylabel('Real')
        axes[0, 1].set_xlabel('Predicción')
    
    # Comparación de métricas clave
    if len(results) == 2:
        metrics = ['FNR', 'Recall', 'Accuracy', 'Precision']
        no_aug_vals = [results['no_aug'][m] for m in metrics]
        with_aug_vals = [results['with_aug'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        axes[1, 0].bar(x - width/2, no_aug_vals, width, label='Sin Augmentation', alpha=0.8)
        axes[1, 0].bar(x + width/2, with_aug_vals, width, label='Con Augmentation', alpha=0.8)
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Comparación de Métricas')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        axes[1, 0].grid(True, axis='y')
        axes[1, 0].set_ylim([0, 1])
        
        # Mejoras
        improvements = [
            results['no_aug']['FNR'] - results['with_aug']['FNR'],  # Reducción de FNR
            results['with_aug']['Recall'] - results['no_aug']['Recall'],
            results['with_aug']['Accuracy'] - results['no_aug']['Accuracy'],
            results['with_aug']['Precision'] - results['no_aug']['Precision']
        ]
        colors = ['green' if x > 0 else 'red' for x in improvements]
        axes[1, 1].bar(metrics, improvements, color=colors, alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_ylabel('Mejora')
        axes[1, 1].set_title('Mejora con Data Augmentation')
        axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plot_file = RESULTS_DIR / "cnn_false_negatives_analysis.png"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en: {plot_file}")
    plt.close()

print(f"\n{'='*70}")
print("EVALUACIÓN COMPLETADA")
print(f"{'='*70}\n")
