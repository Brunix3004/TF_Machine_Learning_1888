#!/usr/bin/env python3
"""
Script para analizar la distribución de clases en el dataset CLAHE
y determinar los dropout rates apropiados.
"""

import sys
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración
DATA_PATH = Path("/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/processed/images_clahe")

def analyze_class_distribution():
    """Analiza la distribución de clases en el dataset."""
    print("=== ANÁLISIS DE DISTRIBUCIÓN DE CLASES ===")
    
    # Analizar conjunto de entrenamiento
    train_images = DATA_PATH / "train" / "images"
    train_labels = DATA_PATH / "train" / "labels"
    
    train_class_counts = Counter()
    train_samples = []
    
    print(f"Analizando conjunto de entrenamiento...")
    for img_path in train_images.glob("*.jpg"):
        label_path = train_labels / f"{img_path.stem}.txt"
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                label_content = f.read().strip()
            
            if label_content:
                lines = label_content.split('\n')
                class_id = int(lines[0].split()[0])
                train_class_counts[class_id] += 1
                train_samples.append((img_path, class_id))
            else:
                train_class_counts[1] += 1  # Other lesion
                train_samples.append((img_path, 1))
        else:
            train_class_counts[1] += 1  # Other lesion
            train_samples.append((img_path, 1))
    
    # Analizar conjunto de validación
    val_images = DATA_PATH / "val" / "images"
    val_labels = DATA_PATH / "val" / "labels"
    
    val_class_counts = Counter()
    val_samples = []
    
    print(f"Analizando conjunto de validación...")
    for img_path in val_images.glob("*.jpg"):
        label_path = val_labels / f"{img_path.stem}.txt"
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                label_content = f.read().strip()
            
            if label_content:
                lines = label_content.split('\n')
                class_id = int(lines[0].split()[0])
                val_class_counts[class_id] += 1
                val_samples.append((img_path, class_id))
            else:
                val_class_counts[1] += 1  # Other lesion
                val_samples.append((img_path, 1))
        else:
            val_class_counts[1] += 1  # Other lesion
            val_samples.append((img_path, 1))
    
    # Combinar resultados
    total_class_counts = train_class_counts + val_class_counts
    total_samples = sum(total_class_counts.values())
    
    print(f"\n=== RESULTADOS ===")
    print(f"Total de muestras: {total_samples}")
    print(f"Muestras de entrenamiento: {len(train_samples)}")
    print(f"Muestras de validación: {len(val_samples)}")
    
    print(f"\nDistribución de clases:")
    class_names = {0: "Nodule/Mass", 1: "Other lesion"}
    
    for class_id in sorted(total_class_counts.keys()):
        count = total_class_counts[class_id]
        percentage = (count / total_samples) * 100
        print(f"  Clase {class_id} ({class_names.get(class_id, 'Unknown')}): {count} muestras ({percentage:.1f}%)")
    
    # Calcular balance
    if len(total_class_counts) > 1:
        counts = list(total_class_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        balance_ratio = max_count / min_count
        print(f"\nBalance de clases:")
        print(f"  Ratio máximo/mínimo: {balance_ratio:.2f}")
        print(f"  {'Balanceado' if balance_ratio < 2.0 else 'Desbalanceado'}")
    
    # Recomendar dropout rates
    print(f"\n=== RECOMENDACIONES DE DROPOUT ===")
    
    if len(total_class_counts) == 2:
        # Dataset binario
        counts = list(total_class_counts.values())
        if counts[0] > counts[1]:
            # Clase 0 es mayoritaria
            print("Dataset binario con clase 0 mayoritaria")
            print("Recomendación: Dropout moderado para evitar overfitting")
            recommended_dropout = [0.3, 0.4, 0.5]
        else:
            # Clase 1 es mayoritaria
            print("Dataset binario con clase 1 mayoritaria")
            print("Recomendación: Dropout moderado para evitar overfitting")
            recommended_dropout = [0.3, 0.4, 0.5]
    else:
        # Dataset multiclase
        print("Dataset multiclase")
        print("Recomendación: Dropout conservador")
        recommended_dropout = [0.2, 0.3, 0.4]
    
    print(f"Dropout rates recomendados: {recommended_dropout}")
    
    # Crear visualización
    create_visualization(total_class_counts, class_names)
    
    return recommended_dropout

def create_visualization(class_counts, class_names):
    """Crea visualizaciones de la distribución de clases."""
    plt.figure(figsize=(12, 5))
    
    # Gráfico de barras
    plt.subplot(1, 2, 1)
    classes = [class_names.get(cid, f'Class {cid}') for cid in sorted(class_counts.keys())]
    counts = [class_counts[cid] for cid in sorted(class_counts.keys())]
    
    bars = plt.bar(classes, counts, color=['skyblue', 'lightcoral'])
    plt.title('Distribución de Clases')
    plt.xlabel('Clases')
    plt.ylabel('Número de Muestras')
    plt.xticks(rotation=45)
    
    # Agregar valores en las barras
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom')
    
    # Gráfico de pastel
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
    plt.title('Proporción de Clases')
    
    plt.tight_layout()
    plt.savefig('/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/training/experiments/hito2/basic_cnn/class_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualización guardada en: class_distribution.png")

if __name__ == "__main__":
    recommended_dropout = analyze_class_distribution()
    print(f"\n=== RESUMEN ===")
    print(f"Dropout rates recomendados para el modelo: {recommended_dropout}")
    print(f"Estos valores deben reemplazar [0.2, 0.3, 0.4] en el modelo")
