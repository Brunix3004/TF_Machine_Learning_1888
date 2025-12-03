#!/usr/bin/env python3
"""
Script para verificar los pesos de clases calculados.
"""

from basic_cnn_model import LungDataset, get_transforms, calculate_class_weights
from pathlib import Path

# Configuración
DATA_PATH = Path("/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/processed/images_clahe")

print("=== VERIFICACIÓN DE PESOS DE CLASES ===")

# Cargar dataset
train_transform = get_transforms(is_training=True)
train_dataset = LungDataset(
    DATA_PATH / "train" / "images",
    DATA_PATH / "train" / "labels",
    transform=train_transform,
    is_training=True
)

print(f"Dataset cargado: {len(train_dataset)} muestras")

# Calcular pesos
class_weights = calculate_class_weights(train_dataset)
print(f"Pesos calculados: {class_weights.numpy()}")

# Mostrar distribución de clases
from collections import Counter
class_counts = Counter()
for _, label in train_dataset.samples:
    class_counts[label] += 1

print(f"\nDistribución de clases:")
for class_id in sorted(class_counts.keys()):
    count = class_counts[class_id]
    percentage = (count / len(train_dataset)) * 100
    weight = class_weights[class_id].item()
    print(f"  Clase {class_id}: {count} muestras ({percentage:.1f}%) → peso: {weight:.3f}")

print(f"\nInterpretación:")
print(f"  - Clase con menos muestras tiene peso más alto")
print(f"  - Clase con más muestras tiene peso más bajo")
print(f"  - Esto compensa el desbalance en la función de pérdida")
