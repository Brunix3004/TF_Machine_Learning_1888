#!/usr/bin/env python3
"""
Script de prueba para verificar que el dataset se carga correctamente.
"""

import sys
from pathlib import Path
from basic_cnn_model import LungDataset, get_transforms

# Configuración
DATA_PATH = Path("/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/processed/images_clahe")

print("=== PRUEBA DE CARGA DEL DATASET ===")
print(f"Dataset path: {DATA_PATH}")
print(f"Existe: {DATA_PATH.exists()}")

# Verificar estructura
train_images = DATA_PATH / "train" / "images"
train_labels = DATA_PATH / "train" / "labels"
val_images = DATA_PATH / "val" / "images"
val_labels = DATA_PATH / "val" / "labels"

print(f"\nTrain images: {train_images} - Existe: {train_images.exists()}")
print(f"Train labels: {train_labels} - Existe: {train_labels.exists()}")
print(f"Val images: {val_images} - Existe: {val_images.exists()}")
print(f"Val labels: {val_labels} - Existe: {val_labels.exists()}")

# Contar imágenes
if train_images.exists():
    train_img_count = len(list(train_images.glob("*.jpg")))
    print(f"\nImágenes de entrenamiento: {train_img_count}")
    
    # Mostrar algunas imágenes
    train_imgs = list(train_images.glob("*.jpg"))[:5]
    print("Primeras 5 imágenes de entrenamiento:")
    for img in train_imgs:
        print(f"  - {img.name}")

if val_images.exists():
    val_img_count = len(list(val_images.glob("*.jpg")))
    print(f"\nImágenes de validación: {val_img_count}")
    
    # Mostrar algunas imágenes
    val_imgs = list(val_images.glob("*.jpg"))[:5]
    print("Primeras 5 imágenes de validación:")
    for img in val_imgs:
        print(f"  - {img.name}")

# Probar carga del dataset
print("\n=== PROBANDO CARGA DEL DATASET ===")
try:
    train_transform = get_transforms(is_training=True)
    train_dataset = LungDataset(
        train_images,
        train_labels,
        transform=train_transform,
        is_training=True
    )
    print(f"✓ Dataset de entrenamiento cargado: {len(train_dataset)} muestras")
    
    if len(train_dataset) > 0:
        # Probar una muestra
        sample_img, sample_label = train_dataset[0]
        print(f"✓ Primera muestra: imagen shape={sample_img.shape}, label={sample_label}")
    
except Exception as e:
    print(f"❌ Error cargando dataset de entrenamiento: {e}")

try:
    val_transform = get_transforms(is_training=False)
    val_dataset = LungDataset(
        val_images,
        val_labels,
        transform=val_transform,
        is_training=False
    )
    print(f"✓ Dataset de validación cargado: {len(val_dataset)} muestras")
    
    if len(val_dataset) > 0:
        # Probar una muestra
        sample_img, sample_label = val_dataset[0]
        print(f"✓ Primera muestra: imagen shape={sample_img.shape}, label={sample_label}")
    
except Exception as e:
    print(f"❌ Error cargando dataset de validación: {e}")

print("\n=== PRUEBA COMPLETADA ===")
