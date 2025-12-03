#!/usr/bin/env python3
"""
Script para entrenar 2 modelos CNN de Keras:
1. Modelo sin data augmentation
2. Modelo con data augmentation

Compara el rendimiento de ambos modelos en el dataset BreaKHis 400X
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets" / "breast_cancer" / "CNN_data" / "BreaKHis 400X"
RESULTS_DIR = BASE_DIR / "training" / "experiments" / "results"
MODELS_DIR = BASE_DIR / "models" / "saved"

# Crear directorios si no existen
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros del modelo
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42
VALIDATION_SPLIT = 0.2

print("=" * 60)
print("ENTRENAMIENTO DE MODELOS CNN CON Y SIN DATA AUGMENTATION")
print("=" * 60)
print(f"\nDirectorio de datos: {DATA_DIR}")
print(f"Directorio de resultados: {RESULTS_DIR}")
print(f"Directorio de modelos: {MODELS_DIR}\n")

# Verificar que existe el directorio de datos
if not DATA_DIR.exists():
    print(f"ERROR: No se encuentra el directorio {DATA_DIR}")
    sys.exit(1)

# ============================================================
# 1. CARGAR Y PREPARAR DATOS
# ============================================================

print("Cargando datasets...")

# Dataset de entrenamiento (sin augmentation inicialmente)
train_ds_no_aug = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=SEED,
    validation_split=VALIDATION_SPLIT,
    subset="training",
)

# Dataset de validación (compartido para ambos modelos)
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=SEED,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
)

# Dataset de test
test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=SEED,
)

class_names = train_ds_no_aug.class_names
print(f"Clases detectadas: {class_names}")
print(f"Imágenes de entrenamiento: {len(train_ds_no_aug) * BATCH_SIZE}")
print(f"Imágenes de validación: {len(val_ds) * BATCH_SIZE}")
print(f"Imágenes de test: {len(test_ds) * BATCH_SIZE}\n")

# Optimizar pipeline
AUTOTUNE = tf.data.AUTOTUNE

train_ds_no_aug = train_ds_no_aug.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Dataset con augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.1, seed=SEED),
    layers.RandomZoom(0.1, seed=SEED),
    layers.RandomBrightness(0.1, seed=SEED),
    layers.RandomContrast(0.1, seed=SEED),
])

train_ds_with_aug = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=SEED,
    validation_split=VALIDATION_SPLIT,
    subset="training",
)

train_ds_with_aug = train_ds_with_aug.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# ============================================================
# 2. DEFINIR ARQUITECTURA DEL MODELO
# ============================================================

def create_cnn_model(name="cnn_model"):
    """Crea un modelo CNN para clasificación binaria"""
    model = keras.Sequential([
        # Normalización
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # Bloque convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        
        # Bloque convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        
        # Bloque convolucional 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        
        # Clasificador
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name=name)
    
    return model

# ============================================================
# 3. ENTRENAR MODELO SIN AUGMENTATION
# ============================================================

print("\n" + "=" * 60)
print("ENTRENANDO MODELO SIN DATA AUGMENTATION")
print("=" * 60)

model_no_aug = create_cnn_model("CNN_NoAugmentation")
model_no_aug.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\nArquitectura del modelo:")
model_no_aug.summary()

# Callbacks
callbacks_no_aug = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODELS_DIR / "cnn_no_aug_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Entrenar
history_no_aug = model_no_aug.fit(
    train_ds_no_aug,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_no_aug,
    verbose=1
)

# Evaluar en test
print("\nEvaluando modelo sin augmentation en test...")
test_results_no_aug = model_no_aug.evaluate(test_ds, verbose=1)
print(f"Test Loss: {test_results_no_aug[0]:.4f}")
print(f"Test Accuracy: {test_results_no_aug[1]:.4f}")
print(f"Test Precision: {test_results_no_aug[2]:.4f}")
print(f"Test Recall: {test_results_no_aug[3]:.4f}")

# Calcular matriz de confusión y falsos negativos
print("\nCalculando matriz de confusión y falsos negativos...")
y_true_no_aug = []
y_pred_no_aug = []
for images, labels in test_ds:
    y_true_no_aug.extend(labels.numpy())
    pred = model_no_aug.predict(images, verbose=0)
    y_pred_no_aug.extend((pred >= 0.5).astype(int).flatten())

y_true_no_aug = np.array(y_true_no_aug)
y_pred_no_aug = np.array(y_pred_no_aug)
cm_no_aug = confusion_matrix(y_true_no_aug, y_pred_no_aug)
TN_no_aug, FP_no_aug, FN_no_aug, TP_no_aug = cm_no_aug.ravel()
total_malignos_no_aug = TP_no_aug + FN_no_aug
FNR_no_aug = FN_no_aug / total_malignos_no_aug if total_malignos_no_aug > 0 else 0

print(f"\nMatriz de Confusión (Sin Augmentation):")
print(f"  TN: {TN_no_aug}, FP: {FP_no_aug}, FN: {FN_no_aug}, TP: {TP_no_aug}")
print(f"  FNR (Probabilidad de Falso Negativo): {FNR_no_aug:.4f} ({FNR_no_aug*100:.2f}%)")
print(f"  Recall (Sensibilidad): {test_results_no_aug[3]:.4f} ({test_results_no_aug[3]*100:.2f}%)")

# Guardar modelo final
model_no_aug.save(str(MODELS_DIR / "cnn_no_augmentation_final.keras"))
print(f"\nModelo guardado en: {MODELS_DIR / 'cnn_no_augmentation_final.keras'}")

# ============================================================
# 4. ENTRENAR MODELO CON AUGMENTATION
# ============================================================

print("\n" + "=" * 60)
print("ENTRENANDO MODELO CON DATA AUGMENTATION")
print("=" * 60)

model_with_aug = create_cnn_model("CNN_WithAugmentation")
model_with_aug.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\nArquitectura del modelo:")
model_with_aug.summary()

# Callbacks
callbacks_with_aug = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODELS_DIR / "cnn_with_aug_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Entrenar
history_with_aug = model_with_aug.fit(
    train_ds_with_aug,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_with_aug,
    verbose=1
)

# Evaluar en test
print("\nEvaluando modelo con augmentation en test...")
test_results_with_aug = model_with_aug.evaluate(test_ds, verbose=1)
print(f"Test Loss: {test_results_with_aug[0]:.4f}")
print(f"Test Accuracy: {test_results_with_aug[1]:.4f}")
print(f"Test Precision: {test_results_with_aug[2]:.4f}")
print(f"Test Recall: {test_results_with_aug[3]:.4f}")

# Calcular matriz de confusión y falsos negativos
print("\nCalculando matriz de confusión y falsos negativos...")
y_true_with_aug = []
y_pred_with_aug = []
for images, labels in test_ds:
    y_true_with_aug.extend(labels.numpy())
    pred = model_with_aug.predict(images, verbose=0)
    y_pred_with_aug.extend((pred >= 0.5).astype(int).flatten())

y_true_with_aug = np.array(y_true_with_aug)
y_pred_with_aug = np.array(y_pred_with_aug)
cm_with_aug = confusion_matrix(y_true_with_aug, y_pred_with_aug)
TN_with_aug, FP_with_aug, FN_with_aug, TP_with_aug = cm_with_aug.ravel()
total_malignos_with_aug = TP_with_aug + FN_with_aug
FNR_with_aug = FN_with_aug / total_malignos_with_aug if total_malignos_with_aug > 0 else 0

print(f"\nMatriz de Confusión (Con Augmentation):")
print(f"  TN: {TN_with_aug}, FP: {FP_with_aug}, FN: {FN_with_aug}, TP: {TP_with_aug}")
print(f"  FNR (Probabilidad de Falso Negativo): {FNR_with_aug:.4f} ({FNR_with_aug*100:.2f}%)")
print(f"  Recall (Sensibilidad): {test_results_with_aug[3]:.4f} ({test_results_with_aug[3]*100:.2f}%)")

# Guardar modelo final
model_with_aug.save(str(MODELS_DIR / "cnn_with_augmentation_final.keras"))
print(f"\nModelo guardado en: {MODELS_DIR / 'cnn_with_augmentation_final.keras'}")

# ============================================================
# 5. COMPARAR RESULTADOS Y GUARDAR MÉTRICAS
# ============================================================

print("\n" + "=" * 60)
print("COMPARACIÓN DE RESULTADOS")
print("=" * 60)

comparison = {
    "timestamp": datetime.now().isoformat(),
    "dataset": "BreaKHis 400X",
    "image_size": f"{IMG_HEIGHT}x{IMG_WIDTH}",
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "models": {
        "no_augmentation": {
            "test_loss": float(test_results_no_aug[0]),
            "test_accuracy": float(test_results_no_aug[1]),
            "test_precision": float(test_results_no_aug[2]),
            "test_recall": float(test_results_no_aug[3]),
            "final_train_accuracy": float(history_no_aug.history['accuracy'][-1]),
            "final_val_accuracy": float(history_no_aug.history['val_accuracy'][-1]),
            "best_val_accuracy": float(max(history_no_aug.history['val_accuracy'])),
            "confusion_matrix": {
                "TN": int(TN_no_aug),
                "FP": int(FP_no_aug),
                "FN": int(FN_no_aug),
                "TP": int(TP_no_aug)
            },
            "FNR": float(FNR_no_aug),
            "model_path": str(MODELS_DIR / "cnn_no_augmentation_final.keras")
        },
        "with_augmentation": {
            "test_loss": float(test_results_with_aug[0]),
            "test_accuracy": float(test_results_with_aug[1]),
            "test_precision": float(test_results_with_aug[2]),
            "test_recall": float(test_results_with_aug[3]),
            "final_train_accuracy": float(history_with_aug.history['accuracy'][-1]),
            "final_val_accuracy": float(history_with_aug.history['val_accuracy'][-1]),
            "best_val_accuracy": float(max(history_with_aug.history['val_accuracy'])),
            "confusion_matrix": {
                "TN": int(TN_with_aug),
                "FP": int(FP_with_aug),
                "FN": int(FN_with_aug),
                "TP": int(TP_with_aug)
            },
            "FNR": float(FNR_with_aug),
            "model_path": str(MODELS_DIR / "cnn_with_augmentation_final.keras")
        }
    }
}

# Calcular mejoras
improvement = {
    "accuracy_improvement": comparison["models"]["with_augmentation"]["test_accuracy"] - comparison["models"]["no_augmentation"]["test_accuracy"],
    "precision_improvement": comparison["models"]["with_augmentation"]["test_precision"] - comparison["models"]["no_augmentation"]["test_precision"],
    "recall_improvement": comparison["models"]["with_augmentation"]["test_recall"] - comparison["models"]["no_augmentation"]["test_recall"],
    "loss_improvement": comparison["models"]["no_augmentation"]["test_loss"] - comparison["models"]["with_augmentation"]["test_loss"]
}
comparison["improvement"] = improvement

# Guardar comparación
comparison_file = RESULTS_DIR / "cnn_augmentation_comparison.json"
with open(comparison_file, 'w') as f:
    json.dump(comparison, f, indent=2)

print("\nResultados guardados en:", comparison_file)
print("\n" + "-" * 60)
print("RESUMEN DE RESULTADOS")
print("-" * 60)
print(f"\n{'Métrica':<25} {'Sin Aug':<15} {'Con Aug':<15} {'Mejora':<15}")
print("-" * 70)
print(f"{'Test Accuracy':<25} {comparison['models']['no_augmentation']['test_accuracy']:<15.4f} {comparison['models']['with_augmentation']['test_accuracy']:<15.4f} {improvement['accuracy_improvement']:+.4f}")
print(f"{'Test Precision':<25} {comparison['models']['no_augmentation']['test_precision']:<15.4f} {comparison['models']['with_augmentation']['test_precision']:<15.4f} {improvement['precision_improvement']:+.4f}")
print(f"{'Test Recall':<25} {comparison['models']['no_augmentation']['test_recall']:<15.4f} {comparison['models']['with_augmentation']['test_recall']:<15.4f} {improvement['recall_improvement']:+.4f}")
print(f"{'FNR (Falsos Negativos)':<25} {comparison['models']['no_augmentation']['FNR']:<15.4f} {comparison['models']['with_augmentation']['FNR']:<15.4f} {comparison['models']['no_augmentation']['FNR'] - comparison['models']['with_augmentation']['FNR']:+.4f}")
print(f"{'Test Loss':<25} {comparison['models']['no_augmentation']['test_loss']:<15.4f} {comparison['models']['with_augmentation']['test_loss']:<15.4f} {improvement['loss_improvement']:+.4f}")
print("-" * 70)
print(f"\n⚠️  PROBABILIDAD DE FALSOS NEGATIVOS:")
print(f"   Sin Augmentation: {comparison['models']['no_augmentation']['FNR']*100:.2f}%")
print(f"   Con Augmentation: {comparison['models']['with_augmentation']['FNR']*100:.2f}%")
print(f"   Mejora: {(comparison['models']['no_augmentation']['FNR'] - comparison['models']['with_augmentation']['FNR'])*100:+.2f}%")
print()

# ============================================================
# 6. VISUALIZAR RESULTADOS
# ============================================================

print("\nGenerando gráficas de comparación...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy
axes[0, 0].plot(history_no_aug.history['accuracy'], label='Train (Sin Aug)', linestyle='--')
axes[0, 0].plot(history_no_aug.history['val_accuracy'], label='Val (Sin Aug)', linestyle='-')
axes[0, 0].plot(history_with_aug.history['accuracy'], label='Train (Con Aug)', linestyle='--')
axes[0, 0].plot(history_with_aug.history['val_accuracy'], label='Val (Con Aug)', linestyle='-')
axes[0, 0].set_title('Accuracy durante el entrenamiento')
axes[0, 0].set_xlabel('Época')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Loss
axes[0, 1].plot(history_no_aug.history['loss'], label='Train (Sin Aug)', linestyle='--')
axes[0, 1].plot(history_no_aug.history['val_loss'], label='Val (Sin Aug)', linestyle='-')
axes[0, 1].plot(history_with_aug.history['loss'], label='Train (Con Aug)', linestyle='--')
axes[0, 1].plot(history_with_aug.history['val_loss'], label='Val (Con Aug)', linestyle='-')
axes[0, 1].set_title('Loss durante el entrenamiento')
axes[0, 1].set_xlabel('Época')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Comparación de métricas en test
metrics = ['Accuracy', 'Precision', 'Recall']
no_aug_values = [
    comparison['models']['no_augmentation']['test_accuracy'],
    comparison['models']['no_augmentation']['test_precision'],
    comparison['models']['no_augmentation']['test_recall']
]
with_aug_values = [
    comparison['models']['with_augmentation']['test_accuracy'],
    comparison['models']['with_augmentation']['test_precision'],
    comparison['models']['with_augmentation']['test_recall']
]

x = np.arange(len(metrics))
width = 0.35
axes[1, 0].bar(x - width/2, no_aug_values, width, label='Sin Augmentation', alpha=0.8)
axes[1, 0].bar(x + width/2, with_aug_values, width, label='Con Augmentation', alpha=0.8)
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Comparación de Métricas en Test')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics)
axes[1, 0].legend()
axes[1, 0].grid(True, axis='y')
axes[1, 0].set_ylim([0, 1])

# Mejoras
improvements = [
    improvement['accuracy_improvement'],
    improvement['precision_improvement'],
    improvement['recall_improvement']
]
colors = ['green' if x > 0 else 'red' for x in improvements]
axes[1, 1].bar(metrics, improvements, color=colors, alpha=0.7)
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 1].set_ylabel('Mejora')
axes[1, 1].set_title('Mejora con Data Augmentation')
axes[1, 1].grid(True, axis='y')

plt.tight_layout()
plot_file = RESULTS_DIR / "cnn_augmentation_comparison.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Gráfica guardada en: {plot_file}")
plt.close()

print("\n" + "=" * 60)
print("ENTRENAMIENTO COMPLETADO")
print("=" * 60)
print(f"\nModelos guardados en: {MODELS_DIR}")
print(f"Resultados guardados en: {RESULTS_DIR}")
print("\n¡Proceso finalizado exitosamente!")
