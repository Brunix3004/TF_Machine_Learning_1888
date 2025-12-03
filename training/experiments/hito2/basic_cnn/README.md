# CNN Básica - Clasificación de Lesiones Pulmonares

Este directorio contiene la implementación de una CNN básica para la clasificación de lesiones pulmonares utilizando el dataset CLAHE procesado.

## 📁 Estructura del Proyecto

```
basic_cnn/
├── basic_cnn_model.py      # Modelo principal y entrenamiento
├── run_experiment.py       # Script de ejecución
├── config.yaml            # Configuración del experimento
└── README.md              # Este archivo
```

## 🏗️ Arquitectura del Modelo

### Especificaciones Técnicas

- **Tipo**: CNN secuencial feed-forward
- **Bloques convolucionales**: 3
- **Profundidad total**: 8 capas
- **Entrada**: 512×512 píxeles (escala de grises)
- **Salida**: 3 clases (Nodule/Mass, Other lesion, No Findings)

### Arquitectura Detallada

```
Entrada: [batch, 1, 512, 512]
    ↓
Bloque 1: Conv2d(1→64) + BatchNorm + ReLU + MaxPool2d(2×2)
    ↓ [batch, 64, 256, 256]
Bloque 2: Conv2d(64→128) + BatchNorm + ReLU + MaxPool2d(2×2) + Dropout2d(0.2)
    ↓ [batch, 128, 128, 128]
Bloque 3: Conv2d(128→256) + BatchNorm + ReLU + MaxPool2d(2×2) + Dropout2d(0.3)
    ↓ [batch, 256, 64, 64]
Global Average Pooling: AdaptiveAvgPool2d(1×1)
    ↓ [batch, 256, 1, 1]
Clasificador: FC(256→128) + ReLU + Dropout(0.4) + FC(128→3)
    ↓ [batch, 3]
```

## 🔧 Características Técnicas

### Preprocesamiento
- **Redimensionamiento**: 1024×1024 → 512×512
- **Normalización**: mean=0.485, std=0.229
- **CLAHE**: Ya aplicado en el dataset (clip_limit=3.0, tile_grid_size=8×8)

### Data Augmentation (Solo Entrenamiento)
- **Horizontal Flip**: 50% probabilidad
- **Rotación**: ±10 grados, 30% probabilidad

### Regularización
- **BatchNormalization**: Después de cada capa convolucional
- **Dropout2d**: En bloques 2 y 3 (0.2, 0.3)
- **Dropout1d**: En clasificador (0.4)
- **Global Average Pooling**: Reduce parámetros vs Flatten tradicional

### Optimización
- **Optimizador**: AdamW
- **Learning Rate**: 1×10⁻³
- **Weight Decay**: 1×10⁻⁴
- **Función de Pérdida**: CrossEntropyLoss ponderado

## 🚀 Uso

### Ejecución Básica
```bash
python run_experiment.py
```

### Ejecución con Configuración Personalizada
```bash
python run_experiment.py --config mi_config.yaml
```

### Ejecución en GPU Específica
```bash
python run_experiment.py --gpu 0
```

### Validación de Configuración (Dry Run)
```bash
python run_experiment.py --dry-run
```

### Modo Verbose
```bash
python run_experiment.py --verbose
```

## 📊 Parámetros del Modelo

- **Parámetros totales**: ~1.2 millones
- **Parámetros entrenables**: ~1.2 millones
- **Memoria estimada**: ~50MB (modelo)
- **Tiempo de entrenamiento**: ~2-4 horas (dependiendo del hardware)

## 📈 Métricas y Evaluación

### Métricas Calculadas
- Accuracy (Precisión)
- Precision por clase
- Recall por clase
- F1-Score por clase
- Matriz de confusión

### Visualizaciones Generadas
- Curvas de pérdida (entrenamiento/validación)
- Curvas de precisión (entrenamiento/validación)
- Matriz de confusión
- Reporte de clasificación

## 📁 Archivos de Salida

### Modelos Guardados
- `basic_cnn_best.pth`: Mejor modelo según validación
- `basic_cnn_final.pth`: Modelo final después de todas las épocas

### Resultados
- `basic_cnn_training_history.csv`: Historial de entrenamiento
- `basic_cnn_training_history.png`: Gráficos de entrenamiento
- `basic_cnn_confusion_matrix.png`: Matriz de confusión
- `basic_cnn_predictions.csv`: Predicciones del conjunto de validación

### Logs
- `basic_cnn_training.log`: Log detallado del entrenamiento

## ⚙️ Configuración

El archivo `config.yaml` contiene toda la configuración del experimento:

### Secciones Principales
- **model**: Configuración de la arquitectura
- **data**: Paths y preprocesamiento
- **training**: Hiperparámetros y optimización
- **evaluation**: Métricas y evaluación
- **save**: Directorios de guardado
- **visualization**: Configuración de gráficos

### Personalización
Puedes modificar cualquier parámetro en `config.yaml` sin cambiar el código:

```yaml
training:
  batch_size: 64        # Cambiar batch size
  num_epochs: 100       # Cambiar número de épocas
  learning_rate: 0.0005 # Cambiar learning rate

model:
  dropout_rates: [0.3, 0.4, 0.5]  # Cambiar dropout
```

## 🔍 Clases del Dataset

1. **Nodule/Mass** (Clase 0): Nódulos y masas pulmonares
2. **Other lesion** (Clase 1): Otras lesiones pulmonares
3. **No Findings** (Clase 2): Sin hallazgos patológicos

## 📋 Requisitos del Sistema

### Hardware Mínimo
- **RAM**: 8GB
- **GPU**: 4GB VRAM (recomendado)
- **CPU**: 4 cores

### Hardware Recomendado
- **RAM**: 16GB+
- **GPU**: 8GB+ VRAM
- **CPU**: 8+ cores

### Software
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (para GPU)
- Dependencias en `requirements.txt`

## 🐛 Solución de Problemas

### Error de Memoria GPU
```bash
# Reducir batch size en config.yaml
training:
  batch_size: 16  # En lugar de 32
```

### Error de CUDA
```bash
# Forzar uso de CPU
export CUDA_VISIBLE_DEVICES=""
python run_experiment.py
```

### Error de Paths
```bash
# Verificar que los paths en config.yaml sean correctos
python run_experiment.py --dry-run
```

## 📚 Referencias

- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **AdamW**: Adam with Weight Decay
- **Global Average Pooling**: Reducción de parámetros en CNNs
- **Batch Normalization**: Estabilización del entrenamiento

## 📝 Notas de Desarrollo

- El modelo está optimizado para imágenes médicas de 512×512
- La arquitectura balancea precisión y eficiencia computacional
- Los pesos de clases se calculan automáticamente para manejar desbalance
- El entrenamiento incluye early stopping implícito (mejor modelo guardado)
