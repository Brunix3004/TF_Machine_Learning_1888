# Análisis del Dataset Cancer_Data.csv

Este directorio contiene scripts para analizar el dataset de cáncer de mama y preparar un modelo SVM.

## 📁 Archivos Incluidos

### Scripts de Análisis
- **`explore_cancer_data.py`** - Exploración inicial del dataset
- **`pca_analysis.py`** - Análisis PCA detallado
- **`column_analysis.py`** - Selección de características
- **`svm_model.py`** - Modelo SVM (preparado, no ejecutado)
- **`run_analysis.py`** - Script principal que ejecuta todo

## 🚀 Cómo Usar

### Opción 1: Ejecutar Todo el Análisis
```bash
cd /Users/benjidry/Documents/Github/TF_Machine_Learning_1888/data_processing/text/
python run_analysis.py
```

### Opción 2: Ejecutar Scripts Individuales
```bash
# 1. Explorar el dataset
python explore_cancer_data.py

# 2. Analizar columnas y seleccionar características
python column_analysis.py

# 3. Realizar análisis PCA
python pca_analysis.py

# 4. Entrenar modelo SVM (cuando esté listo)
python svm_model.py
```

## 📊 Qué Hace Cada Script

### 1. `explore_cancer_data.py`
- Carga y explora el dataset Cancer_Data.csv
- Muestra información general (forma, tipos de datos, valores faltantes)
- Analiza la distribución de la variable objetivo
- Calcula correlaciones con la variable objetivo
- Realiza análisis PCA básico
- Crea visualizaciones del análisis PCA

### 2. `column_analysis.py`
- Analiza correlaciones entre características y variable objetivo
- Calcula importancia de características usando Random Forest
- Realiza pruebas estadísticas (F-test, Mutual Information)
- Crea ranking combinado de características
- Recomienda las mejores características por categoría
- Genera visualizaciones del análisis de características

### 3. `pca_analysis.py`
- Realiza análisis PCA completo del dataset
- Analiza varianza explicada por cada componente
- Encuentra el número óptimo de componentes
- Crea visualizaciones detalladas del PCA
- Analiza contribuciones de características originales
- Recomienda componentes para el modelo SVM

### 4. `svm_model.py`
- Prepara y entrena modelo SVM
- Optimiza hiperparámetros
- Evalúa el modelo con múltiples métricas
- Crea matriz de confusión y curva ROC
- Realiza validación cruzada
- Guarda resultados del modelo

## 📈 Archivos Generados

### Visualizaciones
- `visualization/plots/cancer_pca_analysis.png` - Análisis PCA básico
- `visualization/plots/cancer_pca_detailed_analysis.png` - Análisis PCA detallado
- `visualization/plots/feature_analysis.png` - Análisis de características
- `visualization/plots/svm_confusion_matrix.png` - Matriz de confusión SVM
- `visualization/plots/svm_roc_curve.png` - Curva ROC SVM

### Resultados
- `training/results/feature_recommendations.csv` - Características recomendadas
- `training/results/svm_results.csv` - Resultados del modelo SVM
- `visualization/plots/pca_results.csv` - Resultados del PCA

## 🔍 Dataset Cancer_Data.csv

### Características del Dataset
- **Muestras**: 569 casos
- **Características**: 30 características numéricas
- **Variable objetivo**: diagnosis (M=Maligno, B=Benigno)
- **Sin valores faltantes**

### Categorías de Características
- **`_mean`**: Valores promedio (10 características)
- **`_se`**: Error estándar (10 características)  
- **`_worst`**: Valores más altos (10 características)

### Características Principales
- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`
- `smoothness_mean`, `compactness_mean`, `concavity_mean`
- `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`

## 🎯 Recomendaciones de Uso

### Para Análisis Exploratorio
1. Ejecutar `explore_cancer_data.py` primero
2. Revisar las visualizaciones generadas
3. Entender la distribución de los datos

### Para Selección de Características
1. Ejecutar `column_analysis.py`
2. Revisar las recomendaciones de características
3. Usar las características top para el modelo

### Para Análisis PCA
1. Ejecutar `pca_analysis.py`
2. Determinar el número óptimo de componentes
3. Usar PCA para reducir dimensionalidad

### Para Modelo SVM
1. Ejecutar `svm_model.py` cuando esté listo
2. Revisar métricas de rendimiento
3. Analizar matriz de confusión y curva ROC

## 📋 Próximos Pasos

1. **Ejecutar análisis exploratorio** para entender los datos
2. **Seleccionar características** basado en el análisis
3. **Aplicar PCA** para reducir dimensionalidad
4. **Entrenar modelo SVM** con las características seleccionadas
5. **Evaluar rendimiento** y ajustar hiperparámetros

## ⚠️ Notas Importantes

- Los scripts están preparados pero **NO se ejecutarán automáticamente**
- Primero ejecuta el análisis exploratorio para entender los datos
- Revisa las recomendaciones de características antes de entrenar
- El modelo SVM se puede entrenar cuando estés listo

## 🛠️ Dependencias Requeridas

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## 📞 Soporte

Si encuentras problemas con los scripts, revisa:
1. Que el archivo `Cancer_Data.csv` esté en la ubicación correcta
2. Que todas las dependencias estén instaladas
3. Que los directorios de salida existan
