#!/bin/bash

# Script para crear la estructura de carpetas del proyecto de Machine Learning
# Proyecto: TF_Machine_Learning_1888

echo "🚀 Creando estructura del proyecto de Machine Learning..."

# Crear directorio raíz del proyecto
PROJECT_ROOT="/Users/benjidry/Documents/Github/TF_Machine_Learning_1888"

# Navegar al directorio del proyecto
cd "$PROJECT_ROOT"

# 1. DATASETS - Almacenamiento de datos
echo "📁 Creando estructura de datasets..."
mkdir -p datasets/raw/text
mkdir -p datasets/raw/images
mkdir -p datasets/processed/text
mkdir -p datasets/processed/images
mkdir -p datasets/external
mkdir -p datasets/generated

# 2. DATA PROCESSING - Procesamiento de datos
echo "⚙️ Creando estructura de procesamiento..."
mkdir -p data_processing/text
mkdir -p data_processing/images
mkdir -p data_processing/utils
mkdir -p data_processing/feature_engineering

# 3. MODELS - Modelos de machine learning
echo "🤖 Creando estructura de modelos..."
mkdir -p models/text
mkdir -p models/vision
mkdir -p models/saved
mkdir -p models/checkpoints
mkdir -p models/logs

# 4. VISUALIZATION - Visualización de datos
echo "📊 Creando estructura de visualización..."
mkdir -p visualization/plots
mkdir -p visualization/dashboards
mkdir -p visualization/reports
mkdir -p visualization/notebooks

# 5. TRAINING - Entrenamiento de modelos
echo "🏋️ Creando estructura de entrenamiento..."
mkdir -p training/scripts
mkdir -p training/configs
mkdir -p training/experiments
mkdir -p training/results

# 6. EVALUATION - Evaluación de modelos
echo "📈 Creando estructura de evaluación..."
mkdir -p evaluation/metrics
mkdir -p evaluation/reports
mkdir -p evaluation/comparisons

# 7. UTILITIES - Utilidades y herramientas
echo "🔧 Creando estructura de utilidades..."
mkdir -p utils/data_utils
mkdir -p utils/model_utils
mkdir -p utils/visualization_utils
mkdir -p utils/config

# 8. NOTEBOOKS - Jupyter notebooks
echo "📓 Creando estructura de notebooks..."
mkdir -p notebooks/exploration
mkdir -p notebooks/experiments
mkdir -p notebooks/tutorials

# 9. TESTS - Pruebas unitarias
echo "🧪 Creando estructura de tests..."
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/fixtures

# 10. DOCUMENTATION - Documentación
echo "📚 Creando estructura de documentación..."
mkdir -p docs/api
mkdir -p docs/tutorials
mkdir -p docs/examples

# 11. CONFIGURATION - Archivos de configuración
echo "⚙️ Creando estructura de configuración..."
mkdir -p config/environments
mkdir -p config/models
mkdir -p config/datasets

# 12. SCRIPTS - Scripts de automatización
echo "📜 Creando estructura de scripts..."
mkdir -p scripts/setup
mkdir -p scripts/deployment
mkdir -p scripts/maintenance

# Crear archivos README en cada directorio principal
echo "📝 Creando archivos README..."

cat > datasets/README.md << 'EOF'
# Datasets

Este directorio contiene todos los datasets del proyecto.

## Estructura:
- `raw/` - Datos sin procesar
- `processed/` - Datos procesados y listos para usar
- `external/` - Datasets externos
- `generated/` - Datasets generados sintéticamente
EOF

cat > data_processing/README.md << 'EOF'
# Data Processing

Scripts y utilidades para el procesamiento de datos.

## Estructura:
- `text/` - Procesamiento de texto
- `images/` - Procesamiento de imágenes
- `utils/` - Utilidades generales
- `feature_engineering/` - Ingeniería de características
EOF

cat > models/README.md << 'EOF'
# Models

Modelos de machine learning entrenados y configuraciones.

## Estructura:
- `text/` - Modelos de texto (Naive Bayes, etc.)
- `vision/` - Modelos de visión
- `saved/` - Modelos guardados
- `checkpoints/` - Puntos de control durante entrenamiento
- `logs/` - Logs de entrenamiento
EOF

cat > visualization/README.md << 'EOF'
# Visualization

Visualizaciones y reportes de datos.

## Estructura:
- `plots/` - Gráficos estáticos
- `dashboards/` - Dashboards interactivos
- `reports/` - Reportes generados
- `notebooks/` - Notebooks de visualización
EOF

cat > training/README.md << 'EOF'
# Training

Scripts y configuraciones para entrenar modelos.

## Estructura:
- `scripts/` - Scripts de entrenamiento
- `configs/` - Configuraciones de modelos
- `experiments/` - Experimentos
- `results/` - Resultados de entrenamiento
EOF

# Crear archivo .gitkeep en directorios vacíos para mantener la estructura en git
echo "🔧 Creando archivos .gitkeep..."
find . -type d -empty -exec touch {}/.gitkeep \;

# Crear archivo de configuración de gitignore
echo "📝 Creando .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (opcional - descomenta si no quieres subir datasets grandes)
# datasets/raw/
# datasets/processed/
# *.csv
# *.json
# *.pkl

# Model files (opcional - descomenta si no quieres subir modelos grandes)
# models/saved/
# models/checkpoints/
# *.h5
# *.pkl
# *.joblib

# Logs
logs/
*.log

# Temporary files
tmp/
temp/
EOF

echo "✅ ¡Estructura del proyecto creada exitosamente!"
echo ""
echo "📁 Estructura creada:"
echo "├── datasets/ (raw, processed, external, generated)"
echo "├── data_processing/ (text, images, utils, feature_engineering)"
echo "├── models/ (text, vision, saved, checkpoints, logs)"
echo "├── visualization/ (plots, dashboards, reports, notebooks)"
echo "├── training/ (scripts, configs, experiments, results)"
echo "├── evaluation/ (metrics, reports, comparisons)"
echo "├── utils/ (data_utils, model_utils, visualization_utils, config)"
echo "├── notebooks/ (exploration, experiments, tutorials)"
echo "├── tests/ (unit, integration, fixtures)"
echo "├── docs/ (api, tutorials, examples)"
echo "├── config/ (environments, models, datasets)"
echo "└── scripts/ (setup, deployment, maintenance)"
echo ""
echo "🎉 ¡Proyecto listo para comenzar con Machine Learning!"
