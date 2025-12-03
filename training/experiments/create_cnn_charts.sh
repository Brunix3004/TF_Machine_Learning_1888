#!/bin/bash

# Script para generar gráficos de métricas de modelos CNN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/create_cnn_metrics_charts.py"

echo "=========================================="
echo "Generando gráficos de métricas CNN"
echo "=========================================="
echo ""

# Verificar que existe el script
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: No se encuentra el script $PYTHON_SCRIPT"
    exit 1
fi

# Ejecutar el script Python
python3 "$PYTHON_SCRIPT"

# Verificar el código de salida
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Gráficos generados exitosamente"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: La generación de gráficos falló"
    echo "=========================================="
    exit 1
fi
