#!/bin/bash

# Script para ejecutar la comparación de modelos CNN con y sin data augmentation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/cnn_with_without_augmentation.py"

echo "=========================================="
echo "Ejecutando comparación de modelos CNN"
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
    echo "Ejecución completada exitosamente"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: La ejecución falló"
    echo "=========================================="
    exit 1
fi
