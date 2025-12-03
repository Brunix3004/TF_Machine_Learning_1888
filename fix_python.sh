#!/bin/bash
# Script para eliminar Python 2.7 y hacer Python 3.10 el predeterminado
# Ejecutar con: sudo ./fix_python.sh

PYTHON3_PATH="/opt/homebrew/bin/python3.10"

echo "=========================================="
echo "ELIMINANDO PYTHON 2.7 Y CONFIGURANDO PYTHON 3.10"
echo "=========================================="
echo ""

# Verificar Python 3.10
if [ ! -f "$PYTHON3_PATH" ]; then
    echo "Error: Python 3.10 no encontrado en $PYTHON3_PATH"
    exit 1
fi

echo "✓ Python 3.10 encontrado: $($PYTHON3_PATH --version)"
echo ""

# Eliminar enlaces simbólicos de Python 2.7
echo "Eliminando enlaces de Python 2.7 en /usr/local/bin..."
sudo rm -f /usr/local/bin/python
sudo rm -f /usr/local/bin/python2
sudo rm -f /usr/local/bin/python2.7
sudo rm -f /usr/local/bin/python-config 2>/dev/null
sudo rm -f /usr/local/bin/python2-config 2>/dev/null
sudo rm -f /usr/local/bin/python2.7-config 2>/dev/null
sudo rm -f /usr/local/bin/pythonw 2>/dev/null
sudo rm -f /usr/local/bin/python-32 2>/dev/null
sudo rm -f /usr/local/bin/python2-32 2>/dev/null
sudo rm -f /usr/local/bin/python2.7-32 2>/dev/null

echo "✓ Enlaces de Python 2.7 eliminados"
echo ""

# Crear nuevos enlaces a Python 3.10
echo "Creando enlaces a Python 3.10..."
sudo ln -sf "$PYTHON3_PATH" /usr/local/bin/python
sudo ln -sf "$PYTHON3_PATH" /usr/local/bin/python3

echo "✓ Enlaces creados"
echo ""

# Verificar
echo "Verificando cambios:"
echo "  python -> $(which python) ($(python --version 2>&1))"
echo "  python3 -> $(which python3) ($(python3 --version 2>&1))"
echo ""

echo "=========================================="
echo "¡COMPLETADO!"
echo "=========================================="
echo ""
echo "Cierra y abre una nueva terminal para que los cambios surtan efecto."
echo "O ejecuta: source ~/.zshrc"
