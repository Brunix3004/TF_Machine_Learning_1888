#!/bin/bash
# Script para ejecutar svm_training.py con el Python correcto del entorno conda

# Activar entorno ml
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate ml

# Usar el Python del entorno conda directamente
/opt/homebrew/Caskroom/miniconda/base/envs/ml/bin/python3 svm_training.py
