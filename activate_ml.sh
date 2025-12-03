#!/bin/zsh
# Script para activar entorno ml con Python 3.10 correcto
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate ml
export PATH="/opt/homebrew/Caskroom/miniconda/base/envs/ml/bin:$PATH"
alias python="/opt/homebrew/Caskroom/miniconda/base/envs/ml/bin/python"
alias python3="/opt/homebrew/Caskroom/miniconda/base/envs/ml/bin/python3"
echo "Entorno ml activado con Python 3.10"
python --version
