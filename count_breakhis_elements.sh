#!/bin/bash

# Script para contar elementos en las carpetas benign y malignant
# de train y test del dataset BreaKHis 400X

BASE_DIR="/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/breast_cancer/CNN_data/BreaKHis 400X"

echo "=========================================="
echo "Conteo de elementos - BreaKHis 400X"
echo "=========================================="
echo ""

# Función para contar archivos en una carpeta
count_files() {
    local dir="$1"
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -type f | wc -l | tr -d ' ')
        echo "$count"
    else
        echo "0"
    fi
}

# Contar elementos en train
echo "--- TRAIN ---"
TRAIN_BENIGN_DIR="$BASE_DIR/train/benign"
TRAIN_MALIGNANT_DIR="$BASE_DIR/train/malignant"

TRAIN_BENIGN_COUNT=$(count_files "$TRAIN_BENIGN_DIR")
TRAIN_MALIGNANT_COUNT=$(count_files "$TRAIN_MALIGNANT_DIR")
TRAIN_TOTAL=$((TRAIN_BENIGN_COUNT + TRAIN_MALIGNANT_COUNT))

echo "  Benign:    $TRAIN_BENIGN_COUNT"
echo "  Malignant: $TRAIN_MALIGNANT_COUNT"
echo "  Total:     $TRAIN_TOTAL"
echo ""

# Contar elementos en test
echo "--- TEST ---"
TEST_BENIGN_DIR="$BASE_DIR/test/benign"
TEST_MALIGNANT_DIR="$BASE_DIR/test/malignant"

TEST_BENIGN_COUNT=$(count_files "$TEST_BENIGN_DIR")
TEST_MALIGNANT_COUNT=$(count_files "$TEST_MALIGNANT_DIR")
TEST_TOTAL=$((TEST_BENIGN_COUNT + TEST_MALIGNANT_COUNT))

echo "  Benign:    $TEST_BENIGN_COUNT"
echo "  Malignant: $TEST_MALIGNANT_COUNT"
echo "  Total:     $TEST_TOTAL"
echo ""

# Resumen total
echo "--- RESUMEN TOTAL ---"
TOTAL_BENIGN=$((TRAIN_BENIGN_COUNT + TEST_BENIGN_COUNT))
TOTAL_MALIGNANT=$((TRAIN_MALIGNANT_COUNT + TEST_MALIGNANT_COUNT))
GRAND_TOTAL=$((TRAIN_TOTAL + TEST_TOTAL))

echo "  Total Benign:    $TOTAL_BENIGN"
echo "  Total Malignant: $TOTAL_MALIGNANT"
echo "  Total General:   $GRAND_TOTAL"
echo "=========================================="
