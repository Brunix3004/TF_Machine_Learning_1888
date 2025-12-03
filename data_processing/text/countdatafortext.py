#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para contar repeticiones de valores de validación en un CSV
"""

import csv
import sys
from collections import Counter

def count_validation_values(file_path, column_name='diagnosis'):
    """
    Cuenta las repeticiones de valores en una columna del CSV
    """
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        
        if column_name not in columns:
            print("Error: La columna '{}' no existe en el dataset".format(column_name))
            print("Columnas disponibles: {}".format(', '.join(columns)))
            return
        
        values = [row[column_name] for row in reader]
        counts = Counter(values)
        total = len(values)
    
    print("Conteo de valores en la columna '{}':".format(column_name))
    print("=" * 50)
    for value, count in counts.most_common():
        percentage = (count / total) * 100
        print("  {}: {} ({:.2f}%)".format(value, count, percentage))
    
    print("\nTotal de registros: {}".format(total))
    
    return counts

if __name__ == "__main__":
    default_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/raw/text/Cancer_Data.csv'
    file_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    column_name = sys.argv[2] if len(sys.argv) > 2 else 'diagnosis'
    
    count_validation_values(file_path, column_name)
