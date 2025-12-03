#!/usr/bin/env python3
"""
Script principal para ejecutar todo el análisis del dataset Cancer_Data.csv
Incluye: exploración, PCA, selección de características y preparación para SVM
"""

import sys
import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def run_script(script_path, script_name):
    """
    Ejecuta un script de Python y maneja errores
    """
    print(f"\n{'='*60}")
    print(f"EJECUTANDO: {script_name}")
    print(f"{'='*60}")
    
    try:
        # Ejecutar script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {script_name} ejecutado exitosamente")
            if result.stdout:
                print("📄 Salida del script:")
                print(result.stdout)
        else:
            print(f"❌ Error en {script_name}:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando {script_name}: {str(e)}")
        return False
    
    return True

def create_directories():
    """
    Crea los directorios necesarios si no existen
    """
    directories = [
        '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots',
        '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/training/results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Directorio creado/verificado: {directory}")

def main():
    """
    Función principal que ejecuta todo el análisis
    """
    print("🚀 INICIANDO ANÁLISIS COMPLETO DEL DATASET CANCER_DATA.CSV")
    print("="*80)
    
    # Crear directorios necesarios
    create_directories()
    
    # Definir scripts a ejecutar en orden
    scripts = [
        {
            'path': '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/data_processing/text/explore_cancer_data.py',
            'name': 'Exploración del Dataset'
        },
        {
            'path': '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/data_processing/text/column_analysis.py',
            'name': 'Análisis de Columnas y Selección de Características'
        },
        {
            'path': '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/data_processing/text/pca_analysis.py',
            'name': 'Análisis PCA'
        }
    ]
    
    # Ejecutar scripts en orden
    success_count = 0
    for script in scripts:
        if run_script(script['path'], script['name']):
            success_count += 1
        else:
            print(f"⚠️ Continuando con el siguiente script...")
    
    # Resumen final
    print(f"\n{'='*80}")
    print(f"RESUMEN DEL ANÁLISIS")
    print(f"{'='*80}")
    
    print(f"📊 Scripts ejecutados exitosamente: {success_count}/{len(scripts)}")
    
    if success_count == len(scripts):
        print(f"\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"   • Dataset explorado y analizado")
        print(f"   • PCA realizado")
        print(f"   • Características seleccionadas")
        print(f"   • Archivos generados en visualization/plots/ y training/results/")
        
        print(f"\n📋 PRÓXIMOS PASOS:")
        print(f"   1. Revisar los gráficos generados en visualization/plots/")
        print(f"   2. Revisar las recomendaciones de características en training/results/")
        print(f"   3. Ejecutar el script SVM cuando esté listo: svm_model.py")
        
    else:
        print(f"\n⚠️ ANÁLISIS COMPLETADO CON ERRORES")
        print(f"   • {success_count} scripts ejecutados exitosamente")
        print(f"   • {len(scripts) - success_count} scripts con errores")
        print(f"   • Revisar los errores mostrados arriba")
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"   • visualization/plots/cancer_pca_analysis.png")
    print(f"   • visualization/plots/cancer_pca_detailed_analysis.png")
    print(f"   • visualization/plots/feature_analysis.png")
    print(f"   • training/results/feature_recommendations.csv")
    print(f"   • visualization/plots/pca_results.csv")

if __name__ == "__main__":
    main()
