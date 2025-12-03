#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para analizar métricas del modelo SVM y calcular falsos negativos
"""

import pandas as pd
import numpy as np
import joblib
import os

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'saved', 'svm_breast_cancer.joblib')
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'raw', 'text', 'Cancer_Data.csv')

def load_model_and_data():
    """Carga el modelo y los datos de prueba"""
    print("Cargando modelo y datos...")
    
    # Cargar modelo
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    pca = model_data['pca']
    
    # Cargar datos
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(axis=1)
    
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    y = (y == 'M').astype(int)  # M=1 (maligno), B=0 (benigno)
    
    # Preprocesar
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # Predicciones
    y_pred = model.predict(X_pca)
    y_pred_proba = model.predict_proba(X_pca)[:, 1]
    
    return y, y_pred, y_pred_proba, model

def calculate_confusion_matrix(y_true, y_pred):
    """Calcula la matriz de confusión"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Estructura de la matriz:
    # [[TN, FP],  [Verdaderos Negativos, Falsos Positivos]
    #  [FN, TP]]  [Falsos Negativos, Verdaderos Positivos]
    
    TN, FP, FN, TP = cm.ravel()
    
    return cm, TN, FP, FN, TP

def analyze_metrics(y_true, y_pred, y_pred_proba):
    """Analiza las métricas del modelo"""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score, classification_report)
    
    print("="*60)
    print("ANÁLISIS DE MÉTRICAS DEL MODELO SVM")
    print("="*60)
    print()
    
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    print("MÉTRICAS GENERALES:")
    print("-" * 60)
    print("Accuracy:  {:.4f} ({:.2f}%)".format(accuracy, accuracy*100))
    print("Precision: {:.4f} ({:.2f}%)".format(precision, precision*100))
    print("Recall:    {:.4f} ({:.2f}%)".format(recall, recall*100))
    print("F1-Score:  {:.4f}".format(f1))
    print("AUC-ROC:   {:.4f}".format(auc))
    print()
    
    # Matriz de confusión
    cm, TN, FP, FN, TP = calculate_confusion_matrix(y_true, y_pred)
    
    print("MATRIZ DE CONFUSIÓN:")
    print("-" * 60)
    print("                Predicción")
    print("              Benigno  Maligno")
    print("Real Benigno    {:4d}     {:4d}".format(TN, FP))
    print("Real Maligno    {:4d}     {:4d}".format(FN, TP))
    print()
    print("Leyenda:")
    print("  TN (True Negative):  {} - Correctamente identificados como benignos".format(TN))
    print("  FP (False Positive): {} - Incorrectamente identificados como malignos".format(FP))
    print("  FN (False Negative): {} - Incorrectamente identificados como benignos (¡CRÍTICO!)".format(FN))
    print("  TP (True Positive):  {} - Correctamente identificados como malignos".format(TP))
    print()
    
    # Cálculo de probabilidades y tasas
    total = len(y_true)
    total_malignos = TP + FN  # Total de casos realmente malignos
    total_benignos = TN + FP  # Total de casos realmente benignos
    
    print("ANÁLISIS DE FALSOS NEGATIVOS (CRÍTICO EN DIAGNÓSTICO MÉDICO):")
    print("-" * 60)
    print("Total de casos malignos reales: {}".format(total_malignos))
    print("Falsos negativos (FN): {}".format(FN))
    print()
    
    # Probabilidad de falso negativo
    prob_fn = FN / total_malignos if total_malignos > 0 else 0
    print("PROBABILIDAD DE FALSO NEGATIVO:")
    print("  P(FN) = FN / Total_Malignos = {} / {} = {:.4f} ({:.2f}%)".format(
        FN, total_malignos, prob_fn, prob_fn*100))
    print()
    print("  ⚠️  Esto significa que hay un {:.2f}% de probabilidad de que".format(prob_fn*100))
    print("     un caso maligno sea clasificado incorrectamente como benigno.")
    print()
    
    # Tasa de falsos negativos (False Negative Rate - FNR)
    fnr = FN / total_malignos if total_malignos > 0 else 0
    print("TASA DE FALSOS NEGATIVOS (False Negative Rate - FNR):")
    print("  FNR = FN / (FN + TP) = {} / {} = {:.4f} ({:.2f}%)".format(
        FN, total_malignos, fnr, fnr*100))
    print()
    
    # Recall (Sensitivity) - complemento de FNR
    print("RECALL (Sensibilidad) - Complemento de FNR:")
    print("  Recall = TP / (TP + FN) = {} / {} = {:.4f} ({:.2f}%)".format(
        TP, total_malignos, recall, recall*100))
    print("  Recall = 1 - FNR = 1 - {:.4f} = {:.4f}".format(fnr, 1-fnr))
    print()
    
    # Análisis de falsos positivos
    prob_fp = FP / total_benignos if total_benignos > 0 else 0
    print("ANÁLISIS DE FALSOS POSITIVOS:")
    print("-" * 60)
    print("Total de casos benignos reales: {}".format(total_benignos))
    print("Falsos positivos (FP): {}".format(FP))
    print("Probabilidad de falso positivo: {:.4f} ({:.2f}%)".format(prob_fp, prob_fp*100))
    print()
    
    # Especificidad
    specificity = TN / total_benignos if total_benignos > 0 else 0
    print("ESPECIFICIDAD (True Negative Rate):")
    print("  Specificity = TN / (TN + FP) = {} / {} = {:.4f} ({:.2f}%)".format(
        TN, total_benignos, specificity, specificity*100))
    print()
    
    # Interpretación
    print("="*60)
    print("INTERPRETACIÓN:")
    print("="*60)
    print()
    print("Para diagnóstico médico de cáncer:")
    print()
    print("✓ RECALL (Sensibilidad) = {:.2f}%".format(recall*100))
    print("  - El modelo detecta correctamente el {:.2f}% de los casos malignos".format(recall*100))
    print("  - Esto es BUENO para un modelo médico")
    print()
    print("⚠️  FNR (Tasa de Falsos Negativos) = {:.2f}%".format(fnr*100))
    print("  - El {:.2f}% de los casos malignos NO son detectados".format(fnr*100))
    print("  - En diagnóstico médico, esto es CRÍTICO")
    print("  - Cada falso negativo puede ser un caso de cáncer no detectado")
    print()
    print("✓ PRECISION = {:.2f}%".format(precision*100))
    print("  - De todos los casos predichos como malignos, {:.2f}% realmente lo son".format(precision*100))
    print()
    print("✓ ESPECIFICIDAD = {:.2f}%".format(specificity*100))
    print("  - El modelo identifica correctamente el {:.2f}% de los casos benignos".format(specificity*100))
    print()
    
    # Recomendaciones
    print("="*60)
    print("RECOMENDACIONES:")
    print("="*60)
    print()
    if fnr > 0.10:  # Más del 10% de falsos negativos
        print("⚠️  ALERTA: La tasa de falsos negativos es alta ({:.2f}%)".format(fnr*100))
        print("   Recomendaciones:")
        print("   1. Considerar ajustar el umbral de decisión (bajar threshold)")
        print("   2. Usar técnicas de balanceo de datos (SMOTE)")
        print("   3. Entrenar con más datos de la clase minoritaria")
        print("   4. Usar métricas de costo-asimétrico (penalizar más los FN)")
    elif fnr > 0.05:  # Entre 5% y 10%
        print("✓ La tasa de falsos negativos es moderada ({:.2f}%)".format(fnr*100))
        print("   El modelo es aceptable pero podría mejorarse")
    else:
        print("✓ Excelente: La tasa de falsos negativos es baja ({:.2f}%)".format(fnr*100))
        print("   El modelo tiene buen rendimiento para diagnóstico médico")
    print()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
        'prob_fn': prob_fn,
        'fnr': fnr,
        'specificity': specificity
    }

def main():
    """Función principal"""
    try:
        y_true, y_pred, y_pred_proba, model = load_model_and_data()
        metrics = analyze_metrics(y_true, y_pred, y_pred_proba)
        
        # Guardar análisis detallado
        output_path = os.path.join(BASE_DIR, 'training', 'experiments', 'results', 'svm_detailed_analysis.csv')
        analysis_df = pd.DataFrame([{
            'Métrica': 'Probabilidad_Falso_Negativo',
            'Valor': metrics['prob_fn'],
            'Porcentaje': metrics['prob_fn'] * 100,
            'Interpretación': 'Probabilidad de que un caso maligno sea clasificado como benigno'
        }, {
            'Métrica': 'Tasa_Falsos_Negativos_FNR',
            'Valor': metrics['fnr'],
            'Porcentaje': metrics['fnr'] * 100,
            'Interpretación': 'Tasa de falsos negativos (complemento del Recall)'
        }, {
            'Métrica': 'Recall_Sensibilidad',
            'Valor': metrics['recall'],
            'Porcentaje': metrics['recall'] * 100,
            'Interpretación': 'Capacidad de detectar casos malignos (1 - FNR)'
        }])
        
        analysis_df.to_csv(output_path, index=False)
        print("Análisis detallado guardado en: {}".format(output_path))
        
    except Exception as e:
        print("Error: {}".format(str(e)))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
