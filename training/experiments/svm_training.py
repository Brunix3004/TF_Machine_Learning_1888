#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Nota: Este script requiere Python 3.7+
# Ejecutar con: python3 svm_training.py
# Si estás en entorno conda 'ml', usa: /opt/homebrew/Caskroom/miniconda/base/envs/ml/bin/python3 svm_training.py
"""
Script de entrenamiento para modelo SVM con dataset Cancer_Data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'raw', 'text', 'Cancer_Data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualization', 'plots')
RESULTS_DIR = os.path.join(BASE_DIR, 'training', 'experiments', 'results')

# Crear directorios si no existen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """Carga y prepara el dataset"""
    print("Cargando dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Eliminar columnas con NaN
    df = df.dropna(axis=1)
    
    # Separar características y target
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    
    # Convertir target a numérico: M=1 (maligno), B=0 (benigno)
    y = (y == 'M').astype(int)
    
    print("Dataset cargado: {} muestras, {} características".format(X.shape[0], X.shape[1]))
    print("Distribución de clases: {}".format(np.bincount(y)))
    
    return X, y

def preprocess_data(X, y, use_pca=True, n_components=10):
    """Preprocesa los datos: división, escalado y PCA"""
    print("\nPreprocesando datos...")
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Train: {} muestras, Test: {} muestras".format(X_train.shape[0], X_test.shape[0]))
    
    # Estandarización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if use_pca:
        # Aplicar PCA
        pca = PCA(n_components=n_components)
        X_train_processed = pca.fit_transform(X_train_scaled)
        X_test_processed = pca.transform(X_test_scaled)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        print("PCA aplicado: {} componentes, varianza explicada: {:.4f}".format(n_components, variance_explained))
        
        return X_train_processed, X_test_processed, y_train, y_test, scaler, pca
    else:
        print("Solo estandarización aplicada (sin PCA)")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, None

def train_svm(X_train, y_train, optimize=True):
    """Entrena el modelo SVM"""
    print("\nEntrenando modelo SVM...")
    
    if optimize:
        print("Optimizando hiperparámetros con GridSearchCV...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        
        svm = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print("Mejores parámetros: {}".format(best_params))
        print("Mejor score CV: {:.4f}".format(grid_search.best_score_))
        
        return model, best_params
    else:
        # Modelo con parámetros por defecto
        model = SVC(C=1.0, kernel='rbf', gamma='scale', 
                   probability=True, random_state=42)
        model.fit(X_train, y_train)
        print("Modelo entrenado con parámetros por defecto")
        return model, None

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y retorna métricas"""
    print("\nEvaluando modelo...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\nMétricas de rendimiento:")
    print("  Accuracy:  {:.4f}".format(accuracy))
    print("  Precision: {:.4f}".format(precision))
    print("  Recall:    {:.4f}".format(recall))
    print("  F1-Score:  {:.4f}".format(f1))
    print("  AUC-ROC:   {:.4f}".format(auc))
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrix(y_test, y_pred, save_path):
    """Crea y guarda matriz de confusión"""
    print("\nCreando matriz de confusión...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benigno', 'Maligno'],
                yticklabels=['Benigno', 'Maligno'])
    plt.title('Matriz de Confusión - SVM', fontsize=14, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Matriz de confusión guardada en: {}".format(save_path))

def plot_roc_curve(y_test, y_pred_proba, save_path):
    """Crea y guarda curva ROC"""
    print("Creando curva ROC...")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label='ROC curve (AUC = {:.3f})'.format(auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curva ROC - SVM', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Curva ROC guardada en: {}".format(save_path))

def cross_validation(model, X_train, y_train):
    """Realiza validación cruzada"""
    print("\nRealizando validación cruzada (5-fold)...")
    
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = {}
    
    for metric in scoring:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=metric)
        cv_results[metric] = scores
        print("  {}: {:.4f} (+/- {:.4f})".format(metric.capitalize(), scores.mean(), scores.std() * 2))
    
    return cv_results

def save_model(model, scaler, pca, best_params, save_path):
    """Guarda el modelo y preprocesadores"""
    print("\nGuardando modelo en: {}".format(save_path))
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'best_params': best_params
    }
    
    joblib.dump(model_data, save_path)
    print("Modelo guardado exitosamente")

def save_results(results, cv_results, best_params, save_path):
    """Guarda resultados en CSV"""
    print("\nGuardando resultados en: {}".format(save_path))
    
    results_data = {
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Test_Score': [
            results['accuracy'],
            results['precision'],
            results['recall'],
            results['f1'],
            results['auc']
        ],
        'CV_Mean': [
            cv_results['accuracy'].mean(),
            cv_results['precision'].mean(),
            cv_results['recall'].mean(),
            cv_results['f1'].mean(),
            np.nan
        ],
        'CV_Std': [
            cv_results['accuracy'].std(),
            cv_results['precision'].std(),
            cv_results['recall'].std(),
            cv_results['f1'].std(),
            np.nan
        ]
    }
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(save_path, index=False)
    print("Resultados guardados exitosamente")
    
    # Guardar parámetros si existen
    if best_params:
        params_path = save_path.replace('.csv', '_params.json')
        import json
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print("Parámetros guardados en: {}".format(params_path))

def main():
    """Función principal"""
    print("="*60)
    print("ENTRENAMIENTO DE MODELO SVM")
    print("="*60)
    
    try:
        # 1. Cargar datos
        X, y = load_data()
        
        # 2. Preprocesar datos
        X_train, X_test, y_train, y_test, scaler, pca = preprocess_data(
            X, y, use_pca=True, n_components=10
        )
        
        # 3. Entrenar modelo (con optimización)
        model, best_params = train_svm(X_train, y_train, optimize=True)
        
        # 4. Evaluar modelo
        results = evaluate_model(model, X_test, y_test)
        
        # 5. Validación cruzada
        cv_results = cross_validation(model, X_train, y_train)
        
        # 6. Visualizaciones
        cm_path = os.path.join(VISUALIZATIONS_DIR, 'svm_confusion_matrix.png')
        roc_path = os.path.join(VISUALIZATIONS_DIR, 'svm_roc_curve.png')
        plot_confusion_matrix(y_test, results['y_pred'], cm_path)
        plot_roc_curve(y_test, results['y_pred_proba'], roc_path)
        
        # 7. Guardar modelo
        model_path = os.path.join(MODELS_DIR, 'svm_breast_cancer.joblib')
        save_model(model, scaler, pca, best_params, model_path)
        
        # 8. Guardar resultados
        results_path = os.path.join(RESULTS_DIR, 'svm_results.csv')
        save_results(results, cv_results, best_params, results_path)
        
        print("\n" + "="*60)
        print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("Accuracy final: {:.4f}".format(results['accuracy']))
        print("F1-Score: {:.4f}".format(results['f1']))
        print("AUC-ROC: {:.4f}".format(results['auc']))
        print("\nArchivos generados:")
        print("  - Modelo: {}".format(model_path))
        print("  - Resultados: {}".format(results_path))
        print("  - Visualizaciones: {}, {}".format(cm_path, roc_path))
        
        return model, results
        
    except Exception as e:
        print("\nError durante el entrenamiento: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    model, results = main()
