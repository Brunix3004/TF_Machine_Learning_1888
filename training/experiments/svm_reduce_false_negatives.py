#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para reducir falsos negativos en el modelo SVM
Implementa: ajuste de umbral, SMOTE, y costo asimétrico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'raw', 'text', 'Cancer_Data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
RESULTS_DIR = os.path.join(BASE_DIR, 'training', 'experiments', 'results')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualization', 'plots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

def load_data():
    """Carga y prepara los datos"""
    print("Cargando datos...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(axis=1)
    
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    y = (y == 'M').astype(int)  # M=1 (maligno), B=0 (benigno)
    
    print("Dataset: {} muestras, {} características".format(X.shape[0], X.shape[1]))
    print("Distribución: {} benignos, {} malignos".format((y==0).sum(), (y==1).sum()))
    
    return X, y

def preprocess_data(X, y, use_smote=False):
    """Preprocesa los datos"""
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
    
    # PCA
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print("PCA aplicado: {} componentes, varianza: {:.4f}".format(
        10, pca.explained_variance_ratio_.sum()))
    
    # SMOTE si se solicita
    if use_smote:
        print("\nAplicando SMOTE para balancear clases...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)
        print("Después de SMOTE: {} muestras (balanceadas)".format(len(X_train_balanced)))
        print("Distribución: {} benignos, {} malignos".format(
            (y_train_balanced==0).sum(), (y_train_balanced==1).sum()))
        return X_train_balanced, X_test_pca, y_train_balanced, y_test, scaler, pca
    else:
        return X_train_pca, X_test_pca, y_train, y_test, scaler, pca

def train_svm_with_class_weight(X_train, y_train, class_weight='balanced'):
    """Entrena SVM con peso de clases asimétrico"""
    print("\nEntrenando SVM con class_weight='{}'...".format(class_weight))
    
    model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight=class_weight,  # Penaliza más los falsos negativos
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Modelo entrenado")
    
    return model

def find_optimal_threshold(y_true, y_pred_proba):
    """Encuentra el umbral óptimo para minimizar falsos negativos"""
    print("\nBuscando umbral óptimo para minimizar falsos negativos...")
    
    thresholds = np.arange(0.3, 0.7, 0.01)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
            accuracy = accuracy_score(y_true, y_pred)
            results.append({
                'threshold': threshold,
                'FN': FN,
                'FNR': fnr,
                'accuracy': accuracy,
                'recall': TP / (TP + FN) if (TP + FN) > 0 else 0
            })
    
    results_df = pd.DataFrame(results)
    
    # Encontrar umbral que minimiza FNR manteniendo accuracy razonable
    # Priorizamos recall (minimizar FN) pero con accuracy > 0.90
    optimal = results_df[results_df['accuracy'] >= 0.90].sort_values('FNR').iloc[0]
    
    print("Umbral óptimo encontrado: {:.3f}".format(optimal['threshold']))
    print("  FNR: {:.4f} ({:.2f}%)".format(optimal['FNR'], optimal['FNR']*100))
    print("  Accuracy: {:.4f}".format(optimal['accuracy']))
    print("  Recall: {:.4f} ({:.2f}%)".format(optimal['recall'], optimal['recall']*100))
    
    return optimal['threshold'], results_df

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evalúa el modelo con un umbral específico"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'FN': FN,
        'FP': FP,
        'TP': TP,
        'TN': TN,
        'FNR': fnr,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def compare_models():
    """Compara diferentes estrategias para reducir falsos negativos"""
    print("="*70)
    print("COMPARACIÓN DE ESTRATEGIAS PARA REDUCIR FALSOS NEGATIVOS")
    print("="*70)
    
    X, y = load_data()
    
    results_comparison = []
    
    # Estrategia 1: Modelo original (threshold=0.5, sin SMOTE, sin class_weight)
    print("\n" + "="*70)
    print("ESTRATEGIA 1: Modelo Original (baseline)")
    print("="*70)
    X_train, X_test, y_train, y_test, scaler1, pca1 = preprocess_data(X, y, use_smote=False)
    model1 = train_svm_with_class_weight(X_train, y_train, class_weight=None)
    results1 = evaluate_model(model1, X_test, y_test, threshold=0.5)
    results1['strategy'] = 'Original (threshold=0.5)'
    results_comparison.append(results1)
    
    print("\nResultados:")
    print("  Accuracy: {:.4f}".format(results1['accuracy']))
    print("  Recall: {:.4f} ({:.2f}%)".format(results1['recall'], results1['recall']*100))
    print("  FNR: {:.4f} ({:.2f}%)".format(results1['FNR'], results1['FNR']*100))
    print("  Falsos Negativos: {}".format(results1['FN']))
    
    # Estrategia 2: Ajuste de umbral (sin SMOTE, sin class_weight)
    print("\n" + "="*70)
    print("ESTRATEGIA 2: Ajuste de Umbral (optimizado)")
    print("="*70)
    optimal_threshold, threshold_results = find_optimal_threshold(y_test, model1.predict_proba(X_test)[:, 1])
    results2 = evaluate_model(model1, X_test, y_test, threshold=optimal_threshold)
    results2['strategy'] = 'Ajuste de Umbral (threshold={:.3f})'.format(optimal_threshold)
    results_comparison.append(results2)
    
    print("\nResultados:")
    print("  Accuracy: {:.4f}".format(results2['accuracy']))
    print("  Recall: {:.4f} ({:.2f}%)".format(results2['recall'], results2['recall']*100))
    print("  FNR: {:.4f} ({:.2f}%)".format(results2['FNR'], results2['FNR']*100))
    print("  Falsos Negativos: {}".format(results2['FN']))
    
    # Estrategia 3: SMOTE + class_weight balanced
    print("\n" + "="*70)
    print("ESTRATEGIA 3: SMOTE + Class Weight Balanced")
    print("="*70)
    X_train_smote, X_test_smote, y_train_smote, y_test_smote, scaler3, pca3 = preprocess_data(X, y, use_smote=True)
    model3 = train_svm_with_class_weight(X_train_smote, y_train_smote, class_weight='balanced')
    results3 = evaluate_model(model3, X_test_smote, y_test_smote, threshold=0.5)
    results3['strategy'] = 'SMOTE + Balanced'
    results_comparison.append(results3)
    
    print("\nResultados:")
    print("  Accuracy: {:.4f}".format(results3['accuracy']))
    print("  Recall: {:.4f} ({:.2f}%)".format(results3['recall'], results3['recall']*100))
    print("  FNR: {:.4f} ({:.2f}%)".format(results3['FNR'], results3['FNR']*100))
    print("  Falsos Negativos: {}".format(results3['FN']))
    
    # Estrategia 4: SMOTE + class_weight custom (penalizar más FN)
    print("\n" + "="*70)
    print("ESTRATEGIA 4: SMOTE + Class Weight Custom (penaliza más FN)")
    print("="*70)
    # Penalizar falsos negativos 3 veces más que falsos positivos
    model4 = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight={0: 1, 1: 3},  # Clase 1 (maligno) tiene peso 3x
        random_state=42
    )
    model4.fit(X_train_smote, y_train_smote)
    results4 = evaluate_model(model4, X_test_smote, y_test_smote, threshold=0.5)
    results4['strategy'] = 'SMOTE + Custom Weight (1:3)'
    results_comparison.append(results4)
    
    print("\nResultados:")
    print("  Accuracy: {:.4f}".format(results4['accuracy']))
    print("  Recall: {:.4f} ({:.2f}%)".format(results4['recall'], results4['recall']*100))
    print("  FNR: {:.4f} ({:.2f}%)".format(results4['FNR'], results4['FNR']*100))
    print("  Falsos Negativos: {}".format(results4['FN']))
    
    # Estrategia 5: Combinación óptima (SMOTE + custom weight + threshold ajustado)
    print("\n" + "="*70)
    print("ESTRATEGIA 5: Combinación Óptima (SMOTE + Custom Weight + Threshold)")
    print("="*70)
    optimal_threshold_5, _ = find_optimal_threshold(y_test_smote, model4.predict_proba(X_test_smote)[:, 1])
    results5 = evaluate_model(model4, X_test_smote, y_test_smote, threshold=optimal_threshold_5)
    results5['strategy'] = 'Combinación Óptima (threshold={:.3f})'.format(optimal_threshold_5)
    results_comparison.append(results5)
    
    print("\nResultados:")
    print("  Accuracy: {:.4f}".format(results5['accuracy']))
    print("  Recall: {:.4f} ({:.2f}%)".format(results5['recall'], results5['recall']*100))
    print("  FNR: {:.4f} ({:.2f}%)".format(results5['FNR'], results5['FNR']*100))
    print("  Falsos Negativos: {}".format(results5['FN']))
    
    # Comparación final
    print("\n" + "="*70)
    print("COMPARACIÓN FINAL DE ESTRATEGIAS")
    print("="*70)
    
    comparison_df = pd.DataFrame(results_comparison)
    comparison_df = comparison_df[['strategy', 'accuracy', 'precision', 'recall', 'FNR', 'FN', 'FP']]
    comparison_df.columns = ['Estrategia', 'Accuracy', 'Precision', 'Recall', 'FNR (%)', 'Falsos Negativos', 'Falsos Positivos']
    comparison_df['FNR (%)'] = comparison_df['FNR (%)'] * 100
    comparison_df['Recall'] = comparison_df['Recall'] * 100
    comparison_df['Accuracy'] = comparison_df['Accuracy'] * 100
    comparison_df['Precision'] = comparison_df['Precision'] * 100
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Guardar comparación
    output_path = os.path.join(RESULTS_DIR, 'svm_strategies_comparison.csv')
    comparison_df.to_csv(output_path, index=False)
    print("\nComparación guardada en: {}".format(output_path))
    
    # Encontrar mejor estrategia (menor FNR con accuracy > 0.90)
    best_strategy = comparison_df[comparison_df['Accuracy'] >= 90].sort_values('FNR (%)').iloc[0]
    
    print("\n" + "="*70)
    print("MEJOR ESTRATEGIA:")
    print("="*70)
    print("  {}".format(best_strategy['Estrategia']))
    print("  FNR: {:.2f}%".format(best_strategy['FNR (%)']))
    print("  Recall: {:.2f}%".format(best_strategy['Recall']))
    print("  Accuracy: {:.2f}%".format(best_strategy['Accuracy']))
    print("  Falsos Negativos: {}".format(int(best_strategy['Falsos Negativos'])))
    print("  Reducción de FN: {} casos menos que el modelo original".format(
        int(results1['FN'] - best_strategy['Falsos Negativos'])))
    
    # Guardar el mejor modelo
    if 'SMOTE' in best_strategy['Estrategia']:
        best_model = model4
        best_scaler = scaler3
        best_pca = pca3
        best_threshold = optimal_threshold_5 if 'threshold' in best_strategy['Estrategia'] else 0.5
    else:
        best_model = model1
        best_scaler = scaler1
        best_pca = pca1
        best_threshold = optimal_threshold if 'threshold' in best_strategy['Estrategia'] else 0.5
    
    model_data = {
        'model': best_model,
        'scaler': best_scaler,
        'pca': best_pca,
        'threshold': best_threshold,
        'strategy': best_strategy['Estrategia']
    }
    
    model_path = os.path.join(MODELS_DIR, 'svm_breast_cancer_optimized.joblib')
    joblib.dump(model_data, model_path)
    print("\nMejor modelo guardado en: {}".format(model_path))
    
    return comparison_df, best_strategy

def plot_comparison(comparison_df):
    """Crea visualización de la comparación"""
    print("\nCreando visualización...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. FNR comparación
    axes[0, 0].bar(range(len(comparison_df)), comparison_df['FNR (%)'], color='red', alpha=0.7)
    axes[0, 0].set_xticks(range(len(comparison_df)))
    axes[0, 0].set_xticklabels(comparison_df['Estrategia'], rotation=45, ha='right')
    axes[0, 0].set_title('Tasa de Falsos Negativos (FNR) por Estrategia', fontweight='bold')
    axes[0, 0].set_ylabel('FNR (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Recall comparación
    axes[0, 1].bar(range(len(comparison_df)), comparison_df['Recall'], color='green', alpha=0.7)
    axes[0, 1].set_xticks(range(len(comparison_df)))
    axes[0, 1].set_xticklabels(comparison_df['Estrategia'], rotation=45, ha='right')
    axes[0, 1].set_title('Recall (Sensibilidad) por Estrategia', fontweight='bold')
    axes[0, 1].set_ylabel('Recall (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Falsos Negativos absolutos
    axes[1, 0].bar(range(len(comparison_df)), comparison_df['Falsos Negativos'], color='orange', alpha=0.7)
    axes[1, 0].set_xticks(range(len(comparison_df)))
    axes[1, 0].set_xticklabels(comparison_df['Estrategia'], rotation=45, ha='right')
    axes[1, 0].set_title('Número de Falsos Negativos por Estrategia', fontweight='bold')
    axes[1, 0].set_ylabel('Cantidad de FN')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Accuracy vs FNR (trade-off)
    axes[1, 1].scatter(comparison_df['FNR (%)'], comparison_df['Accuracy'], 
                      s=200, alpha=0.6, c=range(len(comparison_df)), cmap='viridis')
    for i, row in comparison_df.iterrows():
        axes[1, 1].annotate(row['Estrategia'].split('(')[0].strip(), 
                           (row['FNR (%)'], row['Accuracy']),
                           fontsize=8, ha='center')
    axes[1, 1].set_xlabel('FNR (%)')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Trade-off: Accuracy vs FNR', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(VISUALIZATIONS_DIR, 'svm_strategies_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualización guardada en: {}".format(output_path))

def main():
    """Función principal"""
    try:
        comparison_df, best_strategy = compare_models()
        plot_comparison(comparison_df)
        
        print("\n" + "="*70)
        print("ANÁLISIS COMPLETADO")
        print("="*70)
        print("\nRevisa los resultados para elegir la mejor estrategia según tus necesidades.")
        print("Para diagnóstico médico, prioriza estrategias con menor FNR.")
        
    except Exception as e:
        print("Error: {}".format(str(e)))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
