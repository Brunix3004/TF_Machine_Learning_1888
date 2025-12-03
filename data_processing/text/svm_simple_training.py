#!/usr/bin/env python3
"""
Script simplificado para entrenar modelo SVM con gráficos y métricas
Sin optimización pesada de hiperparámetros
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Carga y prepara los datos
    """
    print("📊 Cargando y preparando datos...")
    
    # Cargar dataset
    file_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/raw/text/Cancer_Data.csv'
    df = pd.read_csv(file_path)
    
    # Eliminar columnas con valores NaN
    df = df.dropna(axis=1)
    
    # Separar características y variable objetivo
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    
    # Convertir variable objetivo a numérica
    y = (y == 'M').astype(int)  # M=1, B=0
    
    print(f"   ✅ Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"   📊 Distribución de clases: {np.bincount(y)}")
    
    return X, y

def preprocess_data(X, y):
    """
    Preprocesa los datos con PCA
    """
    print("🔧 Preprocesando datos...")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✅ Datos divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
    
    # Estandarizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Aplicar PCA con 10 componentes
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"   ✅ PCA aplicado: {X_train_pca.shape[1]} componentes")
    print(f"   📈 Varianza explicada: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_train_pca, X_test_pca, y_train, y_test, pca

def train_svm_model(X_train, y_train):
    """
    Entrena modelo SVM
    """
    print("🤖 Entrenando modelo SVM...")
    
    # Usar parámetros razonables sin optimización
    svm_model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    svm_model.fit(X_train, y_train)
    print("   ✅ Modelo SVM entrenado")
    
    return svm_model

def evaluate_model(svm_model, X_test, y_test):
    """
    Evalúa el modelo
    """
    print("📊 Evaluando modelo...")
    
    # Predicciones
    y_pred = svm_model.predict(X_test)
    y_pred_proba = svm_model.predict_proba(X_test)[:, 1]
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"   📈 Métricas de rendimiento:")
    print(f"     • Accuracy: {accuracy:.4f}")
    print(f"     • Precision: {precision:.4f}")
    print(f"     • Recall: {recall:.4f}")
    print(f"     • F1-Score: {f1:.4f}")
    print(f"     • AUC-ROC: {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def create_confusion_matrix(y_test, y_pred):
    """
    Crea matriz de confusión
    """
    print("📊 Creando matriz de confusión...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benigno', 'Maligno'],
                yticklabels=['Benigno', 'Maligno'])
    plt.title('Matriz de Confusión - Modelo SVM', fontsize=16, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    
    # Guardar gráfico
    output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/svm_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Matriz de confusión guardada en: {output_path}")

def create_roc_curve(y_test, y_pred_proba):
    """
    Crea curva ROC
    """
    print("📊 Creando curva ROC...")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curva ROC - Modelo SVM', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico
    output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/svm_roc_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Curva ROC guardada en: {output_path}")

def create_metrics_comparison(results, cv_results):
    """
    Crea gráfico de comparación de métricas
    """
    print("📊 Creando gráfico de métricas...")
    
    # Preparar datos
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [
        results['accuracy'],
        results['precision'],
        results['recall'],
        results['f1']
    ]
    
    cv_values = [
        cv_results['accuracy'].mean(),
        cv_results['precision'].mean(),
        cv_results['recall'].mean(),
        cv_results['f1'].mean()
    ]
    
    cv_std = [
        cv_results['accuracy'].std(),
        cv_results['precision'].std(),
        cv_results['recall'].std(),
        cv_results['f1'].std()
    ]
    
    # Crear gráfico
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Test Set', 
                   color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, cv_values, width, label='Cross-Validation', 
                   color='lightcoral', alpha=0.8, yerr=cv_std, capsize=5)
    
    ax.set_xlabel('Métricas', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparación de Métricas - Train vs Validation', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for i, (v1, v2) in enumerate(zip(train_values, cv_values)):
        ax.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom')
        ax.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/svm_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Gráfico de métricas guardado en: {output_path}")

def cross_validation_analysis(svm_model, X_train, y_train):
    """
    Realiza validación cruzada
    """
    print("🔄 Realizando validación cruzada...")
    
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = {}
    
    for metric in scoring:
        scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring=metric)
        cv_results[metric] = scores
        print(f"   📊 {metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_results

def save_results(results, cv_results):
    """
    Guarda resultados en archivos
    """
    print("💾 Guardando resultados...")
    
    # Crear DataFrame con resultados
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
            np.nan  # AUC no está en CV por defecto
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
    
    # Guardar resultados
    output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/training/results/svm_results.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"   ✅ Resultados guardados en: {output_path}")
    
    # Guardar parámetros del modelo
    model_params = {
        'model_type': 'SVM',
        'parameters': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True
        },
        'pca_components': 10,
        'feature_count_original': 30,
        'feature_count_pca': 10
    }
    
    import json
    params_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/training/results/svm_model_parameters.json'
    with open(params_path, 'w') as f:
        json.dump(model_params, f, indent=2)
    
    print(f"   ✅ Parámetros del modelo guardados en: {params_path}")
    
    return results_df

def main():
    """
    Función principal
    """
    print("🤖 ENTRENAMIENTO SIMPLIFICADO DE MODELO SVM")
    print("="*60)
    
    try:
        # Cargar datos
        X, y = load_and_prepare_data()
        
        # Preprocesar
        X_train, X_test, y_train, y_test, pca = preprocess_data(X, y)
        
        # Entrenar modelo
        svm_model = train_svm_model(X_train, y_train)
        
        # Evaluar modelo
        results = evaluate_model(svm_model, X_test, y_test)
        
        # Crear visualizaciones
        print("\n" + "="*40)
        print("CREANDO VISUALIZACIONES")
        print("="*40)
        create_confusion_matrix(y_test, results['y_pred'])
        create_roc_curve(y_test, results['y_pred_proba'])
        
        # Validación cruzada
        print("\n" + "="*40)
        print("VALIDACIÓN CRUZADA")
        print("="*40)
        cv_results = cross_validation_analysis(svm_model, X_train, y_train)
        
        # Crear gráfico de métricas
        print("\n" + "="*40)
        print("GRÁFICO DE MÉTRICAS")
        print("="*40)
        create_metrics_comparison(results, cv_results)
        
        # Guardar resultados
        print("\n" + "="*40)
        print("GUARDANDO RESULTADOS")
        print("="*40)
        results_df = save_results(results, cv_results)
        
        print(f"\n✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"   • Accuracy: {results['accuracy']:.4f}")
        print(f"   • F1-Score: {results['f1']:.4f}")
        print(f"   • AUC-ROC: {results['auc']:.4f}")
        print(f"   • Archivos generados: visualizaciones y resultados CSV")
        
        return svm_model, results
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        return None, None

if __name__ == "__main__":
    model, results = main()
