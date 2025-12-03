#!/usr/bin/env python3
"""
Script para entrenar modelo SVM con el dataset Cancer_Data.csv
Incluye análisis de características, PCA y optimización de hiperparámetros
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class SVMModel:
    """
    Clase para entrenar y evaluar modelo SVM con el dataset de cáncer
    """
    
    def __init__(self, file_path):
        """
        Inicializa el modelo SVM
        """
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.pca = None
        self.svm_model = None
        self.best_params = None
        
    def load_and_prepare_data(self):
        """
        Carga y prepara los datos
        """
        print("📊 Cargando y preparando datos...")
        
        # Cargar dataset
        self.df = pd.read_csv(self.file_path)
        
        # Eliminar columnas con valores NaN (como 'Unnamed: 32')
        self.df = self.df.dropna(axis=1)
        
        # Separar características y variable objetivo
        self.X = self.df.drop(['id', 'diagnosis'], axis=1)
        self.y = self.df['diagnosis']
        
        # Convertir variable objetivo a numérica
        self.y = (self.y == 'M').astype(int)  # M=1, B=0
        
        print(f"   ✅ Dataset cargado: {self.X.shape[0]} muestras, {self.X.shape[1]} características")
        print(f"   📊 Distribución de clases: {np.bincount(self.y)}")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Divide los datos en entrenamiento y prueba
        """
        print("🔄 Dividiendo datos en entrenamiento y prueba...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"   ✅ Datos divididos:")
        print(f"     • Entrenamiento: {self.X_train.shape[0]} muestras")
        print(f"     • Prueba: {self.X_test.shape[0]} muestras")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess_data(self, use_pca=True, n_components=10):
        """
        Preprocesa los datos (estandarización y opcionalmente PCA)
        """
        print("🔧 Preprocesando datos...")
        
        # Estandarizar datos
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        if use_pca:
            # Usar 10 componentes PCA como recomendó el análisis anterior
            print(f"   📊 Usando {n_components} componentes PCA (recomendación del análisis)")
            
            # Aplicar PCA
            self.pca = PCA(n_components=n_components)
            self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
            self.X_test_pca = self.pca.transform(self.X_test_scaled)
            
            print(f"   ✅ PCA aplicado: {self.X_train_pca.shape[1]} componentes")
            print(f"   📈 Varianza explicada: {self.pca.explained_variance_ratio_.sum():.4f}")
            
            return self.X_train_pca, self.X_test_pca
        else:
            print("   ✅ Solo estandarización aplicada (sin PCA)")
            return self.X_train_scaled, self.X_test_scaled
    
    def train_svm_basic(self, X_train, y_train):
        """
        Entrena un modelo SVM básico
        """
        print("🤖 Entrenando modelo SVM básico...")
        
        self.svm_model = SVC(random_state=42)
        self.svm_model.fit(X_train, y_train)
        
        print("   ✅ Modelo SVM básico entrenado")
        return self.svm_model
    
    def optimize_hyperparameters(self, X_train, y_train, cv=5):
        """
        Optimiza los hiperparámetros del modelo SVM
        """
        print("🔍 Optimizando hiperparámetros...")
        
        # Definir grid de parámetros más pequeño para eficiencia
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        
        # Crear modelo base
        svm = SVC(random_state=42, probability=True)
        
        # Grid search con validación cruzada
        grid_search = GridSearchCV(
            svm, param_grid, cv=cv, scoring='accuracy', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        self.svm_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"   ✅ Mejores parámetros encontrados:")
        for param, value in self.best_params.items():
            print(f"     • {param}: {value}")
        print(f"   📊 Mejor score de validación: {grid_search.best_score_:.4f}")
        
        return self.svm_model, self.best_params
    
    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo entrenado
        """
        print("📊 Evaluando modelo...")
        
        # Predicciones
        y_pred = self.svm_model.predict(X_test)
        y_pred_proba = self.svm_model.predict_proba(X_test)[:, 1] if hasattr(self.svm_model, 'predict_proba') else None
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"   📈 Métricas de rendimiento:")
        print(f"     • Accuracy: {accuracy:.4f}")
        print(f"     • Precision: {precision:.4f}")
        print(f"     • Recall: {recall:.4f}")
        print(f"     • F1-Score: {f1:.4f}")
        
        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"     • AUC-ROC: {auc:.4f}")
        
        # Reporte detallado
        print(f"\n   📋 Reporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_confusion_matrix(self, y_test, y_pred):
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
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """
        Crea curva ROC
        """
        if y_pred_proba is None:
            print("   ⚠️ No se pueden generar probabilidades para la curva ROC")
            return
        
        print("📊 Creando curva ROC...")
        
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
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
    
    def cross_validation_analysis(self, X, y, cv=5):
        """
        Realiza análisis de validación cruzada
        """
        print(f"🔄 Realizando validación cruzada ({cv} folds)...")
        
        # Métricas para validación cruzada
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        cv_results = {}
        
        for metric in scoring:
            scores = cross_val_score(self.svm_model, X, y, cv=cv, scoring=metric)
            cv_results[metric] = scores
            
            print(f"   📊 {metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def plot_learning_curves(self, X_train, y_train, X_test, y_test):
        """
        Crea gráficos de curvas de aprendizaje
        """
        print("📊 Creando curvas de aprendizaje...")
        
        from sklearn.model_selection import learning_curve
        
        # Definir tamaños de entrenamiento
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Calcular curvas de aprendizaje
        train_sizes_abs, train_scores, val_scores = learning_curve(
            self.svm_model, X_train, y_train, 
            train_sizes=train_sizes, cv=5, 
            scoring='accuracy', n_jobs=-1
        )
        
        # Calcular medias y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        # Curva de entrenamiento
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', 
                label='Entrenamiento', linewidth=2, markersize=6)
        plt.fill_between(train_sizes_abs, train_mean - train_std, 
                        train_mean + train_std, alpha=0.1, color='blue')
        
        # Curva de validación
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', 
                label='Validación', linewidth=2, markersize=6)
        plt.fill_between(train_sizes_abs, val_mean - val_std, 
                        val_mean + val_std, alpha=0.1, color='red')
        
        plt.title('Curvas de Aprendizaje - Modelo SVM', fontsize=16, fontweight='bold')
        plt.xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico
        output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/svm_learning_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Curvas de aprendizaje guardadas en: {output_path}")
        
        return train_sizes_abs, train_mean, val_mean
    
    def plot_validation_curves(self, X_train, y_train):
        """
        Crea gráficos de curvas de validación para hiperparámetros
        """
        print("📊 Creando curvas de validación...")
        
        from sklearn.model_selection import validation_curve
        
        # Definir rangos de parámetros
        param_range = np.logspace(-3, 3, 7)  # C: 0.001 a 1000
        
        # Calcular curvas de validación para C
        train_scores, val_scores = validation_curve(
            SVC(kernel='rbf', gamma='scale', random_state=42),
            X_train, y_train, param_name='C', param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        # Calcular medias y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        # Curva de entrenamiento
        plt.semilogx(param_range, train_mean, 'o-', color='blue', 
                    label='Entrenamiento', linewidth=2, markersize=6)
        plt.fill_between(param_range, train_mean - train_std, 
                        train_mean + train_std, alpha=0.1, color='blue')
        
        # Curva de validación
        plt.semilogx(param_range, val_mean, 'o-', color='red', 
                    label='Validación', linewidth=2, markersize=6)
        plt.fill_between(param_range, val_mean - val_std, 
                        val_mean + val_std, alpha=0.1, color='red')
        
        plt.title('Curvas de Validación - Parámetro C', fontsize=16, fontweight='bold')
        plt.xlabel('Valor de C', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico
        output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/visualization/plots/svm_validation_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Curvas de validación guardadas en: {output_path}")
        
        return param_range, train_mean, val_mean
    
    def save_model_results(self, results, cv_results=None):
        """
        Guarda los resultados del modelo
        """
        print("💾 Guardando resultados...")
        
        # Crear DataFrame con resultados
        results_data = {
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Valor': [
                results['accuracy'],
                results['precision'],
                results['recall'],
                results['f1']
            ]
        }
        
        if cv_results:
            results_data['CV_Mean'] = [
                cv_results['accuracy'].mean(),
                cv_results['precision'].mean(),
                cv_results['recall'].mean(),
                cv_results['f1'].mean()
            ]
            results_data['CV_Std'] = [
                cv_results['accuracy'].std(),
                cv_results['precision'].std(),
                cv_results['recall'].std(),
                cv_results['f1'].std()
            ]
        
        results_df = pd.DataFrame(results_data)
        
        # Guardar resultados
        output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/training/results/svm_results.csv'
        results_df.to_csv(output_path, index=False)
        
        print(f"   ✅ Resultados guardados en: {output_path}")
        
        return results_df
    
    def save_optimal_parameters(self):
        """
        Guarda los parámetros óptimos del modelo
        """
        print("💾 Guardando parámetros óptimos...")
        
        if self.best_params is None:
            print("   ⚠️ No hay parámetros óptimos para guardar")
            return None
        
        # Crear diccionario con información completa
        optimal_params = {
            'model_type': 'SVM',
            'best_parameters': self.best_params,
            'pca_components': 10,
            'pca_variance_explained': self.pca.explained_variance_ratio_.sum() if self.pca else None,
            'feature_count_original': 30,
            'feature_count_pca': 10,
            'optimization_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Guardar como JSON
        import json
        output_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/training/results/svm_optimal_parameters.json'
        with open(output_path, 'w') as f:
            json.dump(optimal_params, f, indent=2)
        
        print(f"   ✅ Parámetros óptimos guardados en: {output_path}")
        
        # También guardar como CSV para fácil lectura
        params_df = pd.DataFrame([
            {'Parámetro': k, 'Valor': v} for k, v in self.best_params.items()
        ])
        csv_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/training/results/svm_optimal_parameters.csv'
        params_df.to_csv(csv_path, index=False)
        
        print(f"   ✅ Parámetros óptimos (CSV) guardados en: {csv_path}")
        
        return optimal_params

def main():
    """
    Función principal para entrenar el modelo SVM
    """
    print("🤖 INICIANDO ENTRENAMIENTO DE MODELO SVM")
    print("="*60)
    
    # Inicializar modelo
    file_path = '/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/raw/text/Cancer_Data.csv'
    svm_model = SVMModel(file_path)
    
    try:
        # Cargar y preparar datos
        X, y = svm_model.load_and_prepare_data()
        
        # Dividir datos
        X_train, X_test, y_train, y_test = svm_model.split_data()
        
        # Preprocesar datos (con PCA)
        X_train_processed, X_test_processed = svm_model.preprocess_data(use_pca=True)
        
        # Entrenar modelo básico
        print("\n" + "="*40)
        print("MODELO SVM BÁSICO")
        print("="*40)
        svm_model.train_svm_basic(X_train_processed, y_train)
        basic_results = svm_model.evaluate_model(X_test_processed, y_test)
        
        # Optimizar hiperparámetros
        print("\n" + "="*40)
        print("OPTIMIZACIÓN DE HIPERPARÁMETROS")
        print("="*40)
        svm_model.optimize_hyperparameters(X_train_processed, y_train)
        optimized_results = svm_model.evaluate_model(X_test_processed, y_test)
        
        # Crear visualizaciones
        print("\n" + "="*40)
        print("VISUALIZACIONES")
        print("="*40)
        svm_model.plot_confusion_matrix(y_test, optimized_results['y_pred'])
        svm_model.plot_roc_curve(y_test, optimized_results['y_pred_proba'])
        
        # Curvas de aprendizaje y validación
        print("\n" + "="*40)
        print("CURVAS DE APRENDIZAJE Y VALIDACIÓN")
        print("="*40)
        svm_model.plot_learning_curves(X_train_processed, y_train, X_test_processed, y_test)
        svm_model.plot_validation_curves(X_train_processed, y_train)
        
        # Validación cruzada
        print("\n" + "="*40)
        print("VALIDACIÓN CRUZADA")
        print("="*40)
        cv_results = svm_model.cross_validation_analysis(X_train_processed, y_train)
        
        # Guardar resultados
        print("\n" + "="*40)
        print("GUARDANDO RESULTADOS")
        print("="*40)
        results_df = svm_model.save_model_results(optimized_results, cv_results)
        
        # Guardar parámetros óptimos
        optimal_params = svm_model.save_optimal_parameters()
        
        print(f"\n✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"   • Modelo SVM optimizado entrenado")
        print(f"   • Accuracy final: {optimized_results['accuracy']:.4f}")
        print(f"   • Mejores parámetros: {svm_model.best_params}")
        print(f"   • Archivos generados: visualizaciones, resultados CSV y parámetros óptimos")
        
        return svm_model, optimized_results
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        return None, None

if __name__ == "__main__":
    model, results = main()
