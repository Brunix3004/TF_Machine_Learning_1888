#!/usr/bin/env python3
"""
CNN Básica para Clasificación de Lesiones Pulmonares
Arquitectura: 3 bloques convolucionales + clasificador
Dataset: CLAHE procesado (1024x1024 -> 512x512)
Clases: 3 (Nodule/Mass, Other lesion, No Findings)

Autor: Sistema de Machine Learning
Fecha: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuración de paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = Path("/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/datasets/processed/images_clahe")
MODEL_SAVE_PATH = Path("/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/models/saved")
RESULTS_PATH = Path("/Users/benjidry/Documents/Github/TF_Machine_Learning_1888/training/results")

# Crear directorios
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

class BasicCNN(nn.Module):
    """
    CNN Básica con 3 bloques convolucionales para clasificación de lesiones pulmonares.
    
    Arquitectura:
    - Entrada: [batch, 1, 512, 512]
    - 3 bloques convolucionales con BatchNorm y MaxPool
    - Global Average Pooling
    - Clasificador con 2 capas FC
    """
    
    def __init__(self, num_classes=2, dropout_rates=[0.3, 0.4, 0.5]):
        super(BasicCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rates = dropout_rates
        
        # Bloque 1: 1 -> 64 canales
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloque 2: 64 -> 128 canales
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2d_1 = nn.Dropout2d(dropout_rates[0])
        
        # Bloque 3: 128 -> 256 canales
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout2d_2 = nn.Dropout2d(dropout_rates[1])
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Clasificador
        self.fc1 = nn.Linear(256, 128)
        self.dropout1d = nn.Dropout(dropout_rates[2])
        self.fc2 = nn.Linear(128, num_classes)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización de pesos según especificaciones."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming/He Normal initialization para ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm: weights=1, bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Linear: normal(0, 0.01), bias=0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass de la CNN."""
        # Bloque 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Bloque 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d_1(x)
        
        # Bloque 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d_2(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Clasificador
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_summary(self):
        """Obtiene resumen del modelo."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': '3 Conv Blocks + Global Avg Pool + 2 FC',
            'input_size': '(1, 512, 512)',
            'output_size': f'({self.num_classes},)',
            'dropout_rates': self.dropout_rates
        }


class LungDataset(Dataset):
    """
    Dataset personalizado para imágenes pulmonares con CLAHE.
    """
    
    def __init__(self, data_dir, labels_dir, transform=None, is_training=True):
        self.data_dir = Path(data_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Mapeo de clases
        self.class_names = ['Nodule/Mass', 'Other lesion']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Cargar datos
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Carga las muestras del dataset."""
        samples = []
        
        # Buscar imágenes recursivamente
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for ext in image_extensions:
            # Buscar en el directorio y subdirectorios
            for img_path in self.data_dir.rglob(f'*{ext}'):
                # Buscar etiqueta correspondiente
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    # Leer etiqueta
                    with open(label_path, 'r') as f:
                        label_content = f.read().strip()
                    
                    # Parsear etiqueta (formato YOLO: class_id x_center y_center width height)
                    if label_content:
                        lines = label_content.split('\n')
                        # Tomar la primera clase encontrada
                        class_id = int(lines[0].split()[0])
                        samples.append((img_path, class_id))
                    else:
                        # Si no hay etiqueta, asumir clase "Other lesion" (1)
                        samples.append((img_path, 1))
                else:
                    # Si no hay archivo de etiqueta, asumir "Other lesion"
                    samples.append((img_path, 1))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Cargar imagen
        image = Image.open(img_path).convert('L')  # Convertir a escala de grises
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DataAugmentation:
    """
    Clase para aplicar data augmentation específico para imágenes médicas.
    """
    
    def __init__(self, is_training=True):
        self.is_training = is_training
        
    def __call__(self, image):
        if self.is_training:
            # Horizontal flip (50% probabilidad)
            if torch.rand(1) < 0.5:
                image = TF.hflip(image)
            
            # Rotación (±10 grados, 30% probabilidad)
            if torch.rand(1) < 0.3:
                angle = torch.randint(-10, 11, (1,)).item()
                image = TF.rotate(image, angle)
        
        return image


def get_transforms(is_training=True):
    """
    Obtiene las transformaciones para entrenamiento y validación.
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            DataAugmentation(is_training=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    return transform


def calculate_class_weights(dataset):
    """
    Calcula pesos de clases para CrossEntropyLoss ponderado.
    """
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total_samples = len(dataset.samples)
    num_classes = len(dataset.class_names)
    
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
            weights.append(weight)
        else:
            weights.append(1.0)
    
    return torch.FloatTensor(weights)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena el modelo por una época."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Entrenando")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Actualizar progress bar
        progress_bar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Valida el modelo por una época."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validando")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Actualizar progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_targets


def plot_training_history(history, save_path=None):
    """Grafica el historial de entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['val_acc'], label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Historial guardado en: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Grafica la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")
    
    plt.show()


def main():
    """Función principal de entrenamiento."""
    print("=== ENTRENAMIENTO CNN BÁSICA PARA CLASIFICACIÓN DE LESIONES PULMONARES ===\n")
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Parámetros
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.01  # Learning rate inicial más alto
    weight_decay = 1e-4
    
    # Early stopping
    early_stopping_patience = 5
    early_stopping_min_delta = 0.001
    
    # Cargar datasets
    print("Cargando datasets...")
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    train_dataset = LungDataset(
        DATA_PATH / "train" / "images",
        DATA_PATH / "train" / "labels",
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = LungDataset(
        DATA_PATH / "val" / "images",
        DATA_PATH / "val" / "labels",
        transform=val_transform,
        is_training=False
    )
    
    print(f"Entrenamiento: {len(train_dataset)} muestras")
    print(f"Validación: {len(val_dataset)} muestras")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Modelo
    model = BasicCNN(num_classes=2, dropout_rates=[0.3, 0.4, 0.5])
    model = model.to(device)
    
    # Resumen del modelo
    summary = model.get_model_summary()
    print(f"\nResumen del modelo:")
    print(f"  - Parámetros totales: {summary['total_parameters']:,}")
    print(f"  - Parámetros entrenables: {summary['trainable_parameters']:,}")
    print(f"  - Arquitectura: {summary['architecture']}")
    
    # Calcular pesos de clases
    class_weights = calculate_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    print(f"  - Pesos de clases: {class_weights.cpu().numpy()}")
    
    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Historial de entrenamiento
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = MODEL_SAVE_PATH / "basic_cnn_best.pth"
    
    # Early stopping
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    print(f"\nIniciando entrenamiento por {num_epochs} épocas...")
    print(f"Early stopping: paciencia={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    print(f"DEBUG: num_epochs = {num_epochs}")
    print("=" * 60)
    
    # Loop de entrenamiento
    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Entrenar
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validar
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"✓ Nuevo mejor modelo guardado (Val Acc: {val_acc:.2f}%)")
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0
            print(f"✓ Mejora en validación detectada (nueva mejor pérdida: {val_loss:.4f})")
        else:
            early_stopping_counter += 1
            print(f"⚠ Sin mejora en validación ({early_stopping_counter}/{early_stopping_patience})")
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"\n🛑 EARLY STOPPING ACTIVADO")
                print(f"Sin mejora en {early_stopping_patience} épocas consecutivas")
                print(f"Mejor pérdida de validación: {best_val_loss:.4f}")
                break
        
        # Actualizar learning rate
        scheduler.step()
    
    # Resultados finales
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Mejor precisión de validación: {best_val_acc:.2f}%")
    print(f"Mejor modelo guardado en: {best_model_path}")
    
    # Cargar mejor modelo para evaluación final
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluación final
    print("\nEvaluación final...")
    _, final_acc, final_preds, final_targets = validate_epoch(model, val_loader, criterion, device)
    
    # Reporte de clasificación
    print("\nReporte de clasificación:")
    report = classification_report(final_targets, final_preds, 
                                 target_names=train_dataset.class_names, 
                                 output_dict=True)
    print(classification_report(final_targets, final_preds, 
                              target_names=train_dataset.class_names))
    
    # Guardar métricas detalladas
    import json
    metrics_path = RESULTS_PATH / "basic_cnn_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Métricas detalladas guardadas en: {metrics_path}")
    
    # Visualizaciones
    print("\nGenerando visualizaciones...")
    
    # Historial de entrenamiento
    plot_training_history(history, RESULTS_PATH / "basic_cnn_training_history.png")
    
    # Matriz de confusión
    plot_confusion_matrix(final_targets, final_preds, train_dataset.class_names,
                         RESULTS_PATH / "basic_cnn_confusion_matrix.png")
    
    # Guardar historial
    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_PATH / "basic_cnn_training_history.csv", index=False)
    print(f"Historial guardado en: {RESULTS_PATH / 'basic_cnn_training_history.csv'}")
    
    print("\n¡Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    main()
