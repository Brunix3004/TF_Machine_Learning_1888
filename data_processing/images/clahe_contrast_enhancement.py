#!/usr/bin/env python3
"""
Script para aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization) 
a las imágenes del dataset y comparar los resultados antes/después.

Autor: Sistema de Procesamiento de Imágenes
Fecha: 2024
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuración de paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "images"
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "processed" / "images_clahe"
PLOTS_PATH = PROJECT_ROOT / "visualization" / "plots" / "img_data_plots"

# Crear directorios si no existen
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Aplica CLAHE a una imagen en escala de grises.
    
    Args:
        image: Imagen en escala de grises (numpy array)
        clip_limit: Límite de contraste (default: 2.0)
        tile_grid_size: Tamaño de la grilla para CLAHE (default: 8x8)
    
    Returns:
        Imagen procesada con CLAHE
    """
    # Crear objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Aplicar CLAHE
    clahe_image = clahe.apply(image)
    
    return clahe_image

def process_single_image(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Procesa una sola imagen aplicando CLAHE.
    
    Args:
        image_path: Ruta de la imagen original
        output_path: Ruta donde guardar la imagen procesada
        clip_limit: Límite de contraste para CLAHE
        tile_grid_size: Tamaño de la grilla para CLAHE
    
    Returns:
        dict: Estadísticas de la imagen original y procesada
    """
    try:
        # Cargar imagen
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Aplicar CLAHE
        clahe_img = apply_clahe(img, clip_limit, tile_grid_size)
        
        # Guardar imagen procesada
        cv2.imwrite(str(output_path), clahe_img)
        
        # Calcular estadísticas
        stats = {
            'original_mean': img.mean(),
            'original_std': img.std(),
            'original_min': img.min(),
            'original_max': img.max(),
            'clahe_mean': clahe_img.mean(),
            'clahe_std': clahe_img.std(),
            'clahe_min': clahe_img.min(),
            'clahe_max': clahe_img.max(),
            'contrast_improvement': clahe_img.std() - img.std(),
            'brightness_change': clahe_img.mean() - img.mean()
        }
        
        return stats
        
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

def process_dataset(input_dir, output_dir, labels_dir=None, output_labels_dir=None, sample_size=None, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Procesa todo el dataset aplicando CLAHE y copia las etiquetas correspondientes.
    
    Args:
        input_dir: Directorio con imágenes originales
        output_dir: Directorio donde guardar imágenes procesadas
        labels_dir: Directorio con etiquetas originales (opcional)
        output_labels_dir: Directorio donde guardar etiquetas (opcional)
        sample_size: Número de imágenes a procesar (None para todas)
        clip_limit: Límite de contraste para CLAHE
        tile_grid_size: Tamaño de la grilla para CLAHE
    
    Returns:
        DataFrame con estadísticas de todas las imágenes
    """
    # Buscar todas las imágenes
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(input_dir).rglob(f'*{ext}'))
        image_paths.extend(Path(input_dir).rglob(f'*{ext.upper()}'))
    
    # Muestrear si se especifica sample_size
    if sample_size and len(image_paths) > sample_size:
        image_paths = np.random.choice(image_paths, sample_size, replace=False)
    
    print(f"Procesando {len(image_paths)} imágenes con CLAHE...")
    
    # Procesar imágenes
    all_stats = []
    
    for img_path in tqdm(image_paths, desc="Aplicando CLAHE"):
        # Crear estructura de directorios para esta imagen específica
        rel_path = img_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Procesar imagen
        stats = process_single_image(img_path, output_path, clip_limit, tile_grid_size)
        if stats:
            stats['image_path'] = str(rel_path)
            all_stats.append(stats)
            
            # Copiar etiqueta correspondiente si existe
            if labels_dir and output_labels_dir:
                label_rel_path = rel_path.with_suffix('.txt')
                label_input_path = labels_dir / label_rel_path
                label_output_path = output_labels_dir / label_rel_path
                
                if label_input_path.exists():
                    # Crear directorio de destino si no existe
                    label_output_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(label_input_path, label_output_path)
    
    return pd.DataFrame(all_stats)

def create_comparison_visualization(stats_df, sample_images=None, output_path=None):
    """
    Crea visualizaciones comparando antes y después de CLAHE.
    
    Args:
        stats_df: DataFrame con estadísticas de las imágenes
        sample_images: Lista de rutas de imágenes de ejemplo para mostrar
        output_path: Ruta donde guardar las visualizaciones
    """
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Mejora de contraste
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(stats_df['contrast_improvement'], bins=30, alpha=0.7, color='green')
    plt.title('Mejora de Contraste (Std CLAHE - Std Original)')
    plt.xlabel('Mejora de Desviación Estándar')
    plt.ylabel('Frecuencia')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    # 2. Cambio de brillo
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(stats_df['brightness_change'], bins=30, alpha=0.7, color='orange')
    plt.title('Cambio de Brillo (Media CLAHE - Media Original)')
    plt.xlabel('Cambio de Media')
    plt.ylabel('Frecuencia')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    # 3. Box plot comparativo
    ax3 = plt.subplot(2, 3, 3)
    data_to_plot = [stats_df['original_std'], stats_df['clahe_std']]
    plt.boxplot(data_to_plot, labels=['Original', 'CLAHE'])
    plt.title('Comparación de Desviación Estándar')
    plt.ylabel('Desviación Estándar')
    plt.grid(True, alpha=0.3)
    
    # 4-5. Imagen de ejemplo: Original vs CLAHE
    if sample_images:
        # Tomar la primera imagen como ejemplo
        img_path = sample_images[0]
        
        try:
            # Cargar imagen original
            original_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            # Aplicar CLAHE
            clahe_img = apply_clahe(original_img)
            
            # Mostrar imagen original
            ax4 = plt.subplot(2, 3, 4)
            plt.imshow(original_img, cmap='gray')
            plt.title(f'Original - {Path(img_path).name}')
            plt.axis('off')
            
            # Mostrar imagen CLAHE
            ax5 = plt.subplot(2, 3, 5)
            plt.imshow(clahe_img, cmap='gray')
            plt.title(f'CLAHE - {Path(img_path).name}')
            plt.axis('off')
            
        except Exception as e:
            ax4 = plt.subplot(2, 3, 4)
            plt.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax4.transAxes)
            plt.title('Error cargando imagen')
            
            ax5 = plt.subplot(2, 3, 5)
            plt.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax5.transAxes)
            plt.title('Error cargando imagen')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualización guardada en: {output_path}")
    
    plt.show()

def main():
    """
    Función principal para ejecutar el procesamiento CLAHE.
    """
    print("=== PROCESAMIENTO CLAHE PARA MEJORA DE CONTRASTE ===\n")
    
    # Configuración
    train_input = DATA_PATH / "train" / "images"
    val_input = DATA_PATH / "val" / "images"
    
    train_labels = DATA_PATH / "train" / "labels"
    val_labels = DATA_PATH / "val" / "labels"
    
    train_output = OUTPUT_PATH / "train" / "images"
    val_output = OUTPUT_PATH / "val" / "images"
    
    train_output_labels = OUTPUT_PATH / "train" / "labels"
    val_output_labels = OUTPUT_PATH / "val" / "labels"
    
    # Parámetros CLAHE
    clip_limit = 2.0
    tile_grid_size = (8, 8)
    sample_size = None  # Procesar TODAS las imágenes
    
    print(f"Configuración CLAHE:")
    print(f"  - Clip Limit: {clip_limit}")
    print(f"  - Tile Grid Size: {tile_grid_size}")
    print(f"  - Sample Size: {'TODAS las imágenes' if sample_size is None else sample_size}")
    print()
    
    # Procesar conjunto de entrenamiento
    if train_input.exists():
        print("Procesando conjunto de entrenamiento...")
        train_stats = process_dataset(
            train_input, train_output,
            labels_dir=train_labels if train_labels.exists() else None,
            output_labels_dir=train_output_labels if train_labels.exists() else None,
            sample_size=sample_size,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size
        )
        print(f"✓ Procesadas {len(train_stats)} imágenes de entrenamiento")
        if train_labels.exists():
            print(f"✓ Copiadas etiquetas de entrenamiento")
    else:
        print("⚠ No se encontró directorio de entrenamiento")
        train_stats = pd.DataFrame()
    
    # Procesar conjunto de validación
    if val_input.exists():
        print("Procesando conjunto de validación...")
        val_stats = process_dataset(
            val_input, val_output,
            labels_dir=val_labels if val_labels.exists() else None,
            output_labels_dir=val_output_labels if val_labels.exists() else None,
            sample_size=sample_size,  # Procesar todas las imágenes de validación también
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size
        )
        print(f"✓ Procesadas {len(val_stats)} imágenes de validación")
        if val_labels.exists():
            print(f"✓ Copiadas etiquetas de validación")
    else:
        print("⚠ No se encontró directorio de validación")
        val_stats = pd.DataFrame()
    
    # Combinar estadísticas
    if not train_stats.empty and not val_stats.empty:
        all_stats = pd.concat([train_stats, val_stats], ignore_index=True)
    elif not train_stats.empty:
        all_stats = train_stats
    elif not val_stats.empty:
        all_stats = val_stats
    else:
        print("❌ No se procesaron imágenes")
        return
    
    # Guardar estadísticas
    stats_path = PLOTS_PATH / "clahe_statistics.csv"
    all_stats.to_csv(stats_path, index=False)
    print(f"✓ Estadísticas guardadas en: {stats_path}")
    
    # Crear archivo dataset.yaml para el dataset CLAHE
    dataset_yaml_path = OUTPUT_PATH / "dataset_clahe.yaml"
    with open(dataset_yaml_path, 'w') as f:
        f.write("names:\n")
        f.write("- Nodule/Mass\n")
        f.write("- Other lesion\n")
        f.write("nc: 2\n")
        f.write(f"path: {OUTPUT_PATH}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
    print(f"✓ Archivo dataset.yaml creado en: {dataset_yaml_path}")
    
    # Crear visualizaciones
    print("\nCreando visualizaciones...")
    
    # Seleccionar algunas imágenes de ejemplo
    sample_images = []
    if train_input.exists():
        sample_images.extend(list(train_input.glob("*.jpg"))[:3])
    if val_input.exists():
        sample_images.extend(list(val_input.glob("*.jpg"))[:3])
    
    # Crear visualización
    viz_path = PLOTS_PATH / "clahe_comparison_analysis.png"
    create_comparison_visualization(
        all_stats, 
        sample_images=sample_images,
        output_path=viz_path
    )
    
    print(f"\n=== RESUMEN ===")
    print(f"Total de imágenes procesadas: {len(all_stats)}")
    print(f"Mejora promedio de contraste: {all_stats['contrast_improvement'].mean():.2f}")
    print(f"Porcentaje de imágenes con mejora: {(all_stats['contrast_improvement'] > 0).mean()*100:.1f}%")
    print(f"Imágenes procesadas guardadas en: {OUTPUT_PATH}")
    print(f"Visualizaciones guardadas en: {PLOTS_PATH}")

if __name__ == "__main__":
    main()
