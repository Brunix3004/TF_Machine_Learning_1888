#!/usr/bin/env python3
"""
Script de ejecución para el experimento de CNN Básica
Hito 2: Clasificación de Lesiones Pulmonares

Este script ejecuta el entrenamiento completo de la CNN básica
con todas las configuraciones especificadas.

Uso:
    python run_experiment.py [--config config.yaml] [--gpu 0]

Autor: Sistema de Machine Learning
Fecha: 2024
"""

import argparse
import sys
import yaml
from pathlib import Path
import torch
import logging
from datetime import datetime

# Importar el modelo principal
from basic_cnn_model import main as train_main

def setup_logging(config):
    """Configura el sistema de logging."""
    log_config = config.get('logging', {})
    
    # Resolver path absoluto para logs
    log_dir = Path(config['save']['log_dir']).resolve()
    log_file = log_dir / config['save'].get('log_file', 'experiment.log')
    
    # Crear directorio si no existe
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path):
    """Carga la configuración desde archivo YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config):
    """Crea los directorios necesarios."""
    directories = [
        config['save']['model_dir'],
        config['save']['results_dir'],
        config['save']['log_dir'],
        Path(config['save']['results_dir']) / config['save'].get('plots_dir', 'plots')
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def print_experiment_info(config, logger):
    """Imprime información del experimento."""
    exp_info = config['experiment']
    
    logger.info("=" * 80)
    logger.info(f"EXPERIMENTO: {exp_info['name']}")
    logger.info(f"Descripción: {exp_info['description']}")
    logger.info(f"Versión: {exp_info['version']}")
    logger.info(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Información del modelo
    model_info = config['model']
    logger.info(f"Modelo: {model_info['name']}")
    logger.info(f"Clases: {model_info['num_classes']} - {model_info['class_names']}")
    logger.info(f"Tamaño de entrada: {model_info['architecture']['input_size']}")
    
    # Información de entrenamiento
    train_info = config['training']
    logger.info(f"Batch size: {train_info['batch_size']}")
    logger.info(f"Épocas: {train_info['num_epochs']}")
    logger.info(f"Learning rate: {train_info['learning_rate']}")
    logger.info(f"Optimizador: {train_info['optimizer']['type']}")
    
    # Información de hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    logger.info("=" * 80)

def validate_config(config):
    """Valida la configuración del experimento."""
    required_sections = ['model', 'data', 'training', 'save', 'experiment']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Sección requerida '{section}' no encontrada en la configuración")
    
    # Validar paths
    data_path = Path(config['data']['dataset_path'])
    if not data_path.exists():
        raise ValueError(f"Dataset path no existe: {data_path}")
    
    # Validar configuración del modelo
    if config['model']['num_classes'] != 2:
        raise ValueError("El modelo debe tener exactamente 2 clases")
    
    # Validar configuración de entrenamiento
    if config['training']['batch_size'] <= 0:
        raise ValueError("Batch size debe ser mayor que 0")
    
    if config['training']['num_epochs'] <= 0:
        raise ValueError("Número de épocas debe ser mayor que 0")
    
    if config['training']['learning_rate'] <= 0:
        raise ValueError("Learning rate debe ser mayor que 0")

def main():
    """Función principal del script de ejecución."""
    parser = argparse.ArgumentParser(description='Ejecutar experimento CNN Básica')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Archivo de configuración (default: config.yaml)')
    parser.add_argument('--gpu', type=int, default=None,
                       help='ID de GPU específica a usar')
    parser.add_argument('--dry-run', action='store_true',
                       help='Ejecutar validación sin entrenar')
    parser.add_argument('--verbose', action='store_true',
                       help='Modo verbose')
    
    args = parser.parse_args()
    
    # Inicializar logger básico para manejo de errores
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Cargar configuración
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
        
        config = load_config(config_path)
        
        # Configurar logging completo
        logger = setup_logging(config)
        
        # Configurar GPU si se especifica
        if args.gpu is not None:
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu)
                logger.info(f"Usando GPU {args.gpu}")
            else:
                logger.warning("CUDA no disponible, usando CPU")
        
        # Crear directorios
        setup_directories(config)
        
        # Validar configuración
        validate_config(config)
        logger.info("✓ Configuración validada correctamente")
        
        # Imprimir información del experimento
        print_experiment_info(config, logger)
        
        if args.dry_run:
            logger.info("Modo dry-run: Validación completada, no se ejecutará entrenamiento")
            return 0
        
        # Ejecutar entrenamiento
        logger.info("Iniciando entrenamiento...")
        start_time = datetime.now()
        
        # El entrenamiento se ejecuta desde basic_cnn_model.py
        train_main()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("EXPERIMENTO COMPLETADO EXITOSAMENTE")
        logger.info(f"Duración total: {duration}")
        logger.info(f"Resultados guardados en: {config['save']['results_dir']}")
        logger.info(f"Modelo guardado en: {config['save']['model_dir']}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
