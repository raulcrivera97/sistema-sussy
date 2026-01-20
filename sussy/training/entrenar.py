"""
Script CLI para Entrenar Modelos Sussy.

Uso:
    # Prueba rápida (CPU)
    python -m sussy.training.entrenar --preset prueba --dataset mi_dataset
    
    # Desarrollo
    python -m sussy.training.entrenar --preset desarrollo --dataset mi_dataset
    
    # Producción en Lambda
    python -m sussy.training.entrenar --preset lambda --dataset mi_dataset
    
    # Personalizado
    python -m sussy.training.entrenar --epochs 50 --batch 16 --imgsz 640 --dataset mi_dataset

El script detecta automáticamente el hardware y ajusta la configuración.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Configuración de presets
PRESETS = {
    "prueba": {
        "descripcion": "Prueba rápida para verificar pipeline",
        "epochs": 10,
        "batch": 4,
        "imgsz": 320,
        "patience": 5,
        "modelo": "yolo11n.pt",
    },
    "desarrollo": {
        "descripcion": "Balance entre velocidad y calidad",
        "epochs": 50,
        "batch": 16,
        "imgsz": 640,
        "patience": 15,
        "modelo": "yolo11s.pt",
    },
    "produccion": {
        "descripcion": "Entrenamiento serio para producción",
        "epochs": 100,
        "batch": 32,
        "imgsz": 640,
        "patience": 20,
        "modelo": "yolo11m.pt",
    },
    "lambda": {
        "descripcion": "Optimizado para Lambda Labs (A10/A100)",
        "epochs": 200,
        "batch": 32,
        "imgsz": 960,
        "patience": 30,
        "modelo": "yolo11l.pt",
    },
}


def detectar_cuda_funcional() -> tuple:
    """
    Detecta si CUDA está realmente funcional.
    
    Returns:
        (funcional, info_gpu)
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, "CUDA no disponible"
        
        # Test real
        try:
            test = torch.zeros(1, device="cuda")
            del test
            
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            
            return True, f"{gpu_name} ({vram_gb:.1f}GB VRAM)"
            
        except RuntimeError as e:
            # sm_120 u otra incompatibilidad
            return False, f"GPU detectada pero CUDA no funcional: {e}"
            
    except ImportError:
        return False, "PyTorch no instalado"


def ajustar_config_hardware(config: dict, cuda_funcional: bool, gpu_vram_gb: Optional[float] = None) -> dict:
    """Ajusta la configuración según el hardware disponible."""
    config = config.copy()
    
    if not cuda_funcional:
        # CPU mode: reducir agresivamente
        config["batch"] = min(config["batch"], 4)
        config["imgsz"] = min(config["imgsz"], 512)
        config["epochs"] = min(config["epochs"], 30)  # No tiene sentido más en CPU
        print("[!] Usando CPU: configuracion reducida para viabilidad")
    elif gpu_vram_gb:
        # Ajustar batch según VRAM
        if gpu_vram_gb < 6:
            config["batch"] = min(config["batch"], 4)
        elif gpu_vram_gb < 8:
            config["batch"] = min(config["batch"], 8)
        elif gpu_vram_gb < 12:
            config["batch"] = min(config["batch"], 16)
    
    return config


def entrenar(args):
    """Función principal de entrenamiento."""
    
    print("\n" + "=" * 60)
    print("SISTEMA SUSSY - ENTRENAMIENTO")
    print("=" * 60)
    
    # Detectar hardware
    cuda_funcional, info_gpu = detectar_cuda_funcional()
    print(f"\nHardware: {info_gpu}")
    print(f"CUDA funcional: {'[OK]' if cuda_funcional else '[NO]'}")
    
    # Obtener configuración base
    if args.preset:
        if args.preset not in PRESETS:
            print(f"[ERROR] Preset desconocido: {args.preset}")
            print(f"   Disponibles: {list(PRESETS.keys())}")
            return 1
        config = PRESETS[args.preset].copy()
        print(f"\nPreset: {args.preset.upper()}")
        print(f"   {config['descripcion']}")
    else:
        config = PRESETS["desarrollo"].copy()
    
    # Override con argumentos CLI
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch:
        config["batch"] = args.batch
    if args.imgsz:
        config["imgsz"] = args.imgsz
    if args.modelo:
        config["modelo"] = args.modelo
    if args.patience:
        config["patience"] = args.patience
    
    # Ajustar según hardware
    gpu_vram = None
    if cuda_funcional:
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            gpu_vram = props.total_memory / (1024**3)
        except:
            pass
    
    config = ajustar_config_hardware(config, cuda_funcional, gpu_vram)
    
    # Validar dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        # Buscar en ubicaciones comunes
        posibles = [
            Path("datasets") / args.dataset / "data.yaml",
            Path("training_output/datasets") / args.dataset / "data.yaml",
            Path(args.dataset),
        ]
        for p in posibles:
            if p.exists():
                dataset_path = p
                break
        else:
            print(f"[ERROR] Dataset no encontrado: {args.dataset}")
            print("   Busque en:")
            for p in posibles:
                print(f"     - {p}")
            return 1
    
    print(f"\nDataset: {dataset_path}")
    
    # Importar YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics no instalado")
        print("   Ejecuta: pip install ultralytics")
        return 1
    
    # Mostrar configuración final
    print(f"\nConfiguracion final:")
    print(f"   Modelo: {config['modelo']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch: {config['batch']}")
    print(f"   ImgSize: {config['imgsz']}")
    print(f"   Patience: {config['patience']}")
    print(f"   Device: {'cuda' if cuda_funcional else 'cpu'}")
    
    # Confirmar
    if not args.yes:
        respuesta = input("\n¿Continuar? [S/n]: ").strip().lower()
        if respuesta and respuesta != "s":
            print("Cancelado")
            return 0
    
    # Preparar carpeta de salida
    output_dir = Path(args.output) if args.output else Path("training_output/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nombre_proyecto = args.nombre or f"sussy_{int(time.time())}"
    
    # Entrenar
    print(f"\nIniciando entrenamiento...")
    print("=" * 60)
    
    try:
        model = YOLO(config["modelo"])
        
        start_time = time.time()
        
        results = model.train(
            data=str(dataset_path),
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            patience=config["patience"],
            project=str(output_dir),
            name=nombre_proyecto,
            exist_ok=True,
            device=0 if cuda_funcional else "cpu",
            verbose=True,
            amp=cuda_funcional,  # FP16 solo en GPU
            workers=4 if cuda_funcional else 2,
            cache=not cuda_funcional,  # Cache en RAM si CPU (más lento pero menos I/O)
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"[OK] ENTRENAMIENTO COMPLETADO!")
        print(f"   Tiempo: {elapsed/60:.1f} minutos")
        print(f"   Modelo: {output_dir}/{nombre_proyecto}/weights/best.pt")
        
        # Exportar a ONNX si se solicitó
        if args.exportar_onnx:
            print("\nExportando a ONNX...")
            best_path = output_dir / nombre_proyecto / "weights" / "best.pt"
            if best_path.exists():
                model = YOLO(str(best_path))
                onnx_path = model.export(format="onnx", imgsz=config["imgsz"])
                print(f"   [OK] ONNX: {onnx_path}")
        
        # Guardar info del entrenamiento
        info_path = output_dir / nombre_proyecto / "training_info.json"
        with open(info_path, "w") as f:
            json.dump({
                "preset": args.preset,
                "config": config,
                "dataset": str(dataset_path),
                "cuda_funcional": cuda_funcional,
                "info_gpu": info_gpu,
                "duracion_minutos": elapsed / 60,
            }, f, indent=2)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[!] Entrenamiento interrumpido por el usuario")
        return 130
        
    except Exception as e:
        print(f"\n[ERROR] Error durante entrenamiento: {e}")
        print("\nPosibles soluciones:")
        print("   - Reducir --batch")
        print("   - Usar --preset prueba")
        print("   - Verificar que el dataset existe y es válido")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar modelos de detección/clasificación para Sistema Sussy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s --preset prueba --dataset coco128
  %(prog)s --preset desarrollo --dataset mi_dataset --exportar-onnx
  %(prog)s --epochs 100 --batch 16 --imgsz 640 --dataset mi_dataset
  
Presets disponibles:
  prueba      - Prueba rápida (10 epochs, 320px)
  desarrollo  - Balance velocidad/calidad (50 epochs, 640px)
  produccion  - Entrenamiento serio (100 epochs, 640px)
  lambda      - Optimizado para cloud (200 epochs, 960px)
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Ruta al dataset o data.yaml"
    )
    
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        help="Preset de configuración"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        help="Número de épocas (override preset)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=int,
        help="Tamaño de batch (override preset)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Tamaño de imagen (override preset)"
    )
    
    parser.add_argument(
        "--modelo", "-m",
        help="Modelo base a usar (ej: yolo11n.pt, yolo11s.pt)"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        help="Early stopping patience"
    )
    
    parser.add_argument(
        "--nombre", "-n",
        help="Nombre del proyecto/experimento"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Carpeta de salida para modelos"
    )
    
    parser.add_argument(
        "--exportar-onnx",
        action="store_true",
        help="Exportar modelo a ONNX al finalizar"
    )
    
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="No pedir confirmación"
    )
    
    args = parser.parse_args()
    
    # Si no hay preset ni parámetros, usar desarrollo por defecto
    if not args.preset and not any([args.epochs, args.batch, args.imgsz]):
        args.preset = "desarrollo"
    
    sys.exit(entrenar(args))


if __name__ == "__main__":
    main()
