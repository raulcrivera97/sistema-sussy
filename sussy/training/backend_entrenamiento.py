"""
Backend de Entrenamiento Flexible.

Soporta múltiples backends de entrenamiento:
- Local CPU (lento pero funciona siempre)
- Local GPU via ONNX Runtime + DirectML (RTX 5080 sm_120)
- Cloud (Lambda Labs, Google Colab, etc.)

El sistema detecta automáticamente el mejor backend disponible
y adapta la configuración para cada entorno.
"""

import json
import logging
import os
import platform
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("sussy.training.backend")


class TipoBackend(Enum):
    """Tipos de backend de entrenamiento disponibles."""
    CPU = auto()
    CUDA = auto()           # PyTorch CUDA (GPUs compatibles)
    DIRECTML = auto()       # ONNX Runtime DirectML (RTX 5080, AMD)
    MPS = auto()            # Apple Silicon
    LAMBDA = auto()         # Lambda Labs Cloud
    COLAB = auto()          # Google Colab
    KAGGLE = auto()         # Kaggle Notebooks


@dataclass
class InfoHardware:
    """Información del hardware disponible."""
    cpu_cores: int = 1
    ram_gb: float = 4.0
    gpu_nombre: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_compute_capability: Optional[str] = None  # e.g., "sm_120"
    cuda_disponible: bool = False
    directml_disponible: bool = False
    mps_disponible: bool = False
    en_colab: bool = False
    en_kaggle: bool = False


@dataclass
class ConfigBackend:
    """Configuración del backend de entrenamiento."""
    tipo: TipoBackend
    device: str  # "cpu", "cuda:0", "directml", etc.
    
    # Límites según hardware
    batch_size_max: int = 16
    workers_max: int = 4
    img_size_max: int = 640
    
    # Optimizaciones
    usar_fp16: bool = False
    usar_gradient_checkpointing: bool = False
    acumular_gradientes: int = 1
    
    # Paths
    path_cache: Optional[Path] = None
    
    def to_dict(self) -> Dict:
        return {
            "tipo": self.tipo.name,
            "device": self.device,
            "batch_size_max": self.batch_size_max,
            "workers_max": self.workers_max,
            "img_size_max": self.img_size_max,
            "usar_fp16": self.usar_fp16,
        }


class DetectorHardware:
    """Detecta el hardware disponible y sus capacidades."""
    
    @staticmethod
    def detectar() -> InfoHardware:
        """Detecta el hardware del sistema."""
        info = InfoHardware()
        
        # CPU
        try:
            import psutil
            info.cpu_cores = psutil.cpu_count(logical=False) or 1
            info.ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            info.cpu_cores = os.cpu_count() or 1
        
        # PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                # Obtener info de GPU primero (esto no requiere que CUDA funcione)
                try:
                    info.gpu_nombre = torch.cuda.get_device_name(0)
                    props = torch.cuda.get_device_properties(0)
                    info.gpu_vram_gb = props.total_memory / (1024**3)
                    info.gpu_compute_capability = f"sm_{props.major}{props.minor}0"
                except:
                    pass
                
                # Test real de CUDA - crear tensor Y forzar operación
                try:
                    test_tensor = torch.zeros(10, device="cuda")
                    # Forzar sincronización para detectar errores
                    result = test_tensor.sum().item()
                    del test_tensor
                    torch.cuda.synchronize()
                    
                    info.cuda_disponible = True
                except RuntimeError as e:
                    # CUDA disponible pero no funciona (ej: sm_120)
                    LOGGER.warning(f"CUDA detectado pero no funcional: {e}")
                    info.cuda_disponible = False
            
            # MPS (Apple)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info.mps_disponible = True
        except ImportError:
            pass
        
        # DirectML (Windows)
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "DmlExecutionProvider" in providers:
                info.directml_disponible = True
                LOGGER.info("DirectML disponible para inferencia")
        except ImportError:
            pass
        
        # Detectar entorno cloud
        info.en_colab = "COLAB_GPU" in os.environ or "google.colab" in sys.modules
        info.en_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        
        return info
    
    @staticmethod
    def imprimir_info(info: InfoHardware):
        """Imprime información del hardware."""
        print("\n" + "=" * 50)
        print("INFORMACION DE HARDWARE")
        print("=" * 50)
        print(f"CPU: {info.cpu_cores} cores")
        print(f"RAM: {info.ram_gb:.1f} GB")
        
        if info.gpu_nombre:
            print(f"GPU: {info.gpu_nombre}")
            if info.gpu_vram_gb:
                print(f"VRAM: {info.gpu_vram_gb:.1f} GB")
            if info.gpu_compute_capability:
                print(f"Compute Capability: {info.gpu_compute_capability}")
        
        print(f"\nCUDA PyTorch: {'[OK]' if info.cuda_disponible else '[NO]'}")
        print(f"DirectML: {'[OK]' if info.directml_disponible else '[NO]'}")
        print(f"MPS (Apple): {'[OK]' if info.mps_disponible else '[NO]'}")
        
        if info.en_colab:
            print("\n[CLOUD] Ejecutando en Google Colab")
        if info.en_kaggle:
            print("\n[CLOUD] Ejecutando en Kaggle")


class SelectorBackend:
    """Selecciona el mejor backend según el hardware disponible."""
    
    @staticmethod
    def seleccionar(info: InfoHardware, preferencia: Optional[TipoBackend] = None) -> ConfigBackend:
        """
        Selecciona el mejor backend de entrenamiento.
        
        Args:
            info: Información del hardware
            preferencia: Backend preferido (si está disponible)
        
        Returns:
            Configuración del backend seleccionado
        """
        # Si hay preferencia y está disponible, usarla
        if preferencia:
            config = SelectorBackend._intentar_backend(preferencia, info)
            if config:
                return config
        
        # Orden de preferencia automático
        orden = [
            TipoBackend.COLAB,
            TipoBackend.KAGGLE,
            TipoBackend.CUDA,
            TipoBackend.MPS,
            TipoBackend.DIRECTML,
            TipoBackend.CPU,
        ]
        
        for tipo in orden:
            config = SelectorBackend._intentar_backend(tipo, info)
            if config:
                LOGGER.info(f"Backend seleccionado: {tipo.name}")
                return config
        
        # Fallback a CPU siempre disponible
        return SelectorBackend._config_cpu(info)
    
    @staticmethod
    def _intentar_backend(tipo: TipoBackend, info: InfoHardware) -> Optional[ConfigBackend]:
        """Intenta crear configuración para un tipo de backend."""
        if tipo == TipoBackend.COLAB and info.en_colab:
            return SelectorBackend._config_colab(info)
        
        if tipo == TipoBackend.KAGGLE and info.en_kaggle:
            return SelectorBackend._config_kaggle(info)
        
        if tipo == TipoBackend.CUDA and info.cuda_disponible:
            return SelectorBackend._config_cuda(info)
        
        if tipo == TipoBackend.MPS and info.mps_disponible:
            return SelectorBackend._config_mps(info)
        
        if tipo == TipoBackend.DIRECTML and info.directml_disponible:
            return SelectorBackend._config_directml(info)
        
        if tipo == TipoBackend.CPU:
            return SelectorBackend._config_cpu(info)
        
        return None
    
    @staticmethod
    def _config_cuda(info: InfoHardware) -> ConfigBackend:
        """Configuración para CUDA."""
        vram = info.gpu_vram_gb or 4.0
        
        # Ajustar batch según VRAM
        if vram >= 16:
            batch = 32
            img_size = 960
        elif vram >= 8:
            batch = 16
            img_size = 640
        else:
            batch = 8
            img_size = 512
        
        return ConfigBackend(
            tipo=TipoBackend.CUDA,
            device="cuda:0",
            batch_size_max=batch,
            workers_max=min(8, info.cpu_cores),
            img_size_max=img_size,
            usar_fp16=True,
        )
    
    @staticmethod
    def _config_mps(info: InfoHardware) -> ConfigBackend:
        """Configuración para Apple MPS."""
        return ConfigBackend(
            tipo=TipoBackend.MPS,
            device="mps",
            batch_size_max=16,
            workers_max=min(4, info.cpu_cores),
            img_size_max=640,
            usar_fp16=False,  # MPS tiene issues con FP16
        )
    
    @staticmethod
    def _config_directml(info: InfoHardware) -> ConfigBackend:
        """
        Configuración para DirectML.
        
        NOTA: DirectML funciona para INFERENCIA pero el entrenamiento
        con PyTorch/YOLO no está soportado directamente.
        Para entrenar con RTX 5080 (sm_120), opciones:
        1. Usar CPU para entrenamiento
        2. Usar cloud (Lambda, Colab)
        3. Esperar soporte PyTorch para sm_120
        """
        LOGGER.warning(
            "DirectML detectado pero YOLO no soporta entrenamiento con DirectML. "
            "Usando CPU para entrenamiento. Para mejor rendimiento, usa cloud."
        )
        
        # Volver a CPU pero con nota de que DirectML está disponible para inferencia
        config = SelectorBackend._config_cpu(info)
        config.tipo = TipoBackend.DIRECTML  # Marcar para referencia
        
        return config
    
    @staticmethod
    def _config_cpu(info: InfoHardware) -> ConfigBackend:
        """Configuración para CPU."""
        # Ajustar según RAM disponible
        if info.ram_gb >= 32:
            batch = 8
            img_size = 640
        elif info.ram_gb >= 16:
            batch = 4
            img_size = 512
        else:
            batch = 2
            img_size = 416
        
        return ConfigBackend(
            tipo=TipoBackend.CPU,
            device="cpu",
            batch_size_max=batch,
            workers_max=min(4, info.cpu_cores),
            img_size_max=img_size,
            usar_fp16=False,
            usar_gradient_checkpointing=True,  # Ahorra memoria
            acumular_gradientes=4,  # Simula batch mayor
        )
    
    @staticmethod
    def _config_colab(info: InfoHardware) -> ConfigBackend:
        """Configuración para Google Colab."""
        # Colab tiene T4 (16GB) o A100 (40GB)
        return ConfigBackend(
            tipo=TipoBackend.COLAB,
            device="cuda:0",
            batch_size_max=32,
            workers_max=2,  # Colab tiene límites
            img_size_max=960,
            usar_fp16=True,
        )
    
    @staticmethod
    def _config_kaggle(info: InfoHardware) -> ConfigBackend:
        """Configuración para Kaggle."""
        return ConfigBackend(
            tipo=TipoBackend.KAGGLE,
            device="cuda:0",
            batch_size_max=16,
            workers_max=2,
            img_size_max=640,
            usar_fp16=True,
        )


class EntrenadorAdaptativo:
    """
    Entrenador que se adapta al backend disponible.
    
    Proporciona una interfaz unificada para entrenar en cualquier entorno.
    """
    
    def __init__(self, path_proyecto: Path):
        self.path_proyecto = Path(path_proyecto)
        self.info_hardware = DetectorHardware.detectar()
        self.config_backend = SelectorBackend.seleccionar(self.info_hardware)
        
        self.path_proyecto.mkdir(parents=True, exist_ok=True)
        
        # Guardar info de hardware
        self._guardar_info_hardware()
    
    def _guardar_info_hardware(self):
        """Guarda información del hardware detectado."""
        info = {
            "cpu_cores": self.info_hardware.cpu_cores,
            "ram_gb": self.info_hardware.ram_gb,
            "gpu_nombre": self.info_hardware.gpu_nombre,
            "gpu_vram_gb": self.info_hardware.gpu_vram_gb,
            "gpu_compute_capability": self.info_hardware.gpu_compute_capability,
            "cuda_disponible": self.info_hardware.cuda_disponible,
            "directml_disponible": self.info_hardware.directml_disponible,
            "backend_seleccionado": self.config_backend.tipo.name,
            "device": self.config_backend.device,
        }
        
        path_info = self.path_proyecto / "hardware_info.json"
        with open(path_info, "w") as f:
            json.dump(info, f, indent=2)
    
    def obtener_args_yolo(self, epochs: int = 100, img_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Obtiene argumentos optimizados para entrenar YOLO.
        
        Args:
            epochs: Número de épocas
            img_size: Tamaño de imagen (None = automático)
        
        Returns:
            Dict de argumentos para model.train()
        """
        cfg = self.config_backend
        
        args = {
            "epochs": epochs,
            "imgsz": img_size or cfg.img_size_max,
            "batch": cfg.batch_size_max,
            "device": cfg.device if cfg.tipo != TipoBackend.DIRECTML else "cpu",
            "workers": cfg.workers_max,
            "patience": 20,
            "exist_ok": True,
            "verbose": True,
        }
        
        # Optimizaciones según backend
        if cfg.usar_fp16 and cfg.tipo in [TipoBackend.CUDA, TipoBackend.COLAB, TipoBackend.KAGGLE]:
            args["amp"] = True  # Automatic Mixed Precision
        
        if cfg.tipo == TipoBackend.CPU:
            # Optimizaciones para CPU
            args["cache"] = True  # Cachear imágenes en RAM
            args["patience"] = 30  # Más paciencia porque es más lento
            
            LOGGER.info(
                f"Entrenando en CPU. Batch={cfg.batch_size_max}, "
                f"ImgSize={args['imgsz']}. Esto será lento."
            )
        
        return args
    
    def entrenar_yolo(
        self,
        path_data_yaml: Path,
        modelo_base: str = "yolo11n.pt",
        epochs: int = 100,
        nombre: str = "sussy_model",
        **kwargs,
    ) -> Optional[Path]:
        """
        Entrena un modelo YOLO con la configuración óptima.
        
        Args:
            path_data_yaml: Path al archivo data.yaml
            modelo_base: Modelo base a usar
            epochs: Número de épocas
            nombre: Nombre del experimento
            **kwargs: Argumentos adicionales
        
        Returns:
            Path al mejor modelo entrenado, o None si falló
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            LOGGER.error("Ultralytics no instalado. pip install ultralytics")
            return None
        
        # Obtener argumentos optimizados
        args = self.obtener_args_yolo(epochs)
        args["data"] = str(path_data_yaml)
        args["project"] = str(self.path_proyecto / "runs")
        args["name"] = nombre
        
        # Merge con kwargs del usuario
        args.update(kwargs)
        
        LOGGER.info(f"Iniciando entrenamiento: {nombre}")
        LOGGER.info(f"Backend: {self.config_backend.tipo.name}")
        LOGGER.info(f"Device: {args['device']}")
        LOGGER.info(f"Batch: {args['batch']}, ImgSize: {args['imgsz']}")
        
        try:
            modelo = YOLO(modelo_base)
            results = modelo.train(**args)
            
            # Path al mejor modelo
            path_best = self.path_proyecto / "runs" / nombre / "weights" / "best.pt"
            
            if path_best.exists():
                LOGGER.info(f"Modelo guardado en: {path_best}")
                return path_best
            else:
                LOGGER.warning("No se encontró best.pt")
                return None
                
        except Exception as e:
            LOGGER.error(f"Error en entrenamiento: {e}")
            return None
    
    def exportar_a_onnx(self, path_modelo: Path, img_size: int = 640) -> Optional[Path]:
        """
        Exporta un modelo a ONNX para inferencia con DirectML.
        
        Útil para usar la RTX 5080 en inferencia aunque no soporte entrenamiento.
        """
        try:
            from ultralytics import YOLO
            
            modelo = YOLO(path_modelo)
            path_onnx = modelo.export(format="onnx", imgsz=img_size)
            
            LOGGER.info(f"Modelo exportado a ONNX: {path_onnx}")
            return Path(path_onnx)
            
        except Exception as e:
            LOGGER.error(f"Error exportando a ONNX: {e}")
            return None
    
    def imprimir_info(self):
        """Imprime información del sistema y backend seleccionado."""
        DetectorHardware.imprimir_info(self.info_hardware)
        
        print(f"\n{'=' * 50}")
        print("BACKEND DE ENTRENAMIENTO")
        print(f"{'=' * 50}")
        print(f"Tipo: {self.config_backend.tipo.name}")
        print(f"Device: {self.config_backend.device}")
        print(f"Batch máximo: {self.config_backend.batch_size_max}")
        print(f"Imagen máxima: {self.config_backend.img_size_max}px")
        print(f"Workers: {self.config_backend.workers_max}")
        
        if self.config_backend.tipo == TipoBackend.DIRECTML:
            print("\n[!] NOTA: Tu RTX 5080 (sm_120) no soporta PyTorch CUDA.")
            print("   El entrenamiento usará CPU.")
            print("   Para mejor rendimiento, usa Lambda Labs o Google Colab.")
            print("   La GPU SÍ funcionará para INFERENCIA con ONNX Runtime.")


def generar_script_lambda(
    path_salida: Path,
    path_dataset: str,
    modelo: str = "yolo11m.pt",
    epochs: int = 100,
) -> Path:
    """
    Genera un script listo para ejecutar en Lambda Labs.
    
    Args:
        path_salida: Donde guardar el script
        path_dataset: Path al dataset en Lambda (o URL)
        modelo: Modelo base
        epochs: Épocas de entrenamiento
    
    Returns:
        Path al script generado
    """
    script = f'''#!/bin/bash
# Script de entrenamiento para Lambda Labs
# Generado por Sistema Sussy

# Instalar dependencias
pip install ultralytics

# Configurar dataset
DATASET_PATH="{path_dataset}"

# Entrenar modelo
python -c "
from ultralytics import YOLO
import torch

print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')

model = YOLO('{modelo}')
results = model.train(
    data='$DATASET_PATH/data.yaml',
    epochs={epochs},
    imgsz=960,
    batch=32,
    device='cuda:0',
    workers=8,
    amp=True,
    project='sussy_training',
    name='lambda_run',
)

# Exportar a ONNX para usar en local con DirectML
model.export(format='onnx', imgsz=960)
print('Modelo exportado a ONNX')
"

echo "Entrenamiento completado!"
echo "Descarga los modelos de: sussy_training/lambda_run/weights/"
'''
    
    path_salida = Path(path_salida)
    path_salida.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path_salida, "w") as f:
        f.write(script)
    
    # Hacer ejecutable en Linux
    os.chmod(path_salida, 0o755)
    
    return path_salida


if __name__ == "__main__":
    # Detectar y mostrar info
    info = DetectorHardware.detectar()
    DetectorHardware.imprimir_info(info)
    
    # Seleccionar backend
    config = SelectorBackend.seleccionar(info)
    
    print(f"\nBackend seleccionado: {config.tipo.name}")
    print(f"Device: {config.device}")
    print(f"Batch máximo: {config.batch_size_max}")
