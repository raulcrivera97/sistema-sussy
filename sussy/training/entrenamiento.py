"""
Sistema de Entrenamiento Jerárquico para Sussy.

Estrategia de entrenamiento en cascada:

1. FASE 1 - Detección Base (Categorías):
   - Entrena detector para categorías principales
   - Clases: Persona, Animal, VehiculoAereo, VehiculoTerrestre, VehiculoTerrestreMilitar
   - Modelo: YOLO11 (detección de objetos)

2. FASE 2 - Clasificación Detallada (Subcategorías):
   - Por cada categoría, entrena clasificador de subcategorías
   - Ejemplo: VehiculoAereo → Helicoptero, Avion, Dron
   - Modelo: Clasificador sobre crops del detector

3. FASE 3 - Subtipos Específicos:
   - Clasificación fina dentro de subcategorías
   - Ejemplo: Dron → cuadricoptero, hexacoptero, ala_fija, vtol
   - Modelo: Clasificador especializado

4. FASE 4 - Atributos (Multi-label):
   - Predicción de atributos para cada detección
   - Ejemplo: rol=militar, carga=True, tipo_carga=camara
   - Modelo: Red multi-head o clasificadores independientes

El sistema está diseñado para:
- Usar ONNX Runtime cuando PyTorch CUDA no esté disponible
- Funcionar en CPU si es necesario
- Permitir entrenamiento incremental
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto

import yaml

from sussy.training.taxonomia import (
    GestorTaxonomia,
    obtener_taxonomia,
    ClaseDeteccion,
)
from sussy.training.dataset import GestorDataset, ConfiguracionDataset

LOGGER = logging.getLogger("sussy.training")


class FaseEntrenamiento(Enum):
    """Fases del entrenamiento jerárquico."""
    DETECCION_BASE = auto()
    CLASIFICACION_SUBCATEGORIA = auto()
    CLASIFICACION_SUBTIPO = auto()
    PREDICCION_ATRIBUTOS = auto()


@dataclass
class ConfiguracionEntrenamiento:
    """Configuración para una sesión de entrenamiento."""
    nombre: str
    path_dataset: Path
    path_salida: Path
    
    # Fase a entrenar
    fase: FaseEntrenamiento = FaseEntrenamiento.DETECCION_BASE
    categoria_objetivo: Optional[str] = None  # Para fases 2+
    
    # Hiperparámetros
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 0.01
    
    # Modelo base
    modelo_base: str = "yolo11n.pt"  # n/s/m/l/x
    
    # Opciones de entrenamiento
    usar_pretrained: bool = True
    freeze_backbone: bool = False
    augmentation: bool = True
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    workers: int = 4
    
    # Early stopping
    patience: int = 20
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            "nombre": self.nombre,
            "path_dataset": str(self.path_dataset),
            "path_salida": str(self.path_salida),
            "fase": self.fase.name,
            "categoria_objetivo": self.categoria_objetivo,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "img_size": self.img_size,
            "learning_rate": self.learning_rate,
            "modelo_base": self.modelo_base,
            "usar_pretrained": self.usar_pretrained,
            "freeze_backbone": self.freeze_backbone,
            "augmentation": self.augmentation,
            "device": self.device,
            "workers": self.workers,
            "patience": self.patience,
        }


@dataclass
class ResultadoEntrenamiento:
    """Resultado de una sesión de entrenamiento."""
    exito: bool
    path_modelo: Optional[Path] = None
    metricas: Dict[str, float] = field(default_factory=dict)
    duracion_segundos: float = 0.0
    errores: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "exito": self.exito,
            "path_modelo": str(self.path_modelo) if self.path_modelo else None,
            "metricas": self.metricas,
            "duracion_segundos": self.duracion_segundos,
            "errores": self.errores,
        }


class EntrenadorJerarquico:
    """
    Gestiona el entrenamiento jerárquico de modelos.
    
    Ejemplo de uso:
    
        entrenador = EntrenadorJerarquico(path_proyecto)
        
        # Fase 1: Detector base
        resultado = entrenador.entrenar_fase1_deteccion()
        
        # Fase 2: Clasificadores por categoría
        for categoria in ["VehiculoAereo", "Persona"]:
            resultado = entrenador.entrenar_fase2_clasificacion(categoria)
        
        # Fase 3: Subtipos
        resultado = entrenador.entrenar_fase3_subtipos("VehiculoAereo", "Dron")
    """
    
    def __init__(self, path_proyecto: Path):
        self.path_proyecto = Path(path_proyecto)
        self.taxonomia = obtener_taxonomia()
        
        # Estructura de directorios
        self.path_datasets = self.path_proyecto / "datasets"
        self.path_modelos = self.path_proyecto / "models"
        self.path_logs = self.path_proyecto / "logs"
        self.path_configs = self.path_proyecto / "configs"
        
        # Crear directorios
        for path in [self.path_datasets, self.path_modelos, self.path_logs, self.path_configs]:
            path.mkdir(parents=True, exist_ok=True)
        
        self._verificar_dependencias()
    
    def _verificar_dependencias(self):
        """Verifica que las dependencias necesarias estén disponibles."""
        self.ultralytics_disponible = False
        self.torch_cuda_disponible = False
        self.onnx_disponible = False
        
        try:
            from ultralytics import YOLO
            self.ultralytics_disponible = True
            LOGGER.info("Ultralytics YOLO disponible")
        except ImportError:
            LOGGER.warning("Ultralytics no disponible - pip install ultralytics")
        
        try:
            import torch
            self.torch_cuda_disponible = torch.cuda.is_available()
            if self.torch_cuda_disponible:
                device_name = torch.cuda.get_device_name(0)
                LOGGER.info(f"CUDA disponible: {device_name}")
            else:
                LOGGER.info("CUDA no disponible, usando CPU")
        except ImportError:
            LOGGER.warning("PyTorch no disponible")
        
        try:
            import onnxruntime
            self.onnx_disponible = True
            providers = onnxruntime.get_available_providers()
            LOGGER.info(f"ONNX Runtime disponible: {providers}")
        except ImportError:
            LOGGER.warning("ONNX Runtime no disponible")
    
    def _detectar_device(self, preferido: str = "auto") -> str:
        """Detecta el mejor dispositivo disponible."""
        if preferido != "auto":
            return preferido
        
        if self.torch_cuda_disponible:
            return "cuda"
        
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except:
            pass
        
        return "cpu"
    
    def preparar_dataset_fase1(self) -> Path:
        """
        Prepara el dataset para la Fase 1 (detección de categorías).
        
        Genera un dataset con solo las 5 categorías principales:
        - Persona
        - Animal
        - VehiculoAereo
        - VehiculoTerrestre
        - VehiculoTerrestreMilitar
        """
        LOGGER.info("Preparando dataset para Fase 1 (Detección Base)")
        
        path_salida = self.path_datasets / "fase1_categorias"
        
        # Obtener solo categorías de nivel 1
        categorias = self.taxonomia.obtener_categorias()
        
        # Generar data.yaml
        nombres_clases = {c.id: c.nombre for c in categorias}
        
        config_yaml = {
            "path": str(path_salida.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(categorias),
            "names": nombres_clases,
        }
        
        path_salida.mkdir(parents=True, exist_ok=True)
        
        with open(path_salida / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config_yaml, f, allow_unicode=True, sort_keys=False)
        
        LOGGER.info(f"Dataset Fase 1 preparado en: {path_salida}")
        LOGGER.info(f"Clases: {list(nombres_clases.values())}")
        
        return path_salida
    
    def preparar_dataset_fase2(self, categoria: str) -> Path:
        """
        Prepara dataset para Fase 2 (clasificación de subcategorías).
        
        Para la categoría dada, genera un dataset de clasificación
        con sus subcategorías.
        
        Args:
            categoria: Nombre de la categoría (ej: "VehiculoAereo")
        """
        LOGGER.info(f"Preparando dataset Fase 2 para: {categoria}")
        
        clase_cat = self.taxonomia.obtener_clase(categoria)
        if not clase_cat:
            raise ValueError(f"Categoría no encontrada: {categoria}")
        
        path_salida = self.path_datasets / f"fase2_{categoria.lower()}"
        
        # Obtener subcategorías/subtipos
        hijos = self.taxonomia.obtener_hijos(clase_cat)
        
        if not hijos:
            raise ValueError(f"La categoría {categoria} no tiene subclases")
        
        nombres_clases = {c.id: c.nombre for c in hijos}
        
        config_yaml = {
            "path": str(path_salida.absolute()),
            "train": "train",
            "val": "val",
            "nc": len(hijos),
            "names": list(nombres_clases.values()),
        }
        
        path_salida.mkdir(parents=True, exist_ok=True)
        
        with open(path_salida / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config_yaml, f, allow_unicode=True, sort_keys=False)
        
        LOGGER.info(f"Dataset Fase 2 preparado en: {path_salida}")
        LOGGER.info(f"Clases: {list(nombres_clases.values())}")
        
        return path_salida
    
    def preparar_dataset_fase3(self, categoria: str, subcategoria: str) -> Path:
        """
        Prepara dataset para Fase 3 (clasificación de subtipos).
        
        Args:
            categoria: Categoría principal
            subcategoria: Subcategoría a detallar
        """
        LOGGER.info(f"Preparando dataset Fase 3: {categoria}.{subcategoria}")
        
        nombre_completo = f"{categoria}.{subcategoria}"
        clase_subcat = self.taxonomia.obtener_clase(nombre_completo)
        
        if not clase_subcat:
            # Intentar solo con subcategoria
            clase_subcat = self.taxonomia.obtener_clase(subcategoria)
        
        if not clase_subcat:
            raise ValueError(f"Subcategoría no encontrada: {nombre_completo}")
        
        path_salida = self.path_datasets / f"fase3_{categoria.lower()}_{subcategoria.lower()}"
        
        # Obtener subtipos
        subtipos = self.taxonomia.obtener_hijos(clase_subcat)
        
        if not subtipos:
            raise ValueError(f"La subcategoría {subcategoria} no tiene subtipos")
        
        nombres_clases = {c.id: c.nombre for c in subtipos}
        
        config_yaml = {
            "path": str(path_salida.absolute()),
            "train": "train",
            "val": "val",
            "nc": len(subtipos),
            "names": list(nombres_clases.values()),
        }
        
        path_salida.mkdir(parents=True, exist_ok=True)
        
        with open(path_salida / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config_yaml, f, allow_unicode=True, sort_keys=False)
        
        LOGGER.info(f"Dataset Fase 3 preparado en: {path_salida}")
        LOGGER.info(f"Subtipos: {list(nombres_clases.values())}")
        
        return path_salida
    
    def entrenar_fase1_deteccion(
        self,
        config: Optional[ConfiguracionEntrenamiento] = None,
    ) -> ResultadoEntrenamiento:
        """
        Entrena el modelo de detección base (Fase 1).
        
        Detecta las 5 categorías principales.
        """
        if not self.ultralytics_disponible:
            return ResultadoEntrenamiento(
                exito=False,
                errores=["Ultralytics no disponible. Instala con: pip install ultralytics"]
            )
        
        # Preparar dataset
        path_dataset = self.preparar_dataset_fase1()
        
        # Configuración por defecto
        if config is None:
            config = ConfiguracionEntrenamiento(
                nombre="fase1_categorias",
                path_dataset=path_dataset,
                path_salida=self.path_modelos / "fase1",
                fase=FaseEntrenamiento.DETECCION_BASE,
                epochs=100,
                img_size=640,
                modelo_base="yolo11n.pt",
            )
        
        return self._ejecutar_entrenamiento_yolo(config, tarea="detect")
    
    def entrenar_fase2_clasificacion(
        self,
        categoria: str,
        config: Optional[ConfiguracionEntrenamiento] = None,
    ) -> ResultadoEntrenamiento:
        """
        Entrena clasificador de subcategorías (Fase 2).
        
        Args:
            categoria: Categoría a clasificar (ej: "VehiculoAereo")
        """
        if not self.ultralytics_disponible:
            return ResultadoEntrenamiento(
                exito=False,
                errores=["Ultralytics no disponible"]
            )
        
        # Preparar dataset
        path_dataset = self.preparar_dataset_fase2(categoria)
        
        if config is None:
            config = ConfiguracionEntrenamiento(
                nombre=f"fase2_{categoria.lower()}",
                path_dataset=path_dataset,
                path_salida=self.path_modelos / f"fase2_{categoria.lower()}",
                fase=FaseEntrenamiento.CLASIFICACION_SUBCATEGORIA,
                categoria_objetivo=categoria,
                epochs=50,
                img_size=224,  # Clasificación usa imágenes más pequeñas
                modelo_base="yolo11n-cls.pt",  # Modelo de clasificación
            )
        
        return self._ejecutar_entrenamiento_yolo(config, tarea="classify")
    
    def entrenar_fase3_subtipos(
        self,
        categoria: str,
        subcategoria: str,
        config: Optional[ConfiguracionEntrenamiento] = None,
    ) -> ResultadoEntrenamiento:
        """
        Entrena clasificador de subtipos (Fase 3).
        
        Args:
            categoria: Categoría principal
            subcategoria: Subcategoría a detallar
        """
        if not self.ultralytics_disponible:
            return ResultadoEntrenamiento(
                exito=False,
                errores=["Ultralytics no disponible"]
            )
        
        # Preparar dataset
        path_dataset = self.preparar_dataset_fase3(categoria, subcategoria)
        
        if config is None:
            config = ConfiguracionEntrenamiento(
                nombre=f"fase3_{categoria.lower()}_{subcategoria.lower()}",
                path_dataset=path_dataset,
                path_salida=self.path_modelos / f"fase3_{categoria.lower()}_{subcategoria.lower()}",
                fase=FaseEntrenamiento.CLASIFICACION_SUBTIPO,
                categoria_objetivo=f"{categoria}.{subcategoria}",
                epochs=50,
                img_size=224,
                modelo_base="yolo11n-cls.pt",
            )
        
        return self._ejecutar_entrenamiento_yolo(config, tarea="classify")
    
    def _ejecutar_entrenamiento_yolo(
        self,
        config: ConfiguracionEntrenamiento,
        tarea: str = "detect",
    ) -> ResultadoEntrenamiento:
        """Ejecuta el entrenamiento con YOLO."""
        from ultralytics import YOLO
        import time
        
        inicio = time.time()
        
        try:
            # Detectar dispositivo
            device = self._detectar_device(config.device)
            LOGGER.info(f"Usando dispositivo: {device}")
            
            # Cargar modelo base
            modelo = YOLO(config.modelo_base)
            
            # Configurar entrenamiento
            args = {
                "data": str(config.path_dataset / "data.yaml"),
                "epochs": config.epochs,
                "imgsz": config.img_size,
                "batch": config.batch_size,
                "lr0": config.learning_rate,
                "device": device,
                "workers": config.workers,
                "patience": config.patience,
                "project": str(config.path_salida.parent),
                "name": config.path_salida.name,
                "exist_ok": True,
                "pretrained": config.usar_pretrained,
                "augment": config.augmentation,
                "verbose": True,
            }
            
            if config.freeze_backbone:
                args["freeze"] = 10  # Congelar primeras 10 capas
            
            # Guardar configuración
            config.path_salida.mkdir(parents=True, exist_ok=True)
            with open(config.path_salida / "config.json", "w") as f:
                json.dump(config.to_dict(), f, indent=2)
            
            # Entrenar
            LOGGER.info(f"Iniciando entrenamiento: {config.nombre}")
            LOGGER.info(f"Épocas: {config.epochs}, Batch: {config.batch_size}")
            
            if tarea == "detect":
                results = modelo.train(**args)
            else:
                # Para clasificación, ajustar la ruta de datos
                args["data"] = str(config.path_dataset)
                results = modelo.train(**args)
            
            duracion = time.time() - inicio
            
            # Extraer métricas
            metricas = {}
            if hasattr(results, "results_dict"):
                metricas = results.results_dict
            
            # Path del mejor modelo
            path_mejor = config.path_salida / "weights" / "best.pt"
            
            # Exportar a ONNX si es posible
            try:
                modelo_mejor = YOLO(path_mejor)
                path_onnx = modelo_mejor.export(format="onnx", imgsz=config.img_size)
                LOGGER.info(f"Modelo exportado a ONNX: {path_onnx}")
            except Exception as e:
                LOGGER.warning(f"No se pudo exportar a ONNX: {e}")
            
            LOGGER.info(f"Entrenamiento completado en {duracion:.1f}s")
            
            return ResultadoEntrenamiento(
                exito=True,
                path_modelo=path_mejor,
                metricas=metricas,
                duracion_segundos=duracion,
            )
            
        except Exception as e:
            duracion = time.time() - inicio
            LOGGER.error(f"Error en entrenamiento: {e}")
            
            return ResultadoEntrenamiento(
                exito=False,
                duracion_segundos=duracion,
                errores=[str(e)],
            )
    
    def generar_plan_entrenamiento(self) -> Dict[str, Any]:
        """
        Genera un plan de entrenamiento completo basado en la taxonomía.
        
        Returns:
            Dict con todas las fases y configuraciones necesarias
        """
        plan = {
            "version": "1.0",
            "generado": datetime.now().isoformat(),
            "fases": [],
        }
        
        # Fase 1: Detección base
        categorias = self.taxonomia.obtener_categorias()
        plan["fases"].append({
            "fase": 1,
            "nombre": "Detección Base",
            "tipo": "detect",
            "clases": [c.nombre for c in categorias],
            "modelo_sugerido": "yolo11m.pt",
            "epochs_sugeridos": 100,
        })
        
        # Fase 2: Clasificación por categoría
        for cat in categorias:
            hijos = self.taxonomia.obtener_hijos(cat)
            if hijos:
                plan["fases"].append({
                    "fase": 2,
                    "nombre": f"Clasificación {cat.nombre}",
                    "categoria": cat.nombre,
                    "tipo": "classify",
                    "clases": [h.nombre for h in hijos],
                    "modelo_sugerido": "yolo11n-cls.pt",
                    "epochs_sugeridos": 50,
                })
                
                # Fase 3: Subtipos
                for hijo in hijos:
                    nietos = self.taxonomia.obtener_hijos(hijo)
                    if nietos:
                        plan["fases"].append({
                            "fase": 3,
                            "nombre": f"Subtipos {hijo.nombre}",
                            "categoria": cat.nombre,
                            "subcategoria": hijo.nombre,
                            "tipo": "classify",
                            "clases": [n.nombre for n in nietos],
                            "modelo_sugerido": "yolo11n-cls.pt",
                            "epochs_sugeridos": 30,
                        })
        
        # Guardar plan
        path_plan = self.path_configs / "plan_entrenamiento.json"
        with open(path_plan, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        
        LOGGER.info(f"Plan de entrenamiento guardado en: {path_plan}")
        
        return plan
    
    def imprimir_plan(self):
        """Imprime el plan de entrenamiento de forma legible."""
        plan = self.generar_plan_entrenamiento()
        
        print("\n" + "=" * 60)
        print("PLAN DE ENTRENAMIENTO JERÁRQUICO")
        print("=" * 60)
        
        for fase in plan["fases"]:
            print(f"\n[FASE {fase['fase']}] {fase['nombre']}")
            print(f"  Tipo: {fase['tipo']}")
            print(f"  Modelo: {fase['modelo_sugerido']}")
            print(f"  Épocas: {fase['epochs_sugeridos']}")
            print(f"  Clases ({len(fase['clases'])}): {', '.join(fase['clases'][:5])}")
            if len(fase['clases']) > 5:
                print(f"          ... y {len(fase['clases']) - 5} más")


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def iniciar_proyecto_entrenamiento(path: Path) -> EntrenadorJerarquico:
    """
    Inicializa un nuevo proyecto de entrenamiento.
    
    Crea la estructura de directorios y genera archivos de configuración.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    entrenador = EntrenadorJerarquico(path)
    
    # Generar taxonomía
    entrenador.taxonomia.generar_json_taxonomia(path / "configs" / "taxonomia.json")
    
    # Generar plan
    entrenador.generar_plan_entrenamiento()
    
    print(f"\nProyecto de entrenamiento inicializado en: {path}")
    print("\nEstructura creada:")
    print("  datasets/   - Datasets procesados por fase")
    print("  models/     - Modelos entrenados")
    print("  logs/       - Logs de entrenamiento")
    print("  configs/    - Configuraciones y taxonomía")
    
    return entrenador


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        path_proyecto = Path(sys.argv[1])
    else:
        path_proyecto = Path("proyecto_entrenamiento")
    
    entrenador = iniciar_proyecto_entrenamiento(path_proyecto)
    entrenador.imprimir_plan()
