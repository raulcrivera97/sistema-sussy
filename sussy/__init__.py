"""
Sistema Sussy - Detección y seguimiento de drones mediante visión por computadora.

Este módulo proporciona un pipeline completo para detectar y rastrear objetos voladores
(drones, aviones, pájaros) usando YOLO y detección de movimiento clásica.
"""

__version__ = "1.0.0"
__author__ = "Proyecto Sussy"

# Exports principales
from sussy.config import Config
from sussy.core.pipeline import SussyPipeline, FrameResult

# Exports de componentes principales (opcional, para uso avanzado)
from sussy.core.deteccion import detectar, Detection
from sussy.core.movimiento import DetectorMovimiento
from sussy.core.seguimiento import TrackerSimple
from sussy.core.relevancia import EvaluadorRelevancia

__all__ = [
    # Versión
    "__version__",
    # Configuración
    "Config",
    # Pipeline principal
    "SussyPipeline",
    "FrameResult",
    # Componentes (uso avanzado)
    "detectar",
    "Detection",
    "DetectorMovimiento",
    "TrackerSimple",
    "EvaluadorRelevancia",
]
