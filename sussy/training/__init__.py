"""
Módulo de entrenamiento para Sistema Sussy.

Incluye:
- Taxonomía jerárquica de clases
- Extractor de frames de vídeo
- Organizador de datasets
- Backend de entrenamiento adaptativo (Local/Lambda/Cloud)
- Anotador simple con OpenCV
- Scripts de entrenamiento CLI

Uso rápido:
    # Extraer frames de vídeo
    from sussy.training.extractor_frames import ExtractorFrames
    
    # Organizar dataset
    from sussy.training.organizador_dataset import OrganizadorDataset
    
    # Entrenar (CLI)
    python -m sussy.training.entrenar --preset desarrollo --dataset mi_dataset
    
    # Anotar imágenes (CLI)
    python -m sussy.training.anotador_simple frames/ salida/ --clases drone vehicle
"""

from sussy.training.taxonomia import (
    TAXONOMIA,
    GestorTaxonomia,
    ClaseDeteccion,
    EtiquetaDeteccion,
    obtener_taxonomia,
    Rol,
    TipoCarga,
    Armamento,
    AccionPersona,
)

from sussy.training.extractor_frames import (
    ExtractorFrames,
    ConfigExtraccion,
    FrameExtraido,
    dividir_por_videos,
)

from sussy.training.organizador_dataset import (
    OrganizadorDataset,
    BoundingBox,
    ImagenAnotada,
    crear_desde_carpeta_cvat,
)

from sussy.training.backend_entrenamiento import (
    EntrenadorAdaptativo,
    DetectorHardware,
    SelectorBackend,
    ConfigBackend,
    TipoBackend,
    InfoHardware,
    generar_script_lambda,
)

__all__ = [
    # Taxonomía
    "TAXONOMIA",
    "GestorTaxonomia",
    "ClaseDeteccion",
    "EtiquetaDeteccion",
    "obtener_taxonomia",
    "Rol",
    "TipoCarga",
    "Armamento",
    "AccionPersona",
    # Extractor de frames
    "ExtractorFrames",
    "ConfigExtraccion",
    "FrameExtraido",
    "dividir_por_videos",
    # Organizador de dataset
    "OrganizadorDataset",
    "BoundingBox",
    "ImagenAnotada",
    "crear_desde_carpeta_cvat",
    # Backend de entrenamiento
    "EntrenadorAdaptativo",
    "DetectorHardware",
    "SelectorBackend",
    "ConfigBackend",
    "TipoBackend",
    "InfoHardware",
    "generar_script_lambda",
]
