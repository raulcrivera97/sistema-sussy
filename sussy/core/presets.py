from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sussy.config import Config


@dataclass(frozen=True)
class PresetCamara:
    clave: str
    nombre: str
    descripcion: str
    ajustes: Dict[str, Any]

@dataclass(frozen=True)
class PresetRendimiento:
    clave: str
    nombre: str
    descripcion: str
    ajustes: Dict[str, Any]


PRESETS_CAMARA: Dict[str, PresetCamara] = {
    "fija": PresetCamara(
        clave="fija",
        nombre="Cámara Fija",
        descripcion=(
            "Posición estática sin movimientos ni zoom. "
            "Desactiva cualquier protección de movimiento de cámara para ahorrar recursos."
        ),
        ajustes={
            "USAR_DETECTOR_MOVIMIENTO": True,
            "USAR_MONITOR_ESTABILIDAD": False,
            "USAR_PREDICCION_MOVIMIENTO": True,
            "CAMARA_ZOOM_FAST_RATIO": 0.0,
            "CAMARA_ZOOM_COOLDOWN_FRAMES": Config.CAMARA_ZOOM_COOLDOWN_FRAMES,
            "MOVIMIENTO_RAFAGA_BLOBS": Config.MOVIMIENTO_RAFAGA_BLOBS,
            "MOVIMIENTO_ANOMALIA_TOTAL": Config.MOVIMIENTO_ANOMALIA_TOTAL,
            "MOVIMIENTO_ANOMALIA_POSIBLE_DRON": Config.MOVIMIENTO_ANOMALIA_POSIBLE_DRON,
        },
    ),
    "orientable": PresetCamara(
        clave="orientable",
        nombre="Cámara Orientable",
        descripcion=(
            "Cabezal motorizado con giro/inclinación/zoom. "
            "Activa la lógica de movimiento y zoom para pausar el pipeline cuando la cámara se mueve."
        ),
        ajustes={
            "USAR_DETECTOR_MOVIMIENTO": True,
            "USAR_MONITOR_ESTABILIDAD": True,
            "USAR_PREDICCION_MOVIMIENTO": True,
            "CAMARA_ZOOM_FAST_RATIO": 0.25,
            "CAMARA_ZOOM_COOLDOWN_FRAMES": Config.CAMARA_ZOOM_COOLDOWN_FRAMES,
        },
    ),
    "movil": PresetCamara(
        clave="movil",
        nombre="Cámara Móvil",
        descripcion=(
            "Unidad que se desplaza físicamente sin zoom óptico. "
            "Desactiva los módulos sensibles al movimiento propio."
        ),
        ajustes={
            "USAR_DETECTOR_MOVIMIENTO": False,
            "USAR_MONITOR_ESTABILIDAD": False,
            "USAR_PREDICCION_MOVIMIENTO": False,
            "CAMARA_ZOOM_FAST_RATIO": 0.0,
            "CAMARA_ZOOM_COOLDOWN_FRAMES": 0,
        },
    ),
    "movil_plus": PresetCamara(
        clave="movil_plus",
        nombre="Cámara Móvil+",
        descripcion=(
            "Unidad móvil con capacidad adicional de orientación y barridos. "
            "Mantiene desactivada la detección de movimiento clásica, "
            "pero se apoya en el tracker y la predicción para anticipar zonas."
        ),
        ajustes={
            "USAR_DETECTOR_MOVIMIENTO": False,
            "USAR_MONITOR_ESTABILIDAD": False,
            "USAR_PREDICCION_MOVIMIENTO": True,
            "CAMARA_ZOOM_FAST_RATIO": 0.0,
            "CAMARA_ZOOM_COOLDOWN_FRAMES": 0,
            "TRACKER_MAX_FRAMES_LOST": max(Config.TRACKER_MAX_FRAMES_LOST, 15),
            "TRACKER_MATCH_DIST": max(Config.TRACKER_MATCH_DIST, 140),
        },
    ),
}

PRESETS_RENDIMIENTO: Dict[str, PresetRendimiento] = {
    "ultraligero": PresetRendimiento(
        clave="ultraligero",
        nombre="Ultraligero (GPU)",
        descripcion=(
            "Máxima velocidad: modelo nano en GPU. ~50+ FPS"
        ),
        ajustes={
            "YOLO_MODELO": "yolo11n.pt",
            "YOLO_IMG_SIZE": 512,  # yolo11n.onnx exportado a 512
            "SKIP_FRAMES_DEFECTO": 2,
            "USAR_DETECTOR_MOVIMIENTO": False,
            "USAR_PREDICCION_MOVIMIENTO": False,
            "USAR_FILTRO_IA_EN_MOVIMIENTO": False,
        },
    ),
    "rapido": PresetRendimiento(
        clave="rapido",
        nombre="Rápido (GPU)",
        descripcion=(
            "Buena velocidad con modelo nano en GPU. ~30-40 FPS"
        ),
        ajustes={
            "YOLO_MODELO": "yolo11n.pt",
            "YOLO_IMG_SIZE": 512,  # Usar el mismo tamaño del ONNX
            "SKIP_FRAMES_DEFECTO": 1,
            "USAR_DETECTOR_MOVIMIENTO": True,
            "USAR_PREDICCION_MOVIMIENTO": False,
            "USAR_FILTRO_IA_EN_MOVIMIENTO": False,
        },
    ),
    "equilibrado": PresetRendimiento(
        clave="equilibrado",
        nombre="Equilibrado (GPU)",
        descripcion=(
            "Balance velocidad/precisión con modelo grande. ~15-25 FPS"
        ),
        ajustes={
            "YOLO_MODELO": "yolo11x.pt",
            "YOLO_IMG_SIZE": 960,  # yolo11x.onnx exportado a 960
            "SKIP_FRAMES_DEFECTO": 2,
            "USAR_DETECTOR_MOVIMIENTO": True,
        },
    ),
    "calidad": PresetRendimiento(
        clave="calidad",
        nombre="Alta calidad (GPU)",
        descripcion=(
            "Mayor precisión con modelo grande. ~10-15 FPS"
        ),
        ajustes={
            "YOLO_MODELO": "yolo11x.pt",
            "YOLO_IMG_SIZE": 960,
            "SKIP_FRAMES_DEFECTO": 1,
            "USAR_DETECTOR_MOVIMIENTO": True,
            "USAR_PREDICCION_MOVIMIENTO": True,
        },
    ),
    "maximo": PresetRendimiento(
        clave="maximo",
        nombre="Máxima precisión (GPU)",
        descripcion=(
            "Calidad máxima, todos los módulos activos. ~5-10 FPS"
        ),
        ajustes={
            "YOLO_MODELO": "yolo11x.pt",
            "YOLO_IMG_SIZE": 960,
            "SKIP_FRAMES_DEFECTO": 1,
            "USAR_DETECTOR_MOVIMIENTO": True,
            "USAR_PREDICCION_MOVIMIENTO": True,
            "USAR_FILTRO_IA_EN_MOVIMIENTO": True,
        },
    ),
}


def presets_disponibles() -> List[str]:
    """Devuelve la lista de claves disponibles para argparse u otras UIs."""
    return sorted(PRESETS_CAMARA.keys())

def presets_rendimiento_disponibles() -> List[str]:
    """Lista de presets de rendimiento disponibles."""
    return sorted(PRESETS_RENDIMIENTO.keys())


def aplicar_preset_camara(
    nombre: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> PresetCamara:
    """
    Sobrescribe dinamicamente atributos de Config según el preset solicitado.
    Los overrides se aplican al final para mantener ajustes manuales por ejecución.
    """
    clave = (nombre or "").lower()
    if clave not in PRESETS_CAMARA:
        raise ValueError(f"Preset de cámara desconocido: {nombre}")

    preset = PRESETS_CAMARA[clave]
    ajustes = dict(preset.ajustes)
    if overrides:
        ajustes.update(overrides)

    for atributo, valor in ajustes.items():
        if not hasattr(Config, atributo):
            raise AttributeError(
                f"El atributo '{atributo}' definido en el preset '{preset.nombre}' no existe en Config."
            )
        setattr(Config, atributo, valor)

    return preset


def aplicar_preset_rendimiento(
    nombre: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> PresetRendimiento:
    """
    Sobrescribe dinamicamente atributos de Config según el preset de rendimiento.
    Pensado para ajustar coste computacional variando modelo y skip de frames.
    """
    clave = (nombre or "").lower()
    if clave not in PRESETS_RENDIMIENTO:
        raise ValueError(f"Preset de rendimiento desconocido: {nombre}")

    preset = PRESETS_RENDIMIENTO[clave]
    ajustes = dict(preset.ajustes)
    if overrides:
        ajustes.update(overrides)

    for atributo, valor in ajustes.items():
        if not hasattr(Config, atributo):
            raise AttributeError(
                f"El atributo '{atributo}' definido en el preset '{preset.nombre}' no existe en Config."
            )
        setattr(Config, atributo, valor)

    return preset


