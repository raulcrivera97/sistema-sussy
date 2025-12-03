from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sussy.config import Config


@dataclass(frozen=True)
class PresetCamara:
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
            "Desactiva cualquier análisis de movimiento propio para ahorrar recursos."
        ),
        ajustes={
            "USAR_DETECTOR_MOVIMIENTO": False,
            "USAR_MONITOR_ESTABILIDAD": False,
            "USAR_PREDICCION_MOVIMIENTO": False,
            "CAMARA_ZOOM_FAST_RATIO": 0.0,
            "CAMARA_ZOOM_COOLDOWN_FRAMES": 0,
            "MOVIMIENTO_RAFAGA_BLOBS": 0,
            "MOVIMIENTO_ANOMALIA_TOTAL": 0,
            "MOVIMIENTO_ANOMALIA_POSIBLE_DRON": 0,
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


def presets_disponibles() -> List[str]:
    """Devuelve la lista de claves disponibles para argparse u otras UIs."""
    return sorted(PRESETS_CAMARA.keys())


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


