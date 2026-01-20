"""
Utilidades para captura y gestión de dataset de entrenamiento.
Incluye filtros de calidad y análisis anti-vegetación.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np

LOGGER = logging.getLogger("sussy.dataset_utils")


@dataclass
class CropMetadata:
    """Metadatos del crop para análisis posterior."""
    timestamp: str
    clase: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    area_px: int
    area_pct: float
    velocidad_px: float
    frames_vivos: int
    ratio_aspecto: float
    # Análisis de imagen
    ratio_verde: float
    ratio_marron: float
    desviacion_color: float
    ratio_bordes: float
    es_vegetacion_probable: bool
    # Fuente
    frame_idx: int
    fuente: str


def analizar_vegetacion(crop: np.ndarray) -> Dict[str, float]:
    """
    Analiza el crop para detectar si es probable vegetación.
    Retorna métricas de color y textura.
    """
    if crop is None or crop.size == 0:
        return {
            "ratio_verde": 0.0,
            "ratio_marron": 0.0,
            "desviacion_color": 0.0,
            "ratio_bordes": 0.0,
        }

    # Convertir a HSV para análisis de color
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    total_px = max(1, crop.shape[0] * crop.shape[1])

    # Detección de verde (vegetación)
    # Verde en HSV: H=35-85, S>40, V>40
    mask_verde = (
        (h >= 35) & (h <= 85) &
        (s >= 40) & (v >= 40)
    )
    ratio_verde = float(np.count_nonzero(mask_verde)) / total_px

    # Detección de marrón (ramas, troncos)
    # Marrón en HSV: H=10-30, S>30, V=30-180
    mask_marron = (
        (h >= 10) & (h <= 30) &
        (s >= 30) & (v >= 30) & (v <= 180)
    )
    ratio_marron = float(np.count_nonzero(mask_marron)) / total_px

    # Desviación estándar de color (objetos artificiales suelen ser más uniformes)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    desviacion_color = float(np.std(gray))

    # Detección de bordes (drones tienen bordes más definidos que vegetación)
    edges = cv2.Canny(gray, 50, 150)
    ratio_bordes = float(np.count_nonzero(edges)) / total_px

    return {
        "ratio_verde": ratio_verde,
        "ratio_marron": ratio_marron,
        "desviacion_color": desviacion_color,
        "ratio_bordes": ratio_bordes,
    }


def es_vegetacion_probable(
    analisis: Dict[str, float],
    ratio_verde_min: float = 0.25,
    ratio_marron_min: float = 0.30,
    desviacion_min: float = 15.0,
    bordes_min: float = 0.02,
) -> bool:
    """
    Determina si el crop es probable vegetación basado en el análisis.
    """
    # Demasiado verde = vegetación
    if analisis["ratio_verde"] > ratio_verde_min:
        return True

    # Demasiado marrón = rama/tronco
    if analisis["ratio_marron"] > ratio_marron_min:
        return True

    # Muy poca variación de color Y pocos bordes = fondo difuso
    if analisis["desviacion_color"] < desviacion_min and analisis["ratio_bordes"] < bordes_min:
        return True

    return False


def calcular_calidad_crop(
    det: Dict[str, Any],
    frame_shape: tuple,
    analisis_veg: Dict[str, float],
) -> float:
    """
    Calcula un score de calidad del crop (0-1) basado en múltiples factores.
    """
    score_base = det.get("score", 0.5)
    
    # Penalizar si es vegetación probable
    if es_vegetacion_probable(analisis_veg):
        score_base *= 0.3
    
    # Bonus por velocidad (objetos en movimiento real)
    vel = det.get("velocidad_px", 0)
    if vel > 2.0:
        score_base = min(1.0, score_base + 0.1)
    
    # Bonus por persistencia
    frames_vivos = det.get("frames_vivos", 0)
    if frames_vivos >= 5:
        score_base = min(1.0, score_base + 0.1)
    
    # Penalizar bordes del frame
    alto, ancho = frame_shape[:2]
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    margen = 0.05
    if x1 < ancho * margen or x2 > ancho * (1 - margen):
        score_base *= 0.8
    if y1 < alto * margen or y2 > alto * (1 - margen):
        score_base *= 0.8
    
    return max(0.0, min(1.0, score_base))


def guardar_crop(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    clase: str,
    ruta_base: str,
    padding_pct: float = 0.0,
    det: Optional[Dict[str, Any]] = None,
    frame_idx: int = 0,
    fuente: str = "",
) -> Optional[str]:
    """
    Guarda un recorte de la imagen en la ruta especificada, organizado por fecha y clase.
    Estructura: ruta_base / YYYY-MM-DD / clase / timestamp_id.jpg
    
    Ahora incluye:
    - Filtro de calidad
    - Análisis anti-vegetación
    - Metadatos JSON opcionales
    
    Retorna la ruta del archivo guardado o None si no se guardó.
    """
    from sussy.config import Config

    if frame is None or frame.size == 0:
        return None

    # Validar coordenadas
    alto, ancho = frame.shape[:2]

    if padding_pct > 0.0:
        width = x2 - x1
        height = y2 - y1
        pad_x = int(max(2, round(width * padding_pct)))
        pad_y = int(max(2, round(height * padding_pct)))
        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_y
        y2 += pad_y

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(ancho, x2), min(alto, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    # Filtro de área
    area_px = (x2 - x1) * (y2 - y1)
    area_pct = area_px / max(1, alto * ancho)
    
    min_area = getattr(Config, "CROP_MIN_AREA_PX", 100)
    max_area_pct = getattr(Config, "CROP_MAX_AREA_PCT", 0.15)
    
    if area_px < min_area:
        return None
    if area_pct > max_area_pct:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Análisis anti-vegetación
    analisis_veg = analizar_vegetacion(crop)
    
    filtrar_veg = getattr(Config, "CROP_FILTRAR_VEGETACION", True)
    if filtrar_veg:
        es_veg = es_vegetacion_probable(
            analisis_veg,
            ratio_verde_min=getattr(Config, "VEGETACION_RATIO_VERDE_MIN", 0.25),
            ratio_marron_min=getattr(Config, "VEGETACION_RATIO_MARRON_MIN", 0.30),
            desviacion_min=getattr(Config, "VEGETACION_DESVIACION_COLOR_MIN", 15),
            bordes_min=getattr(Config, "VEGETACION_BORDES_MIN_RATIO", 0.02),
        )
        # Si es vegetación y la clase no indica lo contrario, no guardar
        if es_veg and clase in ("unknown", "movimiento", "posible_dron"):
            LOGGER.debug("Crop descartado por filtro anti-vegetación")
            return None

    # Filtro por score si está habilitado
    solo_validados = getattr(Config, "CROP_GUARDAR_SOLO_VALIDADOS", False)
    min_score = getattr(Config, "CROP_MIN_SCORE", 0.35)
    
    score = det.get("score", 0.5) if det else 0.5
    if solo_validados and score < min_score:
        return None

    # Crear estructura de directorios
    fecha_str = datetime.now().strftime("%Y-%m-%d")
    dir_destino = os.path.join(ruta_base, fecha_str, clase)
    os.makedirs(dir_destino, exist_ok=True)

    # Nombre de archivo único
    timestamp = datetime.now().strftime("%H-%M-%S-%f")
    nombre_archivo = f"{timestamp}.jpg"
    ruta_completa = os.path.join(dir_destino, nombre_archivo)

    try:
        cv2.imwrite(ruta_completa, crop)
    except Exception as e:
        LOGGER.error("Error al guardar crop: %s", e)
        return None

    # Guardar metadatos si está habilitado
    guardar_meta = getattr(Config, "CROP_GUARDAR_METADATOS", True)
    if guardar_meta and det:
        w, h = x2 - x1, y2 - y1
        metadata = CropMetadata(
            timestamp=timestamp,
            clase=clase,
            score=score,
            x1=x1, y1=y1, x2=x2, y2=y2,
            area_px=area_px,
            area_pct=area_pct,
            velocidad_px=det.get("velocidad_px", 0.0),
            frames_vivos=det.get("frames_vivos", 0),
            ratio_aspecto=w / max(1, h),
            ratio_verde=analisis_veg["ratio_verde"],
            ratio_marron=analisis_veg["ratio_marron"],
            desviacion_color=analisis_veg["desviacion_color"],
            ratio_bordes=analisis_veg["ratio_bordes"],
            es_vegetacion_probable=es_vegetacion_probable(analisis_veg),
            frame_idx=frame_idx,
            fuente=fuente,
        )
        ruta_meta = ruta_completa.replace(".jpg", ".json")
        try:
            with open(ruta_meta, "w", encoding="utf-8") as f:
                json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
        except Exception as e:
            LOGGER.warning("Error guardando metadatos: %s", e)

    return ruta_completa


def limpiar_crops_vegetacion(ruta_base: str, umbral_verde: float = 0.3) -> int:
    """
    Utilidad para limpiar crops que probablemente sean vegetación
    basándose en los metadatos guardados.
    Retorna el número de archivos eliminados.
    """
    eliminados = 0
    for root, dirs, files in os.walk(ruta_base):
        for f in files:
            if f.endswith(".json"):
                ruta_json = os.path.join(root, f)
                try:
                    with open(ruta_json, "r", encoding="utf-8") as fp:
                        meta = json.load(fp)
                    
                    if meta.get("es_vegetacion_probable", False) or meta.get("ratio_verde", 0) > umbral_verde:
                        # Eliminar JSON y JPG
                        ruta_jpg = ruta_json.replace(".json", ".jpg")
                        if os.path.exists(ruta_jpg):
                            os.remove(ruta_jpg)
                            eliminados += 1
                        os.remove(ruta_json)
                except Exception:
                    pass
    
    return eliminados
