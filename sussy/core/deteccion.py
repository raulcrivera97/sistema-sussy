from typing import List, Dict, Any, Optional

import logging
import numpy as np
from ultralytics import YOLO

# Modelo global (se carga solo una vez)
_MODEL = None
_WARMED_UP = False

# Tamaño de entrada para YOLO: más grande = mejor para objetos pequeños (a costa de CPU)
_IMG_SIZE = 1280

# Tipo de detección que usamos en todo el sistema
Detection = Dict[str, Any]

LOGGER = logging.getLogger("sussy.deteccion")


def _cargar_modelo(ruta_pesos: str = "yolo11n.pt"):
    """
    Carga el modelo YOLO una sola vez y ejecuta un warmup rápido para estabilizar
    los tiempos de inferencia posteriores.
    """
    global _MODEL, _WARMED_UP
    if _MODEL is None:
        LOGGER.info("Cargando modelo YOLO desde %s (modo CPU).", ruta_pesos)
        _MODEL = YOLO(ruta_pesos)
        _WARMED_UP = False

    if not _WARMED_UP:
        _ejecutar_warmup(_MODEL)
        _WARMED_UP = True

    return _MODEL


def _ejecutar_warmup(modelo: YOLO) -> None:
    """
    Realiza una inferencia sobre un frame vacío para inicializar kernels.
    Cualquier error en esta fase se informa pero no detiene la app.
    """
    try:
        frame_dummy = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
        modelo.predict(
            frame_dummy,
            imgsz=_IMG_SIZE,
            conf=0.01,
            iou=0.25,
            verbose=False,
        )
        LOGGER.debug("Warmup YOLO completado.")
    except Exception as exc:  # pragma: no cover - solo informativo
        LOGGER.warning("No se pudo ejecutar el warmup de YOLO: %s", exc)


def detectar(frame: np.ndarray, conf_umbral: float = 0.5, modelo_path: str = "yolo11n.pt", clases_permitidas: List[str] = None) -> List[Detection]:
    """
    Detección "cruda" con YOLO:
    - Sin filtros por tamaño.
    - Solo conf mínima e IoU por defecto.
    - Filtrado opcional por lista de nombres de clases (clases_permitidas).

    Devolvemos SIEMPRE una lista de dicts con:
      x1, y1, x2, y2, clase (string), score (float)
    """
    model = _cargar_modelo(modelo_path)

    # Llamada a YOLO
    results = model(
        frame,
        imgsz=_IMG_SIZE,
        conf=conf_umbral,
        iou=0.50,
        verbose=False,
    )

    detecciones: List[Detection] = []

    if not results:
        return detecciones

    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return detecciones

    names = model.names  # diccionario id -> nombre de clase

    for box in boxes:
        cls_id = int(box.cls.item())
        score = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        clase = names.get(cls_id, str(cls_id))

        # Filtrado por clase
        if clases_permitidas and len(clases_permitidas) > 0:
            if clase not in clases_permitidas:
                continue

        det: Detection = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "clase": clase,
            "score": score,
        }
        detecciones.append(det)

    return detecciones




def combinar_detecciones(
    dets_yolo: List[Detection],
    dets_mov: List[Detection],
    iou_thresh: float = 0.3
) -> List[Detection]:
    """
    Combina detecciones de YOLO y Movimiento.
    - Prioridad a YOLO: si un objeto de movimiento solapa con uno de YOLO,
      nos quedamos con el de YOLO (que tiene clase específica).
    - Si Movimiento no solapa con nada de YOLO, lo añadimos como posible objeto de interés.
    """
    # Importar aquí para evitar ciclos o simplemente usar la utilidad
    from sussy.core.utilidades_iou import calcular_iou
    
    finales = list(dets_yolo)  # Copia superficial

    for d_mov in dets_mov:
        solapa = False
        for d_yolo in dets_yolo:
            iou = calcular_iou(d_mov, d_yolo)
            if iou > iou_thresh:
                solapa = True
                break
        
        if not solapa:
            # Es un objeto en movimiento que YOLO no ha visto
            finales.append(d_mov)
            
    return finales


def analizar_recorte(
    frame: np.ndarray, 
    x1: int, y1: int, x2: int, y2: int, 
    conf_umbral: float = 0.3, # Umbral más bajo para segunda pasada
    modelo_path: str = "yolo11n.pt",
    clases_permitidas: List[str] = None,
    padding_pct: float = 0.0,
) -> Optional[Detection]:
    """
    Recorta la región indicada y pasa YOLO solo a ese trozo.
    Útil para verificar qué es un objeto en movimiento pequeño.
    Retorna la mejor detección encontrada en el recorte (ajustada a coordenadas globales), o None.
    """
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

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Detectar en el crop
    # NOTA: Usamos un umbral un poco más bajo por defecto porque se supone que "algo se mueve"
    # y queremos saber qué es.
    dets_crop = detectar(crop, conf_umbral=conf_umbral, modelo_path=modelo_path, clases_permitidas=clases_permitidas)
    
    if not dets_crop:
        return None
    
    # Nos quedamos con la detección con mayor score
    mejor_det = max(dets_crop, key=lambda d: d['score'])
    
    # Ajustar coordenadas del crop a globales
    mejor_det['x1'] += x1
    mejor_det['x2'] += x1
    mejor_det['y1'] += y1
    mejor_det['y2'] += y1
    
    return mejor_det

