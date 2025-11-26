from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO

# Modelo global (se carga solo una vez)
_MODEL = None

# Tamaño de entrada para YOLO: más grande = mejor para objetos pequeños (a costa de CPU)
_IMG_SIZE = 1280

# Tipo de detección que usamos en todo el sistema
Detection = Dict[str, Any]


def _cargar_modelo():
    """
    Carga el modelo YOLO una sola vez.
    IMPORTANTE: no movemos a CUDA porque tu instalación de PyTorch no lo soporta bien.
    Trabajamos en CPU.
    """
    global _MODEL
    if _MODEL is None:
        # Puedes cambiar 'yolo11x.pt' por 'yolo11m.pt' o 'yolo11s.pt' si quieres probar otros
        ruta_pesos = "yolo11x.pt"
        print(f"[deteccion] Cargando modelo YOLO desde {ruta_pesos} (CPU)...")
        _MODEL = YOLO(ruta_pesos)
    return _MODEL


def detectar(frame: np.ndarray) -> List[Detection]:
    """
    Detección "cruda" con YOLO:
    - Sin filtros por clase.
    - Sin filtros por tamaño.
    - Solo conf mínima e IoU por defecto.

    Devolvemos SIEMPRE una lista de dicts con:
      x1, y1, x2, y2, clase (string), score (float)
    """
    model = _cargar_modelo()

    # Llamada a YOLO
    results = model(
        frame,
        imgsz=_IMG_SIZE,
        conf=0.10,      # más bajo para no perder cosas débiles
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
