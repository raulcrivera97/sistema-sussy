import os
import cv2
import numpy as np
from datetime import datetime

def guardar_crop(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    clase: str,
    ruta_base: str,
    padding_pct: float = 0.0,
):
    """
    Guarda un recorte de la imagen en la ruta especificada, organizado por fecha y clase.
    Estructura: ruta_base / YYYY-MM-DD / clase / timestamp_id.jpg
    """
    if frame is None or frame.size == 0:
        return

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
        return

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return

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
        print(f"[dataset_utils] Error al guardar crop: {e}")
