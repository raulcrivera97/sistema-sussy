import cv2
import numpy as np
from typing import Generator, Tuple


def abrir_video(ruta: str) -> cv2.VideoCapture:
    """
    Abre un archivo de vídeo y devuelve el objeto VideoCapture.
    Lanza un RuntimeError si no se puede abrir.
    """
    cap = cv2.VideoCapture(ruta)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {ruta}")

    return cap


def frames_desde_video(ruta: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Genera (indice_frame, frame_bgr) desde un archivo de vídeo.

    - indice_frame: empieza en 0 y sube de 1 en 1.
    - frame_bgr: imagen en formato BGR (como la devuelve OpenCV).
    """
    cap = abrir_video(ruta)
    idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # frame es un np.ndarray con shape (alto, ancho, 3)
            yield idx, frame
            idx += 1
    finally:
        cap.release()
