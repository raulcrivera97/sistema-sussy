import cv2
import numpy as np
from typing import List, Dict, Any

def dibujar_tracks(frame: np.ndarray, tracks: List[Dict[str, Any]]) -> None:
    """
    Dibuja los bounding boxes y los IDs de los tracks en el frame.
    """
    for track in tracks:
        x1, y1, x2, y2 = map(int, track['box'])
        tid = track['id']
        clase = track.get('clase', 'unk')
        score = track.get('score', 0.0)

        # Color diferente para movimiento vs yolo?
        # Por ahora verde para todo
        color = (0, 255, 0)
        if clase == "movimiento":
            color = (0, 255, 255) # Amarillo para movimiento puro

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID:{tid} {clase} {score:.2f}"
        
        # Fondo para el texto
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
