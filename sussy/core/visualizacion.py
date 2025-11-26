from typing import List

import cv2
import numpy as np

from sussy.core.seguimiento import Track


def dibujar_tracks(frame: np.ndarray, tracks: List[Track]) -> None:
    """
    Dibuja las cajas y los IDs de los tracks sobre el frame.
    Modifica el frame IN PLACE.
    """
    for track in tracks:
        x1 = track["x1"]
        y1 = track["y1"]
        x2 = track["x2"]
        y2 = track["y2"]
        track_id = track["id"]
        clase = track["clase"]
        score = track["score"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        texto = f"{clase}#{track_id} ({score:.2f})"
        cv2.putText(
            frame,
            texto,
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
