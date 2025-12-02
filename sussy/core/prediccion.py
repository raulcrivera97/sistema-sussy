from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class ZonaPredicha:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class PredictorMovimiento:
    """
    Genera zonas de interés predichas a partir de la velocidad de los tracks.
    La idea es ejecutar recortes específicos en la siguiente iteración para
    dar más contexto a la IA y no perder objetos rápidos o intermitentes.
    """

    def __init__(
        self,
        frames_adelante: int,
        padding_pct: float,
        vel_min: float,
        max_zonas: int,
    ) -> None:
        self.frames_adelante = max(1, int(frames_adelante))
        self.padding_pct = max(0.0, float(padding_pct))
        self.vel_min = max(0.0, float(vel_min))
        self.max_zonas = max(1, int(max_zonas))
        self._zonas_pendientes: List[ZonaPredicha] = []

    def consumir_zonas(self) -> List[ZonaPredicha]:
        zonas = self._zonas_pendientes
        self._zonas_pendientes = []
        return zonas

    def preparar_zonas(self, tracks: List[Dict], frame_shape) -> None:
        if not tracks:
            self._zonas_pendientes = []
            return

        alto, ancho = frame_shape[:2]
        zonas: List[ZonaPredicha] = []

        for track in tracks:
            if track.get("perdido"):
                continue

            vel = track.get("velocidad") or {}
            vx = float(vel.get("x", 0.0))
            vy = float(vel.get("y", 0.0))
            velocidad_mod = (vx ** 2 + vy ** 2) ** 0.5

            if velocidad_mod < self.vel_min:
                continue

            box = track.get("box")
            if not box:
                continue

            zona = self._predecir_box(box, vx, vy, alto, ancho)
            zonas.append(zona)

            if len(zonas) >= self.max_zonas:
                break

        self._zonas_pendientes = zonas

    def _predecir_box(
        self,
        box: List[int],
        vel_x: float,
        vel_y: float,
        alto: int,
        ancho: int,
    ) -> ZonaPredicha:
        x1, y1, x2, y2 = box
        width = max(4.0, float(x2 - x1))
        height = max(4.0, float(y2 - y1))

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Proyectamos varios frames hacia delante
        cx_pred = cx + vel_x * self.frames_adelante
        cy_pred = cy + vel_y * self.frames_adelante

        # Mantener tamaño original, pero con padding configurable
        half_w = width / 2.0
        half_h = height / 2.0

        padding_w = half_w * self.padding_pct
        padding_h = half_h * self.padding_pct

        x1_pred = int(round(cx_pred - half_w - padding_w))
        x2_pred = int(round(cx_pred + half_w + padding_w))
        y1_pred = int(round(cy_pred - half_h - padding_h))
        y2_pred = int(round(cy_pred + half_h + padding_h))

        # Clampear al frame
        x1_pred = max(0, min(ancho - 1, x1_pred))
        y1_pred = max(0, min(alto - 1, y1_pred))
        x2_pred = max(0, min(ancho, x2_pred))
        y2_pred = max(0, min(alto, y2_pred))

        # Garantizar que el box tiene al menos 2px
        if x2_pred <= x1_pred:
            x2_pred = min(ancho, x1_pred + 2)
        if y2_pred <= y1_pred:
            y2_pred = min(alto, y1_pred + 2)

        return ZonaPredicha(x1_pred, y1_pred, x2_pred, y2_pred)

