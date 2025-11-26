from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Dict, List, Optional

import cv2
import numpy as np

from sussy.core.deteccion import Detection


@dataclass
class MotionCandidate:
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float
    frames_alive: int = 1
    frames_without_match: int = 0
    start_cx: float = 0.0
    start_cy: float = 0.0

    def __post_init__(self) -> None:
        self.start_cx = self.cx
        self.start_cy = self.cy


class MotionDetector:
    """
    Detector de movimiento con memoria de varios frames + filtro por contraste.

    Flujo:
    - Calcula diferencia entre frame anterior y actual (para ver qué cambia).
    - Encuentra blobs pequeños que cambian.
    - Para cada blob:
        * Filtra por tamaño (área y lado máximo).
        * Filtra por contraste local (más oscuro o más claro que el entorno).
    - Empareja blobs con candidatos anteriores (mini-tracker).
    - Solo devuelve detecciones cuando un candidato:
        * ha vivido varios frames
        * y se ha desplazado suficiente
      (movimiento claro, aunque sea lento).
    """

    def __init__(
        self,
        umbral_diff: int = 25,         # más sensible a cambios suaves
        min_area: int = 6,             # acepta blobs muy pequeños
        max_area: int = 1500,
        max_lado: int = 60,
        contraste_min: float = 8.0,    # diferencia media vs entorno
        max_dist_match: int = 30,
        min_frames_alive: int = 3,
        min_desplazamiento: int = 6,   # permite movimiento lento
        max_frames_sin_match: int = 60,
    ) -> None:
        self.umbral_diff = umbral_diff
        self.min_area = min_area
        self.max_area = max_area
        self.max_lado = max_lado
        self.contraste_min = contraste_min
        self.max_dist_match = max_dist_match
        self.min_frames_alive = min_frames_alive
        self.min_desplazamiento = min_desplazamiento
        self.max_frames_sin_match = max_frames_sin_match

        self._prev_gray: Optional[np.ndarray] = None
        self._candidates: Dict[int, MotionCandidate] = {}
        self._next_id: int = 1

    def _extraer_blobs(
        self,
        gray_prev: np.ndarray,
        gray_act: np.ndarray,
    ) -> List[tuple[int, int, int, int]]:
        """
        Encuentra blobs de movimiento entre gray_prev y gray_act,
        filtrando por tamaño y contraste local.
        """
        diff = cv2.absdiff(gray_prev, gray_act)
        _, thresh = cv2.threshold(diff, self.umbral_diff, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        # limpiar ruido fino
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs: List[tuple[int, int, int, int]] = []

        alto, ancho = gray_act.shape

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # Filtro por tamaño
            if area < self.min_area or area > self.max_area:
                continue
            if max(w, h) > self.max_lado:
                continue

            # Filtro por contraste local
            # ROI del blob
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(ancho, x + w)
            y1 = min(alto, y + h)

            roi = gray_act[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            mean_roi = float(roi.mean())
            sum_roi = float(roi.sum())
            size_roi = roi.size

            # Región de contexto alrededor del blob
            pad = 8
            cx0 = max(0, x0 - pad)
            cy0 = max(0, y0 - pad)
            cx1 = min(ancho, x1 + pad)
            cy1 = min(alto, y1 + pad)

            region = gray_act[cy0:cy1, cx0:cx1]
            if region.size == 0:
                continue

            sum_region = float(region.sum())
            size_region = region.size

            if size_region <= size_roi:
                mean_context = float(region.mean())
            else:
                # Intento de "restar" el ROI del contexto
                sum_context = sum_region - sum_roi
                size_context = size_region - size_roi
                if size_context <= 0:
                    mean_context = float(region.mean())
                else:
                    mean_context = sum_context / size_context

            contraste = abs(mean_roi - mean_context)

            if contraste < self.contraste_min:
                # Poco contraste con el entorno → probablemente ruido
                continue

            blobs.append((x, y, w, h))

        return blobs

    def _actualizar_candidatos(self, blobs: List[tuple[int, int, int, int]]) -> None:
        usados: set[int] = set()

        # Match blobs con candidatos existentes
        for x, y, w, h in blobs:
            cx = x + w / 2.0
            cy = y + h / 2.0

            mejor_id: Optional[int] = None
            mejor_dist: float = float("inf")

            for cid, cand in self._candidates.items():
                if cid in usados:
                    continue
                dist = hypot(cx - cand.cx, cy - cand.cy)
                if dist < self.max_dist_match and dist < mejor_dist:
                    mejor_dist = dist
                    mejor_id = cid

            if mejor_id is not None:
                # Actualizar candidato existente
                cand = self._candidates[mejor_id]
                cand.x1 = x
                cand.y1 = y
                cand.x2 = x + w
                cand.y2 = y + h
                cand.cx = cx
                cand.cy = cy
                cand.frames_alive += 1
                cand.frames_without_match = 0
                usados.add(mejor_id)
            else:
                # Crear candidato nuevo
                cid = self._next_id
                self._next_id += 1
                self._candidates[cid] = MotionCandidate(
                    id=cid,
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    cx=cx,
                    cy=cy,
                )
                usados.add(cid)

        # Incrementar frames_without_match de los no usados
        para_borrar: List[int] = []
        for cid, cand in self._candidates.items():
            if cid not in usados:
                cand.frames_without_match += 1
                if cand.frames_without_match > self.max_frames_sin_match:
                    para_borrar.append(cid)

        for cid in para_borrar:
            del self._candidates[cid]

    def actualizar(self, frame: np.ndarray) -> List[Detection]:
        gray_act = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_act = cv2.GaussianBlur(gray_act, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray_act
            return []

        blobs = self._extraer_blobs(self._prev_gray, gray_act)
        self._actualizar_candidatos(blobs)
        self._prev_gray = gray_act

        detecciones: List[Detection] = []

        for cand in self._candidates.values():
            # Tiene que haber vivido cierto nº de frames
            if cand.frames_alive < self.min_frames_alive:
                continue

            # Desplazamiento total desde el inicio
            desplazamiento = hypot(cand.cx - cand.start_cx, cand.cy - cand.start_cy)

            # Si nunca se ha desplazado lo mínimo, lo ignoramos
            if desplazamiento < self.min_desplazamiento:
                continue

            det: Detection = {
                "x1": cand.x1,
                "y1": cand.y1,
                "x2": cand.x2,
                "y2": cand.y2,
                "clase": "movimiento",
                "score": 0.5,
            }
            detecciones.append(det)

def detectar_estaticos(
    frame,
    min_area: int = 6,
    max_area: int = 800,
    max_lado: int = 60,
    contraste_min: float = 12.0,
    textura_max: float = 12.0,
    max_estaticos: int = 3,
):
    """
    Busca puntos pequeños de alto contraste (posibles objetos estáticos),
    pero:
      - Filtra por tamaño.
      - Filtra por contraste local.
      - Filtra por textura del fondo (preferimos fondos lisos como cielo).
      - Se queda SOLO con los max_estaticos más contrastados.

    Resultado: pocos candidatos, y la mayoría en zonas tipo cielo.
    """
    import numpy as np
    import cv2
    from sussy.core.deteccion import Detection  # por si cambia algo arriba

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    alto, ancho = gray.shape

    # Top-hat (cosas pequeñas más claras) y black-hat (más oscuras)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    open_img = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    close_img = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    tophat = cv2.subtract(gray, open_img)
    blackhat = cv2.subtract(close_img, gray)

    response = cv2.max(tophat, blackhat)

    # Umbral para sacar puntos fuertes
    _, mask = cv2.threshold(response, 20, 255, cv2.THRESH_BINARY)

    kernel2 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2, iterations=1)
    mask = cv2.dilate(mask, kernel2, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos_info = []  # (contraste, x0, y0, x1, y1)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or area > max_area:
            continue
        if max(w, h) > max_lado:
            continue

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(ancho, x + w)
        y1 = min(alto, y + h)

        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        mean_roi = float(roi.mean())
        sum_roi = float(roi.sum())
        size_roi = roi.size

        pad = 8
        cx0 = max(0, x0 - pad)
        cy0 = max(0, y0 - pad)
        cx1 = min(ancho, x1 + pad)
        cy1 = min(alto, y1 + pad)

        region = gray[cy0:cy1, cx0:cx1]
        if region.size == 0:
            continue

        sum_region = float(region.sum())
        size_region = region.size

        if size_region <= size_roi:
            mean_context = float(region.mean())
        else:
            sum_context = sum_region - sum_roi
            size_context = size_region - size_roi
            if size_context <= 0:
                mean_context = float(region.mean())
            else:
                mean_context = sum_context / size_context

        contraste = abs(mean_roi - mean_context)
        if contraste < contraste_min:
            continue

        # Medimos textura del fondo: si es muy rugoso, descartamos
        # (no queremos bordes de edificios/árboles)
        lap = cv2.Laplacian(region, cv2.CV_64F)
        textura = lap.var()
        if textura > textura_max:
            continue

        candidatos_info.append((contraste, x0, y0, x1, y1))

    # Ordenar por contraste descendente y quedarnos con los mejores
    candidatos_info.sort(key=lambda c: c[0], reverse=True)

    detecciones: list[Detection] = []

    for contraste, x0, y0, x1, y1 in candidatos_info[:max_estaticos]:
        det: Detection = {
            "x1": int(x0),
            "y1": int(y0),
            "x2": int(x1),
            "y2": int(y1),
            "clase": "estatico",
            "score": float(contraste),  # guardamos el contraste como "score"
        }
        detecciones.append(det)

    return detecciones

