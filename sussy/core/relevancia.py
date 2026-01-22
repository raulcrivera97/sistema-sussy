"""
Evaluador de relevancia para objetos detectados por movimiento.
Incluye heurísticas avanzadas para distinguir drones de vegetación/ruido.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from sussy.config import Config
from sussy.core.deteccion import Detection


class HistorialTrayectoria:
    """Mantiene historial de posiciones para análisis de trayectoria."""
    
    def __init__(self, max_frames: int = 10):
        self.max_frames = max_frames
        self._historiales: Dict[int, Deque[Tuple[float, float]]] = {}
    
    def actualizar(self, track_id: int, cx: float, cy: float) -> None:
        """Añade posición al historial del track."""
        if track_id not in self._historiales:
            self._historiales[track_id] = deque(maxlen=self.max_frames)
        self._historiales[track_id].append((cx, cy))
    
    def obtener_linealidad(self, track_id: int) -> float:
        """
        Calcula cuán lineal es la trayectoria (0-1).
        1 = perfectamente recta, 0 = muy errática.
        """
        if track_id not in self._historiales:
            return 0.5
        
        puntos = list(self._historiales[track_id])
        if len(puntos) < 3:
            return 0.5
        
        # Distancia real recorrida
        dist_real = 0.0
        for i in range(1, len(puntos)):
            dx = puntos[i][0] - puntos[i-1][0]
            dy = puntos[i][1] - puntos[i-1][1]
            dist_real += (dx**2 + dy**2) ** 0.5
        
        # Distancia directa (inicio a fin)
        dx_total = puntos[-1][0] - puntos[0][0]
        dy_total = puntos[-1][1] - puntos[0][1]
        dist_directa = (dx_total**2 + dy_total**2) ** 0.5
        
        if dist_real < 1.0:
            return 0.5  # Sin movimiento significativo
        
        # Linealidad = dist_directa / dist_real
        return min(1.0, dist_directa / dist_real)
    
    def limpiar_track(self, track_id: int) -> None:
        """Elimina historial de un track."""
        self._historiales.pop(track_id, None)
    
    def limpiar_antiguos(self, tracks_activos: List[int]) -> None:
        """Limpia tracks que ya no existen."""
        ids_a_eliminar = [tid for tid in self._historiales if tid not in tracks_activos]
        for tid in ids_a_eliminar:
            del self._historiales[tid]


class ContadorClase:
    """Mantiene conteo de frames consecutivos por clase para cada track."""
    
    def __init__(self):
        self._conteos: Dict[int, Dict[str, int]] = {}
    
    def registrar(self, track_id: int, clase: str) -> int:
        """Registra clase y retorna frames consecutivos con esa clase."""
        if track_id not in self._conteos:
            self._conteos[track_id] = {}
        
        conteo = self._conteos[track_id]
        
        # Incrementar clase actual, resetear otras
        for c in list(conteo.keys()):
            if c != clase:
                conteo[c] = 0
        
        conteo[clase] = conteo.get(clase, 0) + 1
        return conteo[clase]
    
    def obtener_frames_como_clase(self, track_id: int, clase: str) -> int:
        """Retorna cuántos frames ha sido clasificado como esta clase."""
        if track_id not in self._conteos:
            return 0
        return self._conteos[track_id].get(clase, 0)
    
    def limpiar_antiguos(self, tracks_activos: List[int]) -> None:
        """Limpia tracks que ya no existen."""
        ids_a_eliminar = [tid for tid in self._conteos if tid not in tracks_activos]
        for tid in ids_a_eliminar:
            del self._conteos[tid]


class EvaluadorRelevancia:
    """
    Evalúa los objetos obtenidos del detector de movimiento y decide si
    merecen pasar al pipeline principal. Su enfoque es heurístico mejorado
    para distinguir drones de vegetación y ruido.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("sussy.relevancia")
        self.historial_trayectoria = HistorialTrayectoria(
            max_frames=getattr(Config, "DRON_TRAYECTORIA_FRAMES", 5)
        )
        self.contador_clase = ContadorClase()

    def filtrar(
        self,
        detecciones: List[Detection],
        frame_shape,
        tracks: Optional[List[Dict]] = None,
    ) -> List[Detection]:
        """
        Filtra detecciones aplicando heurísticas de relevancia.
        Ahora acepta tracks para análisis de trayectoria.
        """
        if not detecciones:
            return []

        alto, ancho = frame_shape[:2]
        area_total = max(1, ancho * alto)
        filtradas: List[Detection] = []

        # Actualizar historial de trayectorias si hay tracks
        if tracks:
            tracks_activos = []
            for t in tracks:
                tid = t.get("id", -1)
                if tid > 0:
                    tracks_activos.append(tid)
                    box = t.get("box", [0, 0, 0, 0])
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    self.historial_trayectoria.actualizar(tid, cx, cy)
            
            # Limpiar tracks antiguos
            self.historial_trayectoria.limpiar_antiguos(tracks_activos)
            self.contador_clase.limpiar_antiguos(tracks_activos)

        for det in detecciones:
            if det.get("clase") != "movimiento":
                det.setdefault("relevante", True)
                filtradas.append(det)
                continue

            evaluada = self._evaluar_movimiento(det, ancho, alto, area_total, tracks)
            if evaluada is None:
                continue  # Demasiado ruido, no lo propagamos
            filtradas.append(evaluada)

        return filtradas

    def _evaluar_movimiento(
        self,
        det: Detection,
        ancho_frame: int,
        alto_frame: int,
        area_total: int,
        tracks: Optional[List[Dict]] = None,
    ) -> Detection | None:
        """
        Evalúa si un blob de movimiento es relevante como ZONA DE ATENCIÓN.
        
        IMPORTANTE: Este método NUNCA clasifica como "posible_dron".
        Solo marca objetos en movimiento que YOLO no identificó como "movimiento"
        para indicar que hay algo interesante que no se pudo clasificar.
        """
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        area_px = det.get("area_px", w * h)
        area_rel = area_px / area_total
        velocidad = float(det.get("velocidad_px", 0.0))
        frames_vivos = int(det.get("frames_vivos", 0))

        margen_x = ancho_frame * Config.RELEVANCIA_BORDE_PCT
        margen_y = alto_frame * Config.RELEVANCIA_BORDE_PCT
        toca_borde = (
            x1 <= margen_x
            or y1 <= margen_y
            or x2 >= (ancho_frame - margen_x)
            or y2 >= (alto_frame - margen_y)
        )

        aspecto = w / max(1, h)

        # ========== REGLAS DE DESCARTE (ajustadas para no ser demasiado estrictas) ==========
        
        # Regla 0: descartar blobs extremadamente pequeños (ruido de píxeles)
        area_min_rel = getattr(Config, "RELEVANCIA_AREA_DRON_MIN", 0.0001)
        if area_rel < area_min_rel:
            return None
        
        # Regla 0b: descartar blobs con área absoluta muy pequeña
        # Reducido a 25 px para permitir objetos pequeños en el cielo
        area_min_px = getattr(Config, "MOVIMIENTO_AREA_MIN", 40)
        if area_px < area_min_px:
            return None

        # Regla 1: descartar oscilaciones grandes pero lentas (ramas, paredes)
        if area_rel >= Config.RELEVANCIA_AREA_RAMA_MIN and velocidad < Config.RELEVANCIA_VEL_MIN:
            return None

        # Regla 2: si está pegado al borde y apenas se desplaza, es ruido
        if toca_borde and velocidad < (Config.RELEVANCIA_VEL_MIN * 1.5):
            return None

        # Regla 3: formas extremadamente alargadas que no avanzan lo suficiente
        if (
            aspecto >= Config.RELEVANCIA_ASPECTO_RAMAS
            or aspecto <= (1 / Config.RELEVANCIA_ASPECTO_RAMAS)
        ) and velocidad < (Config.RELEVANCIA_VEL_MIN * 1.2) and area_rel > Config.RELEVANCIA_AREA_DRON_MAX:
            return None
        
        # Regla 4: blobs con muy poca persistencia no pasan
        # Reducido a 3 frames para ser menos estricto
        min_frames_relevante = getattr(Config, "MOVIMIENTO_MIN_FRAMES", 5)
        min_frames_mostrar = max(3, min_frames_relevante - 2)
        if frames_vivos < min_frames_mostrar:
            return None

        # ========== OBJETO NO IDENTIFICADO EN MOVIMIENTO ==========
        # Si llegamos aquí, el blob pasa los filtros básicos.
        # Lo marcamos como "movimiento" para indicar que hay algo interesante
        # que la IA no pudo identificar.
        
        det["clase"] = "movimiento"
        det["relevante"] = True
        # Score basado en características: más alto si tiene velocidad y persistencia
        base_score = 0.4
        if velocidad >= Config.RELEVANCIA_VEL_MIN:
            base_score += 0.1
        if frames_vivos >= min_frames_relevante:
            base_score += 0.1
        det["score"] = min(0.6, base_score)  # Cap en 0.6 para que sea claramente menor que detecciones IA
        det["descripcion"] = "Objeto en movimiento no identificado"
        
        return det

    def resetear(self) -> None:
        """Limpia todos los historiales."""
        self.historial_trayectoria = HistorialTrayectoria(
            max_frames=getattr(Config, "DRON_TRAYECTORIA_FRAMES", 5)
        )
        self.contador_clase = ContadorClase()