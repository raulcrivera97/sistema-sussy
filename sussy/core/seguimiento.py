"""
Tracker mejorado con estabilización de clases y preparación para Re-ID.

Mejoras v3:
- Estabilización de clases mediante historial ponderado con decay temporal
- Re-ID básico mediante embeddings visuales (opcional)
- Predicción de posición basada en velocidad
- Mejor matching para objetos pequeños/rápidos
- Suavizado de velocidad (EMA)
"""

from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING
from collections import defaultdict

from sussy.core.utilidades_iou import calcular_iou

if TYPE_CHECKING:
    import numpy as np


class HistorialClases:
    """
    Mantiene historial de clasificaciones por track con votación ponderada.
    
    Cada nueva clasificación incrementa el peso de esa clase, mientras que
    las demás clases sufren un decay temporal. La clase "estable" es la
    de mayor peso acumulado.
    
    IMPORTANTE: Ciertas clases (como "movimiento") son temporales y no deben
    prevalecer en el historial sobre clases confirmadas por IA.
    """
    
    # Clases que NO deben ser estabilizadas (son zonas de atención, no objetos)
    CLASES_NO_ESTABILIZABLES = {"movimiento", "unknown", "ruido"}
    
    # Clases que solo YOLO puede asignar (no heurísticas)
    CLASES_SOLO_IA = {"drone", "bird", "airplane", "person", "car", "truck", "bus", "motorcycle", "bicycle"}
    
    def __init__(self, decay: float = 0.92, min_frames_estable: int = 3):
        """
        Args:
            decay: Factor de decay para clases no observadas (0-1).
                   Valores altos = más memoria, más estable.
                   Valores bajos = más reactivo a cambios.
            min_frames_estable: Frames mínimos antes de considerar estable.
        """
        self.decay = decay
        self.min_frames_estable = min_frames_estable
        self._historiales: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._frames_por_track: Dict[int, int] = defaultdict(int)
    
    def registrar(self, track_id: int, clase: str, score: float = 1.0) -> str:
        """
        Registra una clasificación y retorna la clase estabilizada.
        
        Args:
            track_id: ID del track
            clase: Clase detectada en este frame
            score: Confianza de la detección (se usa como peso)
        
        Returns:
            La clase "estable" (la de mayor peso acumulado)
        """
        historial = self._historiales[track_id]
        self._frames_por_track[track_id] += 1
        
        # Aplicar decay a todas las clases existentes
        for c in list(historial.keys()):
            historial[c] *= self.decay
            # Limpiar clases con peso despreciable
            if historial[c] < 0.01:
                del historial[c]
        
        # Si la clase es "movimiento" o similar, NO la añadimos al historial
        # pero seguimos retornando la clase estable si existe
        if clase in self.CLASES_NO_ESTABILIZABLES:
            # Si hay historial con clases IA, usar esa
            if historial:
                clases_ia = {c: w for c, w in historial.items() if c in self.CLASES_SOLO_IA}
                if clases_ia:
                    return max(clases_ia.keys(), key=lambda c: clases_ia[c])
                # Si no hay clases IA pero hay otras, devolver la más pesada
                return max(historial.keys(), key=lambda c: historial[c])
            # Sin historial, devolver la clase tal cual
            return clase
        
        # Para "posible_dron": solo registrar con peso reducido y NUNCA como clase dominante
        # si hay clases IA en el historial
        if clase == "posible_dron":
            # Si ya hay clases IA confirmadas, ignorar posible_dron
            clases_ia = {c: w for c, w in historial.items() if c in self.CLASES_SOLO_IA}
            if clases_ia:
                return max(clases_ia.keys(), key=lambda c: clases_ia[c])
            # Si no hay clases IA, registrar con peso muy reducido
            peso_nuevo = 0.3 * score  # Peso muy bajo para posible_dron
            historial[clase] += peso_nuevo
        else:
            # Incrementar peso de la clase actual (ponderado por score)
            peso_nuevo = 1.0 + (score - 0.5) * 0.5  # score 0.5->1.0, score 1.0->1.25
            historial[clase] += peso_nuevo
        
        # Determinar clase estable
        if not historial:
            return clase
        
        # Preferir clases confirmadas por IA sobre heurísticas
        clases_ia = {c: w for c, w in historial.items() if c in self.CLASES_SOLO_IA}
        if clases_ia:
            clase_estable = max(clases_ia.keys(), key=lambda c: clases_ia[c])
        else:
            clase_estable = max(historial.keys(), key=lambda c: historial[c])
        
        # Si el track es muy nuevo, usar la clase detectada directamente
        # (pero nunca "movimiento" si hay algo mejor)
        if self._frames_por_track[track_id] < self.min_frames_estable:
            if clase not in self.CLASES_NO_ESTABILIZABLES:
                return clase
            elif historial:
                return clase_estable
            return clase
        
        return clase_estable
    
    def obtener_distribucion(self, track_id: int) -> Dict[str, float]:
        """Retorna la distribución de pesos de clases para un track."""
        historial = self._historiales.get(track_id, {})
        total = sum(historial.values()) or 1.0
        return {c: v / total for c, v in historial.items()}
    
    def obtener_confianza_clase(self, track_id: int, clase: str) -> float:
        """Retorna la confianza (0-1) de que el track sea de esa clase."""
        historial = self._historiales.get(track_id, {})
        if not historial:
            return 0.5
        total = sum(historial.values())
        return historial.get(clase, 0.0) / total if total > 0 else 0.0
    
    def limpiar_track(self, track_id: int) -> None:
        """Elimina el historial de un track."""
        self._historiales.pop(track_id, None)
        self._frames_por_track.pop(track_id, None)
    
    def limpiar_tracks_inactivos(self, tracks_activos: List[int]) -> None:
        """Limpia tracks que ya no están activos."""
        tracks_activos_set = set(tracks_activos)
        for tid in list(self._historiales.keys()):
            if tid not in tracks_activos_set:
                self.limpiar_track(tid)


class GaleriaReID:
    """
    Galería de embeddings para Re-ID de tracks perdidos.
    
    Guarda los embeddings de tracks que se pierden para poder
    re-identificarlos cuando vuelven a aparecer.
    """
    
    def __init__(
        self,
        max_tracks_guardados: int = 50,
        max_edad_frames: int = 300,  # ~10 segundos a 30fps
        umbral_similitud: float = 0.65,
    ):
        """
        Args:
            max_tracks_guardados: Máximo de tracks en la galería
            max_edad_frames: Frames máximos antes de expirar un track guardado
            umbral_similitud: Similitud mínima para considerar Re-ID
        """
        self.max_tracks_guardados = max_tracks_guardados
        self.max_edad_frames = max_edad_frames
        self.umbral_similitud = umbral_similitud
        
        # {track_id_original: {"embedding": np.ndarray, "clase": str, "edad": int, "metadata": dict}}
        self._galeria: Dict[int, Dict[str, Any]] = {}
        self._frame_actual = 0
    
    def tick(self) -> None:
        """Incrementa el contador de frames y limpia entradas expiradas."""
        self._frame_actual += 1
        
        # Limpiar entradas expiradas
        expirados = [
            tid for tid, data in self._galeria.items()
            if (self._frame_actual - data.get("frame_guardado", 0)) > self.max_edad_frames
        ]
        for tid in expirados:
            del self._galeria[tid]
    
    def guardar_track(
        self,
        track_id: int,
        embedding: "np.ndarray",
        clase: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Guarda un track en la galería para futura re-identificación.
        
        Args:
            track_id: ID original del track
            embedding: Vector de features visuales
            clase: Última clase conocida
            metadata: Información adicional (velocidad, tamaño, etc.)
        """
        # Si la galería está llena, eliminar el más antiguo
        if len(self._galeria) >= self.max_tracks_guardados:
            mas_antiguo = min(
                self._galeria.keys(),
                key=lambda tid: self._galeria[tid].get("frame_guardado", 0)
            )
            del self._galeria[mas_antiguo]
        
        self._galeria[track_id] = {
            "embedding": embedding,
            "clase": clase,
            "frame_guardado": self._frame_actual,
            "metadata": metadata or {},
        }
    
    def buscar_coincidencia(
        self,
        embedding: "np.ndarray",
        clase_detectada: str,
        func_similitud: Callable[["np.ndarray", "np.ndarray"], float],
    ) -> Optional[int]:
        """
        Busca en la galería un track que coincida con el embedding dado.
        
        Args:
            embedding: Embedding de la nueva detección
            clase_detectada: Clase de la nueva detección (para filtrar)
            func_similitud: Función que calcula similitud entre embeddings
        
        Returns:
            ID del track original si hay coincidencia, None si no.
        """
        if not self._galeria:
            return None
        
        mejor_tid = None
        mejor_sim = -1.0
        
        for tid, data in self._galeria.items():
            # Bonus si la clase coincide
            bonus_clase = 0.1 if data["clase"] == clase_detectada else 0.0
            
            sim = func_similitud(embedding, data["embedding"]) + bonus_clase
            
            if sim > self.umbral_similitud and sim > mejor_sim:
                mejor_sim = sim
                mejor_tid = tid
        
        # Si encontramos match, eliminarlo de la galería
        if mejor_tid is not None:
            del self._galeria[mejor_tid]
        
        return mejor_tid
    
    def eliminar_track(self, track_id: int) -> None:
        """Elimina un track de la galería."""
        self._galeria.pop(track_id, None)
    
    @property
    def num_guardados(self) -> int:
        """Número de tracks actualmente en la galería."""
        return len(self._galeria)


class TrackerSimple:
    """
    Tracker mejorado que asocia detecciones por IoU (prioridad) y distancia (fallback).
    Mantiene tracks perdidos durante 'max_frames_lost' frames para evitar parpadeos.
    
    Mejoras v3:
    - Estabilización de clases mediante historial ponderado
    - Re-ID opcional mediante embeddings visuales
    - Predicción de posición basada en velocidad
    - Mejor matching para objetos pequeños/rápidos
    - Suavizado de velocidad (EMA)
    """
    
    def __init__(
        self,
        max_dist: int = 100,
        max_frames_lost: int = 10,
        iou_threshold: float = 0.3,
        usar_prediccion: bool = True,
        prediccion_factor: float = 0.8,
        aceleracion_max: float = 5.0,
        # Nuevos parámetros v3
        usar_estabilizacion_clases: bool = True,
        clase_decay: float = 0.92,
        clase_min_frames: int = 3,
        usar_reid: bool = False,
        reid_max_edad: int = 300,
        reid_umbral: float = 0.65,
    ):
        self.max_dist = max_dist
        self.max_frames_lost = max_frames_lost
        self.iou_threshold = iou_threshold
        self.usar_prediccion = usar_prediccion
        self.prediccion_factor = prediccion_factor
        self.aceleracion_max = aceleracion_max
        
        # Estabilización de clases
        self.usar_estabilizacion_clases = usar_estabilizacion_clases
        self.historial_clases = HistorialClases(
            decay=clase_decay,
            min_frames_estable=clase_min_frames,
        ) if usar_estabilizacion_clases else None
        
        # Re-ID
        self.usar_reid = usar_reid
        self.galeria_reid = GaleriaReID(
            max_edad_frames=reid_max_edad,
            umbral_similitud=reid_umbral,
        ) if usar_reid else None
        self._extractor_apariencia: Optional[Any] = None
        self._frame_actual: Optional["np.ndarray"] = None
        
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 1

    def set_extractor_apariencia(self, extractor: Any) -> None:
        """
        Configura el extractor de embeddings para Re-ID.
        
        Args:
            extractor: Objeto con método extraer(frame, bbox) -> np.ndarray
        """
        self._extractor_apariencia = extractor

    def _predecir_posicion(self, track: Dict) -> tuple:
        """
        Predice la posición del centroide basándose en la velocidad.
        Retorna (cx_pred, cy_pred).
        """
        cx = track.get('cx', 0)
        cy = track.get('cy', 0)
        
        if not self.usar_prediccion:
            return cx, cy
        
        vel_x = track.get('vel_x', 0) * self.prediccion_factor
        vel_y = track.get('vel_y', 0) * self.prediccion_factor
        
        return cx + vel_x, cy + vel_y

    def _calcular_distancia_predicha(self, det: Dict, track: Dict) -> float:
        """
        Calcula distancia entre detección y posición predicha del track.
        """
        cx_det = (det['x1'] + det['x2']) / 2
        cy_det = (det['y1'] + det['y2']) / 2
        
        cx_pred, cy_pred = self._predecir_posicion(track)
        
        return ((cx_det - cx_pred)**2 + (cy_det - cy_pred)**2)**0.5

    def _extraer_embedding(self, det: Dict) -> Optional["np.ndarray"]:
        """Extrae embedding visual de una detección si hay extractor configurado."""
        if not self._extractor_apariencia or self._frame_actual is None:
            return None
        
        try:
            bbox = [det['x1'], det['y1'], det['x2'], det['y2']]
            return self._extractor_apariencia.extraer(self._frame_actual, bbox)
        except Exception:
            return None

    def _intentar_reid(self, det: Dict) -> Optional[int]:
        """
        Intenta re-identificar una detección usando la galería de Re-ID.
        
        Returns:
            ID del track original si hay match, None si no.
        """
        if not self.usar_reid or not self.galeria_reid or not self._extractor_apariencia:
            return None
        
        embedding = self._extraer_embedding(det)
        if embedding is None:
            return None
        
        return self.galeria_reid.buscar_coincidencia(
            embedding,
            det.get('clase', 'unknown'),
            self._extractor_apariencia.similitud,
        )

    def actualizar(
        self,
        detecciones: List[Dict[str, Any]],
        frame: Optional["np.ndarray"] = None,
    ) -> List[Dict[str, Any]]:
        """
        Actualiza los tracks con las nuevas detecciones.
        
        Args:
            detecciones: Lista de detecciones del frame actual
            frame: Frame actual (opcional, necesario para Re-ID)
        
        Returns:
            Lista de tracks activos con información estabilizada
        """
        self._frame_actual = frame
        
        # Tick de la galería Re-ID
        if self.galeria_reid:
            self.galeria_reid.tick()
        
        # 1. Asociación
        used_det_indices = set()
        used_track_ids = set()
        
        # --- PASO 1: Asociación por IoU (Objetos grandes/lentos) ---
        sorted_dets = sorted(enumerate(detecciones), key=lambda x: x[1]['score'], reverse=True)
        
        for idx, det in sorted_dets:
            best_tid = None
            best_iou = -1.0
            
            for tid, track in self.tracks.items():
                if tid in used_track_ids:
                    continue
                
                iou = calcular_iou(track['box'], det)
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid is not None:
                self._actualizar_track(best_tid, det)
                used_track_ids.add(best_tid)
                used_det_indices.add(idx)

        # --- PASO 2: Asociación por Distancia con Predicción ---
        for idx, det in sorted_dets:
            if idx in used_det_indices:
                continue
            
            best_tid = None
            best_dist = float('inf')

            for tid, track in self.tracks.items():
                if tid in used_track_ids:
                    continue
                
                dist = self._calcular_distancia_predicha(det, track)
                
                if dist < self.max_dist and dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                self._actualizar_track(best_tid, det)
                used_track_ids.add(best_tid)
                used_det_indices.add(idx)
            else:
                # --- PASO 3: Intentar Re-ID antes de crear nuevo track ---
                reid_tid = self._intentar_reid(det)
                
                if reid_tid is not None:
                    # ¡Re-identificado! Restaurar el ID original
                    self._crear_track(det, track_id=reid_tid)
                else:
                    # Nueva detección sin match
                    self._crear_track(det)
                
                # Marcar el nuevo track como usado
                new_tid = reid_tid if reid_tid else (self.next_id - 1)
                used_track_ids.add(new_tid)

        # --- PASO 4: Gestión de Tracks Perdidos ---
        final_tracks = []
        to_delete = []
        to_archive = []  # Para Re-ID
        
        for tid, track in self.tracks.items():
            if tid in used_track_ids:
                # Track actualizado en este frame
                clase_final = track['clase']
                score_clase = 1.0
                
                # Obtener distribución de clases para metadata
                distribucion = {}
                if self.historial_clases:
                    distribucion = self.historial_clases.obtener_distribucion(tid)
                    score_clase = self.historial_clases.obtener_confianza_clase(tid, clase_final)
                
                final_tracks.append({
                    "id": tid,
                    "box": track['box'],
                    "clase": clase_final,
                    "score": track['score'],
                    "score_clase": score_clase,
                    "distribucion_clases": distribucion,
                    "perdido": False,
                    "frames_vivos": track.get('frames_vivos', 0),
                    "velocidad": {
                        "x": track.get('vel_x', 0.0),
                        "y": track.get('vel_y', 0.0),
                    },
                })
            else:
                # Track perdido en este frame
                track['frames_lost'] += 1
                
                if track['frames_lost'] <= self.max_frames_lost:
                    self._actualizar_track_perdido(track)
                    
                    clase_final = track['clase']
                    if self.historial_clases:
                        score_clase = self.historial_clases.obtener_confianza_clase(tid, clase_final)
                    else:
                        score_clase = 1.0
                    
                    final_tracks.append({
                        "id": tid,
                        "box": track['box'],
                        "clase": clase_final,
                        "score": track['score'] * 0.95,
                        "score_clase": score_clase,
                        "perdido": True,
                        "frames_vivos": track.get('frames_vivos', 0),
                        "velocidad": {
                            "x": track.get('vel_x', 0.0),
                            "y": track.get('vel_y', 0.0),
                        },
                    })
                else:
                    # Track expirado - archivar para Re-ID si está habilitado
                    if self.usar_reid:
                        to_archive.append(tid)
                    to_delete.append(tid)
        
        # Archivar tracks para Re-ID antes de eliminarlos
        for tid in to_archive:
            self._archivar_para_reid(tid)
        
        # Eliminar tracks expirados
        for tid in to_delete:
            del self.tracks[tid]
            if self.historial_clases:
                self.historial_clases.limpiar_track(tid)

        return final_tracks

    def resetear(self) -> None:
        """Limpia todos los tracks y reinicia el estado."""
        self.tracks.clear()
        self.next_id = 1
        if self.historial_clases:
            self.historial_clases = HistorialClases(
                decay=self.historial_clases.decay,
                min_frames_estable=self.historial_clases.min_frames_estable,
            )
        if self.galeria_reid:
            self.galeria_reid = GaleriaReID(
                max_edad_frames=self.galeria_reid.max_edad_frames,
                umbral_similitud=self.galeria_reid.umbral_similitud,
            )

    def _actualizar_track(self, tid: int, det: Dict) -> None:
        """Actualiza un track existente con nueva detección."""
        track = self.tracks[tid]
        new_box = [det['x1'], det['y1'], det['x2'], det['y2']]
        new_cx = (new_box[0] + new_box[2]) / 2
        new_cy = (new_box[1] + new_box[3]) / 2
        prev_cx = track.get('cx', new_cx)
        prev_cy = track.get('cy', new_cy)

        # Calcular velocidad instantánea
        vel_x_nuevo = new_cx - prev_cx
        vel_y_nuevo = new_cy - prev_cy
        
        # Limitar aceleración
        vel_x_prev = track.get('vel_x', 0)
        vel_y_prev = track.get('vel_y', 0)
        
        delta_vx = vel_x_nuevo - vel_x_prev
        delta_vy = vel_y_nuevo - vel_y_prev
        
        if abs(delta_vx) > self.aceleracion_max:
            vel_x_nuevo = vel_x_prev + self.aceleracion_max * (1 if delta_vx > 0 else -1)
        if abs(delta_vy) > self.aceleracion_max:
            vel_y_nuevo = vel_y_prev + self.aceleracion_max * (1 if delta_vy > 0 else -1)
        
        # Suavizado EMA de velocidad
        alpha = 0.7
        track['vel_x'] = alpha * vel_x_nuevo + (1 - alpha) * vel_x_prev
        track['vel_y'] = alpha * vel_y_nuevo + (1 - alpha) * vel_y_prev
        
        track['cx'] = new_cx
        track['cy'] = new_cy
        track['box'] = new_box
        track['frames_lost'] = 0
        track['score'] = det['score']
        track['frames_vivos'] = track.get('frames_vivos', 0) + 1
        
        # Estabilización de clases
        clase_detectada = det['clase']
        if self.historial_clases:
            clase_estable = self.historial_clases.registrar(
                tid, clase_detectada, det['score']
            )
            track['clase'] = clase_estable
            track['clase_raw'] = clase_detectada  # Guardar la original también
        else:
            track['clase'] = clase_detectada
        
        # Actualizar embedding si hay extractor
        if self._extractor_apariencia and self._frame_actual is not None:
            embedding = self._extraer_embedding(det)
            if embedding is not None:
                track['embedding'] = embedding

    def _crear_track(self, det: Dict, track_id: Optional[int] = None) -> None:
        """
        Crea un nuevo track a partir de una detección.
        
        Args:
            det: Detección
            track_id: ID específico a usar (para Re-ID), o None para nuevo ID
        """
        if track_id is not None:
            new_id = track_id
            # Asegurar que next_id sea mayor
            if track_id >= self.next_id:
                self.next_id = track_id + 1
        else:
            new_id = self.next_id
            self.next_id += 1
        
        box = [det['x1'], det['y1'], det['x2'], det['y2']]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        
        clase = det['clase']
        if self.historial_clases:
            # Registrar primera observación
            clase = self.historial_clases.registrar(new_id, det['clase'], det['score'])
        
        self.tracks[new_id] = {
            "box": box,
            "frames_lost": 0,
            "frames_vivos": 1,
            "clase": clase,
            "clase_raw": det['clase'],
            "score": det['score'],
            "cx": cx,
            "cy": cy,
            "vel_x": 0.0,
            "vel_y": 0.0,
        }
        
        # Extraer embedding inicial si hay extractor
        if self._extractor_apariencia and self._frame_actual is not None:
            embedding = self._extraer_embedding(det)
            if embedding is not None:
                self.tracks[new_id]['embedding'] = embedding

    def _actualizar_track_perdido(self, track: Dict) -> None:
        """Actualiza posición de un track perdido usando predicción."""
        if self.usar_prediccion and track['frames_lost'] <= 3:
            cx_pred, cy_pred = self._predecir_posicion(track)
            track['cx'] = cx_pred
            track['cy'] = cy_pred
            
            w = track['box'][2] - track['box'][0]
            h = track['box'][3] - track['box'][1]
            track['box'] = [
                int(cx_pred - w/2),
                int(cy_pred - h/2),
                int(cx_pred + w/2),
                int(cy_pred + h/2),
            ]
            
            track['vel_x'] *= 0.9
            track['vel_y'] *= 0.9

    def _archivar_para_reid(self, tid: int) -> None:
        """Archiva un track en la galería de Re-ID antes de eliminarlo."""
        if not self.usar_reid or not self.galeria_reid:
            return
        
        track = self.tracks.get(tid)
        if not track:
            return
        
        embedding = track.get('embedding')
        if embedding is None:
            return
        
        metadata = {
            "vel_x": track.get('vel_x', 0),
            "vel_y": track.get('vel_y', 0),
            "frames_vivos": track.get('frames_vivos', 0),
            "box_size": (
                track['box'][2] - track['box'][0],
                track['box'][3] - track['box'][1],
            ),
        }
        
        self.galeria_reid.guardar_track(
            tid,
            embedding,
            track['clase'],
            metadata,
        )
