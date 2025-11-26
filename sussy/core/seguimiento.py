from typing import TypedDict, List, Dict, Any
import math

from sussy.core.deteccion import Detection


class Track(TypedDict):
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    clase: str
    score: float

def _misma_categoria(clase1: str, clase2: str) -> bool:
    """
    Considera ciertas clases como la misma categoría a efectos de tracking.
    Por ejemplo, 'bird' y 'airplane' las tratamos como 'aéreo'.
    """
    aereas = {"bird", "airplane"}

    if clase1 in aereas and clase2 in aereas:
        return True

    return clase1 == clase2


def _centro(box: Detection | Track) -> tuple[float, float]:
    """
    Devuelve el centro (cx, cy) de una caja.
    """
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    return cx, cy


class TrackerSimple:
    """
    Tracker sencillo por centroides.

    Lógica:
    - Mantiene una lista de tracks activos.
    - Cada frame:
        - Para cada detección, busca el track cercano de la MISMA clase.
        - Si la distancia de centros < max_dist -> se asocia al mismo ID.
        - Si no hay track cercano -> crea un track nuevo.
        - Los tracks que no se actualizan suman 'missed'; si pasan de max_missed, se eliminan.
    """

    def __init__(self, max_dist: float = 80.0, max_missed: int = 10) -> None:
        self._max_dist = max_dist
        self._max_missed = max_missed
        # Cada track interno es un dict con las claves de Track + 'missed'
        self._tracks: List[Dict[str, Any]] = []
        self._next_id: int = 1

    def _crear_track(self, det: Detection) -> None:
        track: Dict[str, Any] = {
            "id": self._next_id,
            "x1": det["x1"],
            "y1": det["y1"],
            "x2": det["x2"],
            "y2": det["y2"],
            "clase": det["clase"],
            "score": det["score"],
            "missed": 0,
        }
        self._next_id += 1
        self._tracks.append(track)

    def actualizar(self, detecciones: List[Detection]) -> List[Track]:
        # Si no hay tracks activos, creamos uno por cada detección
        if not self._tracks:
            for det in detecciones:
                self._crear_track(det)
        else:
            # Marcamos todos como no actualizados
            for t in self._tracks:
                t["updated"] = False

            usados_tracks: set[int] = set()
            usados_dets: set[int] = set()

            # Asignación greedy por distancia de centro
            for det_idx, det in enumerate(detecciones):
                mejor_dist = float("inf")
                mejor_track_idx = -1

                cx_det, cy_det = _centro(det)

                for track_idx, track in enumerate(self._tracks):
                    if track_idx in usados_tracks:
                        continue

                    # Solo emparejamos si son de la misma categoría
                    # (por ejemplo, 'bird' y 'airplane' cuentan como la misma: "aéreo")
                    if not _misma_categoria(track["clase"], det["clase"]):
                        continue


                    cx_tr, cy_tr = _centro(track)
                    dist = math.hypot(cx_det - cx_tr, cy_det - cy_tr)

                    if dist < mejor_dist:
                        mejor_dist = dist
                        mejor_track_idx = track_idx

                # ¿Hay track cercano suficiente?
                if mejor_track_idx >= 0 and mejor_dist <= self._max_dist:
                    track = self._tracks[mejor_track_idx]
                    track["x1"] = det["x1"]
                    track["y1"] = det["y1"]
                    track["x2"] = det["x2"]
                    track["y2"] = det["y2"]
                    track["clase"] = det["clase"]
                    track["score"] = det["score"]
                    track["missed"] = 0
                    track["updated"] = True

                    usados_tracks.add(mejor_track_idx)
                    usados_dets.add(det_idx)

            # Detecciones no emparejadas -> nuevos tracks
            for det_idx, det in enumerate(detecciones):
                if det_idx not in usados_dets:
                    self._crear_track(det)

            # Tracks no actualizados -> missed +1
            tracks_vivos: List[Dict[str, Any]] = []
            for track in self._tracks:
                if not track.get("updated", False):
                    track["missed"] += 1
                if track["missed"] <= self._max_missed:
                    tracks_vivos.append(track)

            self._tracks = tracks_vivos

        # Preparamos salida limpia (sin campos internos)
        salida: List[Track] = []
        for t in self._tracks:
            salida.append(
                Track(
                    id=t["id"],
                    x1=t["x1"],
                    y1=t["y1"],
                    x2=t["x2"],
                    y2=t["y2"],
                    clase=t["clase"],
                    score=t["score"],
                )
            )
        return salida
