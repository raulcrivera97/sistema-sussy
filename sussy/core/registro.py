import csv
from pathlib import Path
from typing import List, Optional

from sussy.core.seguimiento import Track


class RegistradorCSV:
    """
    Registra los tracks frame a frame en un CSV.

    Formato columnas:
    frame,id,clase,score,x1,y1,x2,y2
    """

    def __init__(self, ruta_csv: str) -> None:
        # Aseguramos que la ruta tiene extensión .csv
        path = Path(ruta_csv)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        self._ruta = path
        self._f = open(self._ruta, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)

        # Cabecera
        self._writer.writerow(["frame", "id", "clase", "score", "x1", "y1", "x2", "y2"])

    def registrar(self, frame_idx: int, tracks: List[Track]) -> None:
        """
        Registra todos los tracks de un frame.
        """
        for t in tracks:
            self._writer.writerow(
                [
                    frame_idx,
                    t["id"],
                    t["clase"],
                    f"{t['score']:.4f}",
                    t["x1"],
                    t["y1"],
                    t["x2"],
                    t["y2"],
                ]
            )

    def cerrar(self) -> None:
        if not self._f.closed:
            self._f.close()

    @property
    def ruta(self) -> Path:
        return self._ruta
