import csv
import os
from typing import List, Dict, Any

class RegistradorCSV:
    def __init__(self, ruta: str):
        self.ruta = ruta
        self.archivo = open(ruta, mode='w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.archivo)
        # Cabecera
        self.writer.writerow(["frame", "id", "x1", "y1", "x2", "y2", "clase", "score"])

    def registrar(self, frame_idx: int, tracks: List[Dict[str, Any]]) -> None:
        for t in tracks:
            x1, y1, x2, y2 = t['box']
            self.writer.writerow([
                frame_idx,
                t['id'],
                x1, y1, x2, y2,
                t.get('clase', ''),
                t.get('score', 0.0)
            ])

    def cerrar(self):
        if self.archivo:
            self.archivo.close()
