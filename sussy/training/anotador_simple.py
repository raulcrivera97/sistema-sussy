"""
Anotador Simple con OpenCV para Dataset Sussy.

Para pruebas rápidas de anotación. Para datasets serios,
usa Label Studio o CVAT.

Uso:
    python -m sussy.training.anotador_simple carpeta_imagenes/ salida/ --clases drone vehicle person

Controles:
    - Click izquierdo: primer punto del bounding box
    - Click derecho: segundo punto (confirma box)
    - 1-9: Seleccionar clase
    - N: Siguiente imagen
    - P: Imagen anterior
    - D: Borrar última anotación
    - S: Guardar anotaciones actuales
    - Q: Salir (guarda automáticamente)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class AnotadorSimple:
    """Anotador básico de bounding boxes con OpenCV."""
    
    def __init__(
        self,
        path_imagenes: Path,
        path_salida: Path,
        clases: List[str],
        cargar_existentes: bool = True,
    ):
        self.path_imagenes = Path(path_imagenes)
        self.path_salida = Path(path_salida)
        self.clases = clases
        
        # Crear carpetas de salida
        (self.path_salida / "images").mkdir(parents=True, exist_ok=True)
        (self.path_salida / "labels").mkdir(parents=True, exist_ok=True)
        
        # Buscar imágenes
        self.imagenes = sorted([
            p for p in self.path_imagenes.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        ])
        
        if not self.imagenes:
            raise ValueError(f"No se encontraron imágenes en {path_imagenes}")
        
        print(f"📸 {len(self.imagenes)} imágenes encontradas")
        print(f"🏷️  Clases: {clases}")
        
        # Estado
        self.idx_actual = 0
        self.clase_actual = 0
        self.punto_inicio: Optional[Tuple[int, int]] = None
        self.anotaciones: Dict[str, List[dict]] = {}
        
        # Cargar anotaciones existentes
        if cargar_existentes:
            self._cargar_anotaciones_existentes()
        
        # Colores para cada clase
        np.random.seed(42)
        self.colores = [
            tuple(map(int, c))
            for c in np.random.randint(100, 255, (len(clases), 3))
        ]
    
    def _cargar_anotaciones_existentes(self):
        """Carga anotaciones YOLO existentes."""
        for img_path in self.imagenes:
            nombre = img_path.stem
            label_path = self.path_salida / "labels" / f"{nombre}.txt"
            
            if label_path.exists():
                boxes = []
                with open(label_path) as f:
                    for linea in f:
                        partes = linea.strip().split()
                        if len(partes) >= 5:
                            boxes.append({
                                "clase": int(partes[0]),
                                "cx": float(partes[1]),
                                "cy": float(partes[2]),
                                "w": float(partes[3]),
                                "h": float(partes[4]),
                            })
                
                if boxes:
                    self.anotaciones[nombre] = boxes
        
        n_anotadas = len(self.anotaciones)
        if n_anotadas:
            print(f"[OK] Cargadas {n_anotadas} imagenes con anotaciones previas")
    
    def ejecutar(self):
        """Bucle principal de anotación."""
        cv2.namedWindow("Anotador Sussy", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Anotador Sussy", self._mouse_callback)
        
        while True:
            img_path = self.imagenes[self.idx_actual]
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"[ERROR] No se pudo cargar: {img_path}")
                self.idx_actual = (self.idx_actual + 1) % len(self.imagenes)
                continue
            
            self.img_actual = img.copy()
            self.h, self.w = img.shape[:2]
            
            # Dibujar imagen con anotaciones
            display = self._dibujar_anotaciones(img)
            
            # HUD
            display = self._dibujar_hud(display, img_path.name)
            
            cv2.imshow("Anotador Sussy", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self._guardar_todas()
                break
            elif key == ord('n'):
                self._guardar_actual()
                self.idx_actual = (self.idx_actual + 1) % len(self.imagenes)
                self.punto_inicio = None
            elif key == ord('p'):
                self._guardar_actual()
                self.idx_actual = (self.idx_actual - 1) % len(self.imagenes)
                self.punto_inicio = None
            elif key == ord('d'):
                self._borrar_ultima()
            elif key == ord('s'):
                self._guardar_actual()
                print(f"💾 Guardado: {img_path.name}")
            elif ord('1') <= key <= ord('9'):
                clase_idx = key - ord('1')
                if clase_idx < len(self.clases):
                    self.clase_actual = clase_idx
                    print(f"🏷️  Clase: {self.clases[self.clase_actual]}")
        
        cv2.destroyAllWindows()
        print("\n[OK] Anotacion completada")
        print(f"   {len(self.anotaciones)} imágenes anotadas")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del ratón."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Primer punto
            self.punto_inicio = (x, y)
        
        elif event == cv2.EVENT_RBUTTONDOWN and self.punto_inicio:
            # Segundo punto - crear box
            x1, y1 = self.punto_inicio
            x2, y2 = x, y
            
            # Normalizar a formato YOLO
            cx = ((x1 + x2) / 2) / self.w
            cy = ((y1 + y2) / 2) / self.h
            w = abs(x2 - x1) / self.w
            h = abs(y2 - y1) / self.h
            
            # Validar tamaño mínimo
            if w > 0.01 and h > 0.01:
                nombre = self.imagenes[self.idx_actual].stem
                
                if nombre not in self.anotaciones:
                    self.anotaciones[nombre] = []
                
                self.anotaciones[nombre].append({
                    "clase": self.clase_actual,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                })
                
                print(f"   + {self.clases[self.clase_actual]} ({len(self.anotaciones[nombre])} boxes)")
            
            self.punto_inicio = None
    
    def _dibujar_anotaciones(self, img: np.ndarray) -> np.ndarray:
        """Dibuja las anotaciones sobre la imagen."""
        display = img.copy()
        nombre = self.imagenes[self.idx_actual].stem
        
        # Dibujar boxes guardados
        if nombre in self.anotaciones:
            for box in self.anotaciones[nombre]:
                color = self.colores[box["clase"] % len(self.colores)]
                
                # Denormalizar
                cx, cy = box["cx"] * self.w, box["cy"] * self.h
                w, h = box["w"] * self.w, box["h"] * self.h
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)
                
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = self.clases[box["clase"]]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(display, label, (x1, y1 - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Dibujar box en progreso
        if self.punto_inicio:
            cv2.circle(display, self.punto_inicio, 5, (0, 255, 0), -1)
        
        return display
    
    def _dibujar_hud(self, img: np.ndarray, nombre_img: str) -> np.ndarray:
        """Dibuja información en pantalla."""
        display = img.copy()
        h, w = display.shape[:2]
        
        # Panel superior
        cv2.rectangle(display, (0, 0), (w, 60), (40, 40, 40), -1)
        
        # Texto
        info = f"[{self.idx_actual + 1}/{len(self.imagenes)}] {nombre_img}"
        cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        clase_text = f"Clase: [{self.clase_actual + 1}] {self.clases[self.clase_actual]}"
        color = self.colores[self.clase_actual]
        cv2.putText(display, clase_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Controles
        controles = "N:Sig  P:Ant  1-9:Clase  D:Borrar  S:Guardar  Q:Salir"
        cv2.putText(display, controles, (w - 450, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Número de boxes
        nombre = self.imagenes[self.idx_actual].stem
        n_boxes = len(self.anotaciones.get(nombre, []))
        cv2.putText(display, f"Boxes: {n_boxes}", (w - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return display
    
    def _borrar_ultima(self):
        """Borra la última anotación de la imagen actual."""
        nombre = self.imagenes[self.idx_actual].stem
        
        if nombre in self.anotaciones and self.anotaciones[nombre]:
            eliminada = self.anotaciones[nombre].pop()
            clase = self.clases[eliminada["clase"]]
            print(f"   - Eliminado: {clase}")
            
            if not self.anotaciones[nombre]:
                del self.anotaciones[nombre]
    
    def _guardar_actual(self):
        """Guarda las anotaciones de la imagen actual."""
        img_path = self.imagenes[self.idx_actual]
        nombre = img_path.stem
        
        # Copiar imagen si no existe
        img_dest = self.path_salida / "images" / img_path.name
        if not img_dest.exists():
            import shutil
            shutil.copy2(img_path, img_dest)
        
        # Guardar labels
        label_path = self.path_salida / "labels" / f"{nombre}.txt"
        
        if nombre in self.anotaciones and self.anotaciones[nombre]:
            lineas = []
            for box in self.anotaciones[nombre]:
                linea = f"{box['clase']} {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}"
                lineas.append(linea)
            
            with open(label_path, "w") as f:
                f.write("\n".join(lineas))
        elif label_path.exists():
            # Borrar si no hay anotaciones
            label_path.unlink()
    
    def _guardar_todas(self):
        """Guarda todas las anotaciones."""
        for idx in range(len(self.imagenes)):
            self.idx_actual = idx
            self._guardar_actual()
        
        # Generar data.yaml
        data_yaml = {
            "path": str(self.path_salida.absolute()),
            "train": "images",
            "val": "images",
            "nc": len(self.clases),
            "names": self.clases,
        }
        
        yaml_path = self.path_salida / "data.yaml"
        import yaml
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        # Estadísticas
        total_boxes = sum(len(boxes) for boxes in self.anotaciones.values())
        print(f"\n📊 Estadísticas finales:")
        print(f"   Imágenes anotadas: {len(self.anotaciones)}")
        print(f"   Total boxes: {total_boxes}")
        print(f"   Guardado en: {self.path_salida}")


def main():
    parser = argparse.ArgumentParser(
        description="Anotador simple de bounding boxes con OpenCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controles:
  Click izquierdo   Primer punto del box
  Click derecho     Segundo punto (confirma)
  1-9               Seleccionar clase
  N                 Siguiente imagen
  P                 Imagen anterior
  D                 Borrar última anotación
  S                 Guardar actual
  Q                 Salir y guardar todo

Ejemplo:
  python -m sussy.training.anotador_simple frames/ dataset/ --clases drone vehicle person
        """
    )
    
    parser.add_argument(
        "imagenes",
        help="Carpeta con las imágenes a anotar"
    )
    
    parser.add_argument(
        "salida",
        help="Carpeta de salida para el dataset"
    )
    
    parser.add_argument(
        "--clases", "-c",
        nargs="+",
        default=["drone", "vehicle", "person", "bird", "aircraft"],
        help="Lista de clases a anotar"
    )
    
    args = parser.parse_args()
    
    try:
        anotador = AnotadorSimple(
            Path(args.imagenes),
            Path(args.salida),
            args.clases,
        )
        anotador.ejecutar()
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1
    except KeyboardInterrupt:
        print("\n[!] Cancelado")
        return 130
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
