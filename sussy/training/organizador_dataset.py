"""
Organizador de Dataset para Entrenamiento de Sussy.

Estructura el dataset para diferentes fases de entrenamiento:
- Fase 1: Detección de objetos (YOLO format)
- Fase 2-3: Clasificación/Sub-clasificación (carpetas de crops)
- Fase 4: Atributos (multi-label classification)

También maneja:
- División train/val/test sin data leakage entre vídeos
- Generación de archivos YAML para YOLO
- Estadísticas del dataset
"""

import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

LOGGER = logging.getLogger("sussy.training.organizador")


@dataclass
class BoundingBox:
    """Bounding box con etiqueta."""
    x1: int
    y1: int
    x2: int
    y2: int
    clase: str
    subclase: Optional[str] = None
    atributos: Dict[str, Any] = field(default_factory=dict)
    confianza: float = 1.0
    
    @property
    def centro_x(self) -> float:
        return (self.x1 + self.x2) / 2
    
    @property
    def centro_y(self) -> float:
        return (self.y1 + self.y2) / 2
    
    @property
    def ancho(self) -> int:
        return self.x2 - self.x1
    
    @property
    def alto(self) -> int:
        return self.y2 - self.y1
    
    def to_yolo(self, img_width: int, img_height: int, clase_idx: int) -> str:
        """Convierte a formato YOLO: clase cx cy w h (normalizado)."""
        cx = self.centro_x / img_width
        cy = self.centro_y / img_height
        w = self.ancho / img_width
        h = self.alto / img_height
        return f"{clase_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


@dataclass
class ImagenAnotada:
    """Imagen con sus anotaciones."""
    path_imagen: Path
    boxes: List[BoundingBox] = field(default_factory=list)
    video_origen: Optional[str] = None
    metadatos: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tiene_anotaciones(self) -> bool:
        return len(self.boxes) > 0


class OrganizadorDataset:
    """
    Organiza y estructura el dataset para entrenamiento.
    
    Ejemplo de uso:
        org = OrganizadorDataset(Path("mi_dataset"))
        org.agregar_imagen_con_boxes("frame001.jpg", [box1, box2])
        org.generar_yolo(clases_detectar=["drone", "vehicle"])
    """
    
    def __init__(self, path_base: Path):
        self.path_base = Path(path_base)
        self.path_base.mkdir(parents=True, exist_ok=True)
        
        self.imagenes: Dict[str, ImagenAnotada] = {}
        self._videos_a_imagenes: Dict[str, List[str]] = defaultdict(list)
        
    def agregar_imagen(
        self,
        path_imagen: Path,
        video_origen: Optional[str] = None,
    ) -> str:
        """
        Registra una imagen (sin anotaciones todavía).
        
        Returns:
            ID único de la imagen
        """
        path_imagen = Path(path_imagen)
        img_id = path_imagen.stem
        
        if img_id in self.imagenes:
            return img_id
        
        self.imagenes[img_id] = ImagenAnotada(
            path_imagen=path_imagen,
            video_origen=video_origen,
        )
        
        if video_origen:
            self._videos_a_imagenes[video_origen].append(img_id)
        
        return img_id
    
    def agregar_imagen_con_boxes(
        self,
        path_imagen: Path,
        boxes: List[BoundingBox],
        video_origen: Optional[str] = None,
    ) -> str:
        """
        Registra una imagen con sus bounding boxes.
        
        Args:
            path_imagen: Ruta a la imagen
            boxes: Lista de bounding boxes
            video_origen: ID del vídeo de origen (para evitar data leakage)
        
        Returns:
            ID único de la imagen
        """
        img_id = self.agregar_imagen(path_imagen, video_origen)
        self.imagenes[img_id].boxes.extend(boxes)
        return img_id
    
    def importar_yolo(
        self,
        path_imagenes: Path,
        path_etiquetas: Path,
        clases: List[str],
        video_origen: Optional[str] = None,
    ) -> int:
        """
        Importa anotaciones en formato YOLO existente.
        
        Args:
            path_imagenes: Carpeta con imágenes
            path_etiquetas: Carpeta con archivos .txt
            clases: Lista de nombres de clases (en orden)
            video_origen: ID de vídeo origen
        
        Returns:
            Número de imágenes importadas
        """
        path_imagenes = Path(path_imagenes)
        path_etiquetas = Path(path_etiquetas)
        
        contador = 0
        
        for path_txt in path_etiquetas.glob("*.txt"):
            nombre_base = path_txt.stem
            
            # Buscar imagen correspondiente
            path_img = None
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidato = path_imagenes / f"{nombre_base}{ext}"
                if candidato.exists():
                    path_img = candidato
                    break
            
            if not path_img:
                continue
            
            # Leer dimensiones de imagen
            img = cv2.imread(str(path_img))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Leer anotaciones
            boxes = []
            with open(path_txt, "r") as f:
                for linea in f:
                    partes = linea.strip().split()
                    if len(partes) >= 5:
                        clase_idx = int(partes[0])
                        cx, cy, bw, bh = map(float, partes[1:5])
                        
                        # Denormalizar
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        
                        if 0 <= clase_idx < len(clases):
                            boxes.append(BoundingBox(
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                clase=clases[clase_idx],
                            ))
            
            self.agregar_imagen_con_boxes(path_img, boxes, video_origen)
            contador += 1
        
        LOGGER.info(f"Importadas {contador} imágenes con anotaciones YOLO")
        return contador
    
    def dividir_train_val_test(
        self,
        ratio_train: float = 0.7,
        ratio_val: float = 0.2,
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Divide las imágenes en train/val/test.
        
        IMPORTANTE: Si hay info de video_origen, mantiene vídeos completos
        en el mismo split para evitar data leakage.
        
        Returns:
            (ids_train, ids_val, ids_test)
        """
        random.seed(seed)
        
        # Si tenemos info de vídeos, dividir por vídeo
        if self._videos_a_imagenes:
            return self._dividir_por_videos(ratio_train, ratio_val)
        
        # Si no, dividir aleatoriamente por imagen
        ids = list(self.imagenes.keys())
        random.shuffle(ids)
        
        n = len(ids)
        n_train = int(n * ratio_train)
        n_val = int(n * ratio_val)
        
        return (
            ids[:n_train],
            ids[n_train:n_train + n_val],
            ids[n_train + n_val:],
        )
    
    def _dividir_por_videos(
        self,
        ratio_train: float,
        ratio_val: float,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Divide manteniendo vídeos completos en cada split."""
        videos = list(self._videos_a_imagenes.keys())
        random.shuffle(videos)
        
        n = len(videos)
        n_train = int(n * ratio_train)
        n_val = int(n * ratio_val)
        
        videos_train = videos[:n_train]
        videos_val = videos[n_train:n_train + n_val]
        videos_test = videos[n_train + n_val:]
        
        ids_train = []
        ids_val = []
        ids_test = []
        
        for vid in videos_train:
            ids_train.extend(self._videos_a_imagenes[vid])
        for vid in videos_val:
            ids_val.extend(self._videos_a_imagenes[vid])
        for vid in videos_test:
            ids_test.extend(self._videos_a_imagenes[vid])
        
        # Imágenes sin vídeo van a train
        ids_con_video = set(ids_train + ids_val + ids_test)
        for img_id in self.imagenes:
            if img_id not in ids_con_video:
                ids_train.append(img_id)
        
        LOGGER.info(f"División por vídeos: {len(videos_train)} train, "
                   f"{len(videos_val)} val, {len(videos_test)} test")
        
        return ids_train, ids_val, ids_test
    
    def generar_yolo(
        self,
        path_salida: Path,
        clases: List[str],
        nombre_dataset: str = "sussy_dataset",
        ratio_train: float = 0.7,
        ratio_val: float = 0.2,
    ) -> Path:
        """
        Genera dataset en formato YOLO.
        
        Args:
            path_salida: Carpeta de salida
            clases: Lista de clases a incluir (en orden de índice)
            nombre_dataset: Nombre para el data.yaml
            ratio_train: Ratio para train
            ratio_val: Ratio para val
        
        Returns:
            Path al archivo data.yaml generado
        """
        path_salida = Path(path_salida)
        
        # Crear estructura
        paths = {
            "train": path_salida / "train",
            "val": path_salida / "val",
            "test": path_salida / "test",
        }
        
        for split_path in paths.values():
            (split_path / "images").mkdir(parents=True, exist_ok=True)
            (split_path / "labels").mkdir(parents=True, exist_ok=True)
        
        # Dividir dataset
        ids_train, ids_val, ids_test = self.dividir_train_val_test(
            ratio_train, ratio_val
        )
        
        splits = {
            "train": ids_train,
            "val": ids_val,
            "test": ids_test,
        }
        
        # Mapeo clase -> índice
        clase_a_idx = {c: i for i, c in enumerate(clases)}
        
        # Estadísticas
        stats = {split: Counter() for split in splits}
        
        # Copiar imágenes y generar etiquetas
        for split_name, img_ids in splits.items():
            split_path = paths[split_name]
            
            for img_id in img_ids:
                img_data = self.imagenes.get(img_id)
                if not img_data or not img_data.path_imagen.exists():
                    continue
                
                # Copiar imagen
                ext = img_data.path_imagen.suffix
                path_img_dest = split_path / "images" / f"{img_id}{ext}"
                shutil.copy2(img_data.path_imagen, path_img_dest)
                
                # Leer dimensiones
                img = cv2.imread(str(img_data.path_imagen))
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                # Generar etiquetas
                path_lbl = split_path / "labels" / f"{img_id}.txt"
                lineas = []
                
                for box in img_data.boxes:
                    if box.clase not in clase_a_idx:
                        continue
                    
                    idx = clase_a_idx[box.clase]
                    linea = box.to_yolo(w, h, idx)
                    lineas.append(linea)
                    stats[split_name][box.clase] += 1
                
                with open(path_lbl, "w") as f:
                    f.write("\n".join(lineas))
        
        # Generar data.yaml
        data_yaml = {
            "path": str(path_salida.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(clases),
            "names": clases,
        }
        
        path_yaml = path_salida / "data.yaml"
        with open(path_yaml, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        # Guardar estadísticas
        self._guardar_estadisticas(path_salida, stats, clases)
        
        LOGGER.info(f"Dataset YOLO generado en: {path_salida}")
        LOGGER.info(f"  Train: {len(ids_train)} imágenes")
        LOGGER.info(f"  Val: {len(ids_val)} imágenes")
        LOGGER.info(f"  Test: {len(ids_test)} imágenes")
        
        return path_yaml
    
    def generar_clasificacion(
        self,
        path_salida: Path,
        campo_clase: str = "clase",  # "clase", "subclase", o atributo
        min_area: int = 32 * 32,
        padding: float = 0.1,
        size_uniforme: Optional[Tuple[int, int]] = (224, 224),
    ) -> Path:
        """
        Genera dataset de clasificación (crops organizados en carpetas).
        
        Estructura:
            path_salida/
                train/
                    clase1/
                        crop_001.jpg
                    clase2/
                        crop_002.jpg
                val/
                    ...
        
        Args:
            path_salida: Carpeta de salida
            campo_clase: Qué campo usar como etiqueta
            min_area: Área mínima del crop para incluir
            padding: Porcentaje de padding alrededor del crop
            size_uniforme: Redimensionar todos los crops a este tamaño
        
        Returns:
            Path a la carpeta generada
        """
        path_salida = Path(path_salida)
        
        # Dividir dataset
        ids_train, ids_val, ids_test = self.dividir_train_val_test()
        
        splits = {"train": ids_train, "val": ids_val, "test": ids_test}
        stats = {split: Counter() for split in splits}
        
        crop_idx = 0
        
        for split_name, img_ids in splits.items():
            for img_id in img_ids:
                img_data = self.imagenes.get(img_id)
                if not img_data or not img_data.path_imagen.exists():
                    continue
                
                img = cv2.imread(str(img_data.path_imagen))
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                for box in img_data.boxes:
                    # Obtener etiqueta según campo
                    if campo_clase == "clase":
                        etiqueta = box.clase
                    elif campo_clase == "subclase":
                        etiqueta = box.subclase or box.clase
                    else:
                        etiqueta = box.atributos.get(campo_clase)
                    
                    if not etiqueta:
                        continue
                    
                    # Verificar área mínima
                    area = box.ancho * box.alto
                    if area < min_area:
                        continue
                    
                    # Calcular crop con padding
                    pad_x = int(box.ancho * padding)
                    pad_y = int(box.alto * padding)
                    
                    x1 = max(0, box.x1 - pad_x)
                    y1 = max(0, box.y1 - pad_y)
                    x2 = min(w, box.x2 + pad_x)
                    y2 = min(h, box.y2 + pad_y)
                    
                    crop = img[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                    
                    # Redimensionar si se especifica
                    if size_uniforme:
                        crop = cv2.resize(crop, size_uniforme)
                    
                    # Guardar crop
                    etiqueta_limpia = self._limpiar_nombre(etiqueta)
                    path_clase = path_salida / split_name / etiqueta_limpia
                    path_clase.mkdir(parents=True, exist_ok=True)
                    
                    path_crop = path_clase / f"crop_{crop_idx:06d}.jpg"
                    cv2.imwrite(str(path_crop), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    stats[split_name][etiqueta] += 1
                    crop_idx += 1
        
        # Guardar estadísticas
        self._guardar_estadisticas_clasificacion(path_salida, stats)
        
        LOGGER.info(f"Dataset de clasificación generado en: {path_salida}")
        
        return path_salida
    
    def _limpiar_nombre(self, nombre: str) -> str:
        """Limpia un nombre para usarlo como nombre de carpeta."""
        import re
        nombre = nombre.lower()
        nombre = re.sub(r'[^\w\-_]', '_', nombre)
        nombre = re.sub(r'_+', '_', nombre)
        return nombre.strip('_')
    
    def _guardar_estadisticas(
        self,
        path_salida: Path,
        stats: Dict[str, Counter],
        clases: List[str],
    ):
        """Guarda estadísticas del dataset YOLO."""
        total_por_clase = Counter()
        for split_stats in stats.values():
            total_por_clase.update(split_stats)
        
        info = {
            "total_imagenes": len(self.imagenes),
            "clases": clases,
            "instancias_por_clase": dict(total_por_clase),
            "por_split": {
                split: {"total": sum(s.values()), "por_clase": dict(s)}
                for split, s in stats.items()
            },
        }
        
        path_stats = path_salida / "estadisticas.json"
        with open(path_stats, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        # Imprimir resumen
        print("\n" + "=" * 50)
        print("ESTADÍSTICAS DEL DATASET")
        print("=" * 50)
        for clase in clases:
            print(f"  {clase}: {total_por_clase[clase]}")
        print("-" * 50)
        for split, s in stats.items():
            print(f"  {split}: {sum(s.values())} instancias")
    
    def _guardar_estadisticas_clasificacion(
        self,
        path_salida: Path,
        stats: Dict[str, Counter],
    ):
        """Guarda estadísticas del dataset de clasificación."""
        total_por_clase = Counter()
        for split_stats in stats.values():
            total_por_clase.update(split_stats)
        
        info = {
            "total_crops": sum(total_por_clase.values()),
            "clases": list(total_por_clase.keys()),
            "instancias_por_clase": dict(total_por_clase),
            "por_split": {
                split: {"total": sum(s.values()), "por_clase": dict(s)}
                for split, s in stats.items()
            },
        }
        
        path_stats = path_salida / "estadisticas.json"
        with open(path_stats, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def estadisticas(self) -> Dict[str, Any]:
        """Obtiene estadísticas del dataset actual."""
        total_boxes = sum(len(img.boxes) for img in self.imagenes.values())
        clases = Counter()
        for img in self.imagenes.values():
            for box in img.boxes:
                clases[box.clase] += 1
        
        return {
            "total_imagenes": len(self.imagenes),
            "imagenes_anotadas": sum(1 for img in self.imagenes.values() if img.tiene_anotaciones),
            "total_boxes": total_boxes,
            "por_clase": dict(clases),
            "videos_unicos": len(self._videos_a_imagenes),
        }


def crear_desde_carpeta_cvat(
    path_cvat: Path,
    formato: str = "yolo",
) -> OrganizadorDataset:
    """
    Crea un OrganizadorDataset desde una exportación de CVAT.
    
    Args:
        path_cvat: Carpeta con la exportación de CVAT
        formato: "yolo" o "coco"
    
    Returns:
        OrganizadorDataset con los datos importados
    """
    path_cvat = Path(path_cvat)
    org = OrganizadorDataset(path_cvat)
    
    if formato == "yolo":
        # Buscar data.yaml para obtener clases
        path_yaml = path_cvat / "data.yaml"
        if path_yaml.exists():
            with open(path_yaml) as f:
                data = yaml.safe_load(f)
            clases = data.get("names", [])
            
            # Importar train, val, test si existen
            for split in ["train", "val", "test"]:
                path_img = path_cvat / split / "images"
                path_lbl = path_cvat / split / "labels"
                
                if path_img.exists() and path_lbl.exists():
                    org.importar_yolo(path_img, path_lbl, clases, video_origen=f"{split}_cvat")
    
    return org


if __name__ == "__main__":
    # Ejemplo de uso
    from sussy.training.taxonomia import obtener_taxonomia
    
    # Obtener clases de la taxonomía
    taxonomia = obtener_taxonomia()
    clases_yolo = [info["nombre_yolo"] for info in taxonomia.values()]
    
    print(f"Clases para detección: {clases_yolo[:10]}...")
    
    # Crear organizador vacío
    org = OrganizadorDataset(Path("dataset_sussy"))
    
    print("\nOrganizador de dataset listo")
    print("Usa .agregar_imagen_con_boxes() para agregar datos")
