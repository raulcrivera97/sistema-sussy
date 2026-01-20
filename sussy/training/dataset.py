"""
Gestión de Datasets para entrenamiento jerárquico.

Este módulo proporciona herramientas para:
- Organizar imágenes según la taxonomía
- Generar etiquetas en formato YOLO
- Crear splits train/val/test
- Exportar configuraciones de entrenamiento
"""

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import hashlib

from sussy.training.taxonomia import (
    GestorTaxonomia,
    ClaseDeteccion,
    EtiquetaDeteccion,
    obtener_taxonomia,
)


@dataclass
class ImagenDataset:
    """Representa una imagen con sus etiquetas."""
    path: Path
    etiquetas: List[EtiquetaDeteccion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def nombre(self) -> str:
        return self.path.stem
    
    @property
    def extension(self) -> str:
        return self.path.suffix
    
    def obtener_clases(self) -> List[ClaseDeteccion]:
        """Obtiene las clases únicas en esta imagen."""
        return list(set(e.clase for e in self.etiquetas))
    
    def obtener_categorias(self) -> List[str]:
        """Obtiene las categorías únicas."""
        return list(set(e.clase.categoria for e in self.etiquetas))
    
    def guardar_etiquetas_yolo(self, path_labels: Path):
        """Guarda las etiquetas en formato YOLO."""
        path_txt = path_labels / f"{self.nombre}.txt"
        
        with open(path_txt, "w") as f:
            for etiqueta in self.etiquetas:
                f.write(etiqueta.to_yolo_line() + "\n")
    
    def guardar_etiquetas_json(self, path_labels: Path):
        """Guarda las etiquetas en formato JSON (con atributos completos)."""
        path_json = path_labels / f"{self.nombre}.json"
        
        data = {
            "imagen": str(self.path.name),
            "etiquetas": [e.to_dict() for e in self.etiquetas],
            "metadata": self.metadata,
        }
        
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


@dataclass
class ConfiguracionDataset:
    """Configuración para la creación del dataset."""
    nombre: str
    path_salida: Path
    path_imagenes: Optional[Path] = None
    
    # Splits
    ratio_train: float = 0.7
    ratio_val: float = 0.2
    ratio_test: float = 0.1
    
    # Niveles de detalle
    nivel_deteccion: int = 3  # 1=categoría, 2=subcategoría, 3=subtipo
    incluir_atributos: bool = True
    
    # Formatos de salida
    formato_etiquetas: str = "yolo"  # "yolo", "json", "ambos"
    
    # Filtros
    categorias_incluidas: Optional[List[str]] = None
    min_imagenes_por_clase: int = 5
    
    # Augmentación
    augmentar_clases_escasas: bool = True
    factor_augmentacion: int = 3
    
    def validar(self) -> List[str]:
        """Valida la configuración."""
        errores = []
        
        if abs(self.ratio_train + self.ratio_val + self.ratio_test - 1.0) > 0.01:
            errores.append("Los ratios de split deben sumar 1.0")
        
        if self.nivel_deteccion not in [1, 2, 3]:
            errores.append("nivel_deteccion debe ser 1, 2 o 3")
        
        if self.formato_etiquetas not in ["yolo", "json", "ambos"]:
            errores.append("formato_etiquetas debe ser 'yolo', 'json' o 'ambos'")
        
        return errores


class GestorDataset:
    """
    Gestiona la creación y organización de datasets para entrenamiento.
    
    Flujo típico:
    1. Crear gestor con configuración
    2. Añadir imágenes con etiquetas
    3. Generar splits
    4. Exportar a formato YOLO
    """
    
    def __init__(self, config: ConfiguracionDataset):
        self.config = config
        self.taxonomia = obtener_taxonomia()
        self.imagenes: List[ImagenDataset] = []
        self._estadisticas: Dict[str, int] = defaultdict(int)
    
    def añadir_imagen(self, imagen: ImagenDataset) -> bool:
        """
        Añade una imagen al dataset.
        
        Returns:
            True si se añadió correctamente
        """
        # Validar que las clases existen
        for etiqueta in imagen.etiquetas:
            errores = self.taxonomia.validar_etiqueta(etiqueta)
            if errores:
                print(f"Errores en {imagen.path}: {errores}")
                return False
        
        self.imagenes.append(imagen)
        
        # Actualizar estadísticas
        for etiqueta in imagen.etiquetas:
            self._estadisticas[etiqueta.clase.nombre_completo] += 1
        
        return True
    
    def importar_desde_carpeta(
        self,
        path: Path,
        etiquetador: Optional[callable] = None,
        extensiones: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> int:
        """
        Importa imágenes desde una carpeta.
        
        La estructura esperada es:
        path/
          categoria/
            subcategoria/
              subtipo/
                imagen.jpg
        
        O con archivo de etiquetas:
        path/
          images/
            imagen.jpg
          labels/
            imagen.txt (formato YOLO)
            imagen.json (formato extendido)
        
        Args:
            path: Carpeta raíz
            etiquetador: Función opcional para generar etiquetas automáticamente
            extensiones: Extensiones de imagen a incluir
        
        Returns:
            Número de imágenes importadas
        """
        path = Path(path)
        importadas = 0
        
        # Detectar estructura
        if (path / "images").exists() and (path / "labels").exists():
            # Estructura YOLO estándar
            importadas = self._importar_estructura_yolo(path, extensiones)
        else:
            # Estructura por carpetas (jerarquía)
            importadas = self._importar_estructura_carpetas(path, extensiones)
        
        return importadas
    
    def _importar_estructura_yolo(
        self,
        path: Path,
        extensiones: Tuple[str, ...],
    ) -> int:
        """Importa desde estructura YOLO (images/ + labels/)."""
        path_images = path / "images"
        path_labels = path / "labels"
        importadas = 0
        
        for img_path in path_images.iterdir():
            if img_path.suffix.lower() not in extensiones:
                continue
            
            # Buscar archivo de etiquetas
            label_txt = path_labels / f"{img_path.stem}.txt"
            label_json = path_labels / f"{img_path.stem}.json"
            
            etiquetas = []
            
            if label_json.exists():
                etiquetas = self._cargar_etiquetas_json(label_json)
            elif label_txt.exists():
                etiquetas = self._cargar_etiquetas_yolo(label_txt)
            
            imagen = ImagenDataset(path=img_path, etiquetas=etiquetas)
            if self.añadir_imagen(imagen):
                importadas += 1
        
        return importadas
    
    def _importar_estructura_carpetas(
        self,
        path: Path,
        extensiones: Tuple[str, ...],
    ) -> int:
        """Importa desde estructura de carpetas jerárquica."""
        importadas = 0
        
        for img_path in path.rglob("*"):
            if img_path.suffix.lower() not in extensiones:
                continue
            
            # Inferir clase desde la ruta
            partes_ruta = img_path.relative_to(path).parts[:-1]  # Sin el nombre del archivo
            
            if not partes_ruta:
                # Imagen en raíz, sin clasificar
                clase_nombre = "sin_clasificar"
            else:
                clase_nombre = ".".join(partes_ruta)
            
            clase = self.taxonomia.obtener_clase(clase_nombre)
            
            if clase is None:
                # Intentar con partes individuales
                for parte in reversed(partes_ruta):
                    clase = self.taxonomia.obtener_clase(parte)
                    if clase:
                        break
            
            if clase is None:
                print(f"Clase no encontrada para: {img_path.relative_to(path)}")
                continue
            
            # Crear etiqueta (bbox completo ya que es clasificación)
            etiqueta = EtiquetaDeteccion(
                clase=clase,
                bbox=(0.0, 0.0, 1.0, 1.0),  # Imagen completa
            )
            
            imagen = ImagenDataset(path=img_path, etiquetas=[etiqueta])
            if self.añadir_imagen(imagen):
                importadas += 1
        
        return importadas
    
    def _cargar_etiquetas_yolo(self, path: Path) -> List[EtiquetaDeteccion]:
        """Carga etiquetas desde formato YOLO."""
        etiquetas = []
        
        with open(path, "r") as f:
            for linea in f:
                partes = linea.strip().split()
                if len(partes) < 5:
                    continue
                
                clase_id = int(partes[0])
                cx, cy, w, h = map(float, partes[1:5])
                
                # Convertir a x1,y1,x2,y2
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                
                clase = self.taxonomia.obtener_clase(clase_id)
                if clase:
                    etiquetas.append(EtiquetaDeteccion(
                        clase=clase,
                        bbox=(x1, y1, x2, y2),
                    ))
        
        return etiquetas
    
    def _cargar_etiquetas_json(self, path: Path) -> List[EtiquetaDeteccion]:
        """Carga etiquetas desde formato JSON extendido."""
        etiquetas = []
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data.get("etiquetas", []):
            clase_id = item.get("clase_id")
            clase = self.taxonomia.obtener_clase(clase_id)
            
            if clase:
                etiquetas.append(EtiquetaDeteccion(
                    clase=clase,
                    bbox=tuple(item.get("bbox", [0, 0, 1, 1])),
                    confianza=item.get("confianza", 1.0),
                    atributos=item.get("atributos", {}),
                ))
        
        return etiquetas
    
    def generar_splits(self, seed: int = 42) -> Dict[str, List[ImagenDataset]]:
        """
        Genera splits train/val/test estratificados por clase.
        
        Returns:
            Dict con listas de imágenes por split
        """
        random.seed(seed)
        
        # Agrupar imágenes por clase principal
        por_clase: Dict[str, List[ImagenDataset]] = defaultdict(list)
        
        for imagen in self.imagenes:
            if imagen.etiquetas:
                clase_principal = imagen.etiquetas[0].clase.nombre_completo
                por_clase[clase_principal].append(imagen)
        
        splits = {"train": [], "val": [], "test": []}
        
        for clase, imagenes_clase in por_clase.items():
            random.shuffle(imagenes_clase)
            
            n = len(imagenes_clase)
            n_train = int(n * self.config.ratio_train)
            n_val = int(n * self.config.ratio_val)
            
            splits["train"].extend(imagenes_clase[:n_train])
            splits["val"].extend(imagenes_clase[n_train:n_train + n_val])
            splits["test"].extend(imagenes_clase[n_train + n_val:])
        
        # Mezclar cada split
        for split in splits.values():
            random.shuffle(split)
        
        return splits
    
    def exportar_yolo(
        self,
        path_salida: Optional[Path] = None,
        splits: Optional[Dict[str, List[ImagenDataset]]] = None,
    ) -> Path:
        """
        Exporta el dataset en formato YOLO.
        
        Estructura generada:
        path_salida/
          data.yaml
          train/
            images/
            labels/
          val/
            images/
            labels/
          test/
            images/
            labels/
        
        Returns:
            Path del directorio creado
        """
        path_salida = path_salida or self.config.path_salida
        path_salida = Path(path_salida)
        
        # Generar splits si no se proporcionan
        if splits is None:
            splits = self.generar_splits()
        
        # Crear estructura de directorios
        for split_name in ["train", "val", "test"]:
            (path_salida / split_name / "images").mkdir(parents=True, exist_ok=True)
            (path_salida / split_name / "labels").mkdir(parents=True, exist_ok=True)
        
        # Copiar imágenes y generar etiquetas
        for split_name, imagenes in splits.items():
            path_images = path_salida / split_name / "images"
            path_labels = path_salida / split_name / "labels"
            
            for imagen in imagenes:
                # Copiar imagen
                dest_img = path_images / imagen.path.name
                if not dest_img.exists():
                    shutil.copy2(imagen.path, dest_img)
                
                # Guardar etiquetas
                imagen.guardar_etiquetas_yolo(path_labels)
                
                if self.config.formato_etiquetas in ["json", "ambos"]:
                    imagen.guardar_etiquetas_json(path_labels)
        
        # Generar data.yaml
        self._generar_yaml_config(path_salida)
        
        # Generar estadísticas
        self._generar_estadisticas(path_salida, splits)
        
        return path_salida
    
    def _generar_yaml_config(self, path: Path):
        """Genera el archivo data.yaml para YOLO."""
        # Obtener clases según nivel de detalle
        clases = [
            c for c in self.taxonomia.obtener_todas_clases()
            if c.nivel <= self.config.nivel_deteccion
        ]
        clases.sort(key=lambda c: c.id)
        
        import yaml
        
        config = {
            "path": str(path.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(clases),
            "names": {c.id: c.nombre_completo for c in clases},
        }
        
        with open(path / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    
    def _generar_estadisticas(
        self,
        path: Path,
        splits: Dict[str, List[ImagenDataset]],
    ):
        """Genera archivo de estadísticas del dataset."""
        stats = {
            "total_imagenes": len(self.imagenes),
            "splits": {
                split: len(imgs) for split, imgs in splits.items()
            },
            "clases": dict(self._estadisticas),
            "configuracion": {
                "nivel_deteccion": self.config.nivel_deteccion,
                "ratio_train": self.config.ratio_train,
                "ratio_val": self.config.ratio_val,
                "ratio_test": self.config.ratio_test,
            },
        }
        
        with open(path / "estadisticas.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Obtiene estadísticas del dataset actual."""
        return {
            "total_imagenes": len(self.imagenes),
            "clases": dict(self._estadisticas),
            "clases_unicas": len(self._estadisticas),
        }
    
    def imprimir_resumen(self):
        """Imprime un resumen del dataset."""
        stats = self.obtener_estadisticas()
        
        print("\n" + "=" * 50)
        print(f"DATASET: {self.config.nombre}")
        print("=" * 50)
        print(f"Total imágenes: {stats['total_imagenes']}")
        print(f"Clases únicas: {stats['clases_unicas']}")
        print("\nDistribución por clase:")
        
        for clase, count in sorted(stats['clases'].items(), key=lambda x: -x[1]):
            print(f"  {clase}: {count}")


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def crear_dataset_desde_carpeta(
    path_imagenes: Path,
    path_salida: Path,
    nombre: str = "sussy_dataset",
    **kwargs,
) -> GestorDataset:
    """
    Crea un dataset a partir de una carpeta de imágenes.
    
    Args:
        path_imagenes: Carpeta con imágenes organizadas
        path_salida: Donde guardar el dataset procesado
        nombre: Nombre del dataset
        **kwargs: Configuración adicional
    
    Returns:
        GestorDataset configurado
    """
    config = ConfiguracionDataset(
        nombre=nombre,
        path_salida=Path(path_salida),
        path_imagenes=Path(path_imagenes),
        **kwargs,
    )
    
    errores = config.validar()
    if errores:
        raise ValueError(f"Configuración inválida: {errores}")
    
    gestor = GestorDataset(config)
    n_importadas = gestor.importar_desde_carpeta(path_imagenes)
    
    print(f"Importadas {n_importadas} imágenes")
    
    return gestor


if __name__ == "__main__":
    # Ejemplo de uso
    from pathlib import Path
    
    # Crear configuración
    config = ConfiguracionDataset(
        nombre="sussy_v1",
        path_salida=Path("datasets/sussy_v1"),
        nivel_deteccion=3,
        incluir_atributos=True,
    )
    
    gestor = GestorDataset(config)
    
    # Importar desde carpeta existente
    # gestor.importar_desde_carpeta(Path("datasets/raw"))
    
    # Mostrar estadísticas
    gestor.imprimir_resumen()
