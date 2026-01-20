"""
Extractor de Frames para Dataset de Entrenamiento.

Utilidades para:
- Extraer frames de vídeos con filtrado de duplicados
- Detectar y descartar frames similares
- Organizar frames por escenas/vídeos (evitar data leakage)
- Preprocesar para diferentes fases de entrenamiento
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import shutil

import cv2
import numpy as np

LOGGER = logging.getLogger("sussy.training.extractor")


@dataclass
class ConfigExtraccion:
    """Configuración para extracción de frames."""
    # Intervalo de extracción
    cada_n_frames: int = 30  # Extraer 1 de cada N frames
    cada_n_segundos: Optional[float] = None  # Alternativa: cada N segundos
    
    # Filtrado de duplicados
    filtrar_duplicados: bool = True
    umbral_similitud: float = 0.95  # 0-1, mayor = más estricto
    metodo_similitud: str = "dhash"  # "dhash", "phash", "histogram"
    
    # Calidad
    min_blur_score: float = 100.0  # Descartar frames muy borrosos (Laplacian variance)
    min_contraste: float = 30.0  # Descartar frames con poco contraste
    
    # Salida
    formato_salida: str = "jpg"
    calidad_jpg: int = 95
    prefijo_nombre: str = ""
    
    # Metadatos
    guardar_metadatos: bool = True


@dataclass
class FrameExtraido:
    """Información de un frame extraído."""
    path: Path
    frame_idx: int
    timestamp_ms: float
    video_origen: str
    hash_imagen: str
    blur_score: float
    es_duplicado: bool = False


class ExtractorFrames:
    """
    Extrae frames de vídeos con filtrado inteligente.
    
    Ejemplo:
        extractor = ExtractorFrames(config)
        frames = extractor.extraer_video("video.mp4", "salida/")
        print(f"Extraídos {len(frames)} frames únicos")
    """
    
    def __init__(self, config: Optional[ConfigExtraccion] = None):
        self.config = config or ConfigExtraccion()
        self._hashes_vistos: Set[str] = set()
        self._estadisticas = defaultdict(int)
    
    def extraer_video(
        self,
        path_video: Path,
        path_salida: Path,
        video_id: Optional[str] = None,
    ) -> List[FrameExtraido]:
        """
        Extrae frames de un vídeo.
        
        Args:
            path_video: Ruta al vídeo
            path_salida: Carpeta donde guardar los frames
            video_id: Identificador único del vídeo (para evitar colisiones)
        
        Returns:
            Lista de frames extraídos
        """
        path_video = Path(path_video)
        path_salida = Path(path_salida)
        path_salida.mkdir(parents=True, exist_ok=True)
        
        if video_id is None:
            video_id = path_video.stem
        
        cap = cv2.VideoCapture(str(path_video))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el vídeo: {path_video}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calcular intervalo
        if self.config.cada_n_segundos:
            intervalo = int(fps * self.config.cada_n_segundos)
        else:
            intervalo = self.config.cada_n_frames
        
        intervalo = max(1, intervalo)
        
        LOGGER.info(f"Extrayendo frames de {path_video.name}")
        LOGGER.info(f"  FPS: {fps:.1f}, Total frames: {total_frames}")
        LOGGER.info(f"  Intervalo: cada {intervalo} frames")
        
        frames_extraidos = []
        frame_idx = 0
        ultimo_hash = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Solo procesar cada N frames
            if frame_idx % intervalo != 0:
                frame_idx += 1
                continue
            
            self._estadisticas["frames_analizados"] += 1
            
            # Calcular métricas de calidad
            blur_score = self._calcular_blur(frame)
            contraste = self._calcular_contraste(frame)
            
            # Filtrar por calidad
            if blur_score < self.config.min_blur_score:
                self._estadisticas["descartados_blur"] += 1
                frame_idx += 1
                continue
            
            if contraste < self.config.min_contraste:
                self._estadisticas["descartados_contraste"] += 1
                frame_idx += 1
                continue
            
            # Calcular hash para detectar duplicados
            hash_img = self._calcular_hash(frame)
            
            es_duplicado = False
            if self.config.filtrar_duplicados:
                if hash_img in self._hashes_vistos:
                    es_duplicado = True
                    self._estadisticas["duplicados_exactos"] += 1
                elif ultimo_hash and self._similitud_hash(hash_img, ultimo_hash) > self.config.umbral_similitud:
                    es_duplicado = True
                    self._estadisticas["duplicados_similares"] += 1
            
            if es_duplicado:
                frame_idx += 1
                continue
            
            # Guardar frame
            timestamp_ms = (frame_idx / fps) * 1000
            nombre = f"{self.config.prefijo_nombre}{video_id}_f{frame_idx:06d}"
            
            if self.config.formato_salida == "jpg":
                path_frame = path_salida / f"{nombre}.jpg"
                cv2.imwrite(
                    str(path_frame),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.calidad_jpg]
                )
            else:
                path_frame = path_salida / f"{nombre}.png"
                cv2.imwrite(str(path_frame), frame)
            
            frame_info = FrameExtraido(
                path=path_frame,
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                video_origen=video_id,
                hash_imagen=hash_img,
                blur_score=blur_score,
            )
            
            frames_extraidos.append(frame_info)
            self._hashes_vistos.add(hash_img)
            ultimo_hash = hash_img
            self._estadisticas["frames_guardados"] += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Guardar metadatos
        if self.config.guardar_metadatos:
            self._guardar_metadatos(path_salida, video_id, frames_extraidos)
        
        LOGGER.info(f"  Extraídos: {len(frames_extraidos)} frames únicos")
        
        return frames_extraidos
    
    def extraer_multiples_videos(
        self,
        paths_videos: List[Path],
        path_salida: Path,
        separar_por_video: bool = True,
    ) -> Dict[str, List[FrameExtraido]]:
        """
        Extrae frames de múltiples vídeos.
        
        Args:
            paths_videos: Lista de rutas a vídeos
            path_salida: Carpeta base de salida
            separar_por_video: Si crear subcarpetas por vídeo
        
        Returns:
            Dict con frames por vídeo
        """
        resultados = {}
        
        for path_video in paths_videos:
            video_id = path_video.stem
            
            if separar_por_video:
                path_video_salida = path_salida / video_id
            else:
                path_video_salida = path_salida
            
            frames = self.extraer_video(path_video, path_video_salida, video_id)
            resultados[video_id] = frames
        
        return resultados
    
    def _calcular_blur(self, frame: np.ndarray) -> float:
        """Calcula score de nitidez (mayor = más nítido)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _calcular_contraste(self, frame: np.ndarray) -> float:
        """Calcula contraste del frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.std()
    
    def _calcular_hash(self, frame: np.ndarray) -> str:
        """Calcula hash perceptual del frame."""
        if self.config.metodo_similitud == "dhash":
            return self._dhash(frame)
        elif self.config.metodo_similitud == "phash":
            return self._phash(frame)
        else:
            return self._histogram_hash(frame)
    
    def _dhash(self, frame: np.ndarray, hash_size: int = 8) -> str:
        """Difference hash - rápido y efectivo."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        
        # Comparar píxeles adyacentes
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convertir a hash hexadecimal
        return ''.join(str(int(b)) for b in diff.flatten())
    
    def _phash(self, frame: np.ndarray, hash_size: int = 8) -> str:
        """Perceptual hash usando DCT."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size * 4, hash_size * 4))
        
        # DCT
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:hash_size, :hash_size]
        
        # Binarizar por mediana
        median = np.median(dct_low)
        diff = dct_low > median
        
        return ''.join(str(int(b)) for b in diff.flatten())
    
    def _histogram_hash(self, frame: np.ndarray) -> str:
        """Hash basado en histograma de color."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = hist.flatten()
        hist = hist / hist.sum()
        
        # Cuantizar a 4 bits
        quantized = (hist * 15).astype(int)
        return ''.join(f'{v:x}' for v in quantized)
    
    def _similitud_hash(self, hash1: str, hash2: str) -> float:
        """Calcula similitud entre dos hashes (0-1)."""
        if len(hash1) != len(hash2):
            return 0.0
        
        # Distancia de Hamming normalizada
        diferencias = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        return 1.0 - (diferencias / len(hash1))
    
    def _guardar_metadatos(
        self,
        path_salida: Path,
        video_id: str,
        frames: List[FrameExtraido],
    ):
        """Guarda metadatos de la extracción."""
        metadatos = {
            "video_id": video_id,
            "total_frames": len(frames),
            "config": {
                "cada_n_frames": self.config.cada_n_frames,
                "umbral_similitud": self.config.umbral_similitud,
                "min_blur_score": self.config.min_blur_score,
            },
            "frames": [
                {
                    "nombre": f.path.name,
                    "frame_idx": f.frame_idx,
                    "timestamp_ms": f.timestamp_ms,
                    "blur_score": f.blur_score,
                }
                for f in frames
            ],
        }
        
        path_meta = path_salida / f"{video_id}_metadata.json"
        with open(path_meta, "w", encoding="utf-8") as f:
            json.dump(metadatos, f, indent=2)
    
    def obtener_estadisticas(self) -> Dict[str, int]:
        """Obtiene estadísticas de la extracción."""
        return dict(self._estadisticas)
    
    def resetear(self):
        """Resetea el estado del extractor."""
        self._hashes_vistos.clear()
        self._estadisticas.clear()


def dividir_por_videos(
    frames_por_video: Dict[str, List[FrameExtraido]],
    ratio_train: float = 0.7,
    ratio_val: float = 0.2,
    seed: int = 42,
) -> Tuple[List[FrameExtraido], List[FrameExtraido], List[FrameExtraido]]:
    """
    Divide frames en train/val/test MANTENIENDO vídeos completos juntos.
    
    Esto evita data leakage: frames del mismo vídeo no estarán en splits diferentes.
    
    Returns:
        (train_frames, val_frames, test_frames)
    """
    import random
    random.seed(seed)
    
    videos = list(frames_por_video.keys())
    random.shuffle(videos)
    
    n_videos = len(videos)
    n_train = int(n_videos * ratio_train)
    n_val = int(n_videos * ratio_val)
    
    videos_train = videos[:n_train]
    videos_val = videos[n_train:n_train + n_val]
    videos_test = videos[n_train + n_val:]
    
    train_frames = []
    val_frames = []
    test_frames = []
    
    for vid in videos_train:
        train_frames.extend(frames_por_video[vid])
    for vid in videos_val:
        val_frames.extend(frames_por_video[vid])
    for vid in videos_test:
        test_frames.extend(frames_por_video[vid])
    
    return train_frames, val_frames, test_frames


if __name__ == "__main__":
    # Ejemplo de uso
    config = ConfigExtraccion(
        cada_n_frames=30,
        filtrar_duplicados=True,
        umbral_similitud=0.92,
    )
    
    extractor = ExtractorFrames(config)
    
    # Extraer de un vídeo
    # frames = extractor.extraer_video("video.mp4", "frames_salida/")
    
    print("Extractor de frames listo")
    print(f"Configuración: {config}")
