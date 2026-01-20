"""
Extractor de embeddings visuales para Re-ID.

Este módulo proporciona extractores de features visuales que funcionan
completamente en CPU, sin depender de PyTorch/CUDA. Útil para Re-ID
básico cuando no hay GPU compatible disponible.

Los embeddings combinan:
- Histograma de color HSV (robusto a cambios de iluminación)
- Características de forma/aspecto
- Textura básica (gradientes)
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class ExtractorAparienciaCPU:
    """
    Extractor de features visuales que funciona completamente en CPU.
    
    Genera embeddings combinando:
    - Histograma de color HSV (16 bins H, 8 bins S, 8 bins V)
    - Ratio de aspecto
    - Intensidad media y desviación
    - Características de bordes/textura
    
    El embedding resultante tiene dimensión fija (50 elementos por defecto).
    """
    
    def __init__(
        self,
        bins_h: int = 16,
        bins_s: int = 8,
        bins_v: int = 8,
        usar_textura: bool = True,
        normalizar: bool = True,
    ):
        """
        Args:
            bins_h: Número de bins para el canal Hue
            bins_s: Número de bins para el canal Saturation
            bins_v: Número de bins para el canal Value
            usar_textura: Si calcular características de textura/bordes
            normalizar: Si normalizar el embedding final a norma unitaria
        """
        self.bins_h = bins_h
        self.bins_s = bins_s
        self.bins_v = bins_v
        self.usar_textura = usar_textura
        self.normalizar = normalizar
        
        # Dimensión del embedding
        # HSV: bins_h + bins_s + bins_v
        # Forma: 2 (aspect, area_rel)
        # Intensidad: 2 (mean, std)
        # Textura: 4 (gradiente medio/std en x/y) si usar_textura
        self._dim = bins_h + bins_s + bins_v + 4
        if usar_textura:
            self._dim += 4
    
    @property
    def dimension(self) -> int:
        """Dimensión del embedding generado."""
        return self._dim
    
    def extraer(
        self,
        frame: np.ndarray,
        bbox: List[int],
        padding: float = 0.1,
    ) -> np.ndarray:
        """
        Extrae embedding visual de una región del frame.
        
        Args:
            frame: Frame BGR completo
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding relativo alrededor del bbox (0.1 = 10%)
        
        Returns:
            Vector de features de dimensión fija
        """
        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]
        
        # Aplicar padding
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w_frame, x2 + pad_x)
        y2 = min(h_frame, y2 + pad_y)
        
        # Extraer crop
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return np.zeros(self._dim, dtype=np.float32)
        
        # Redimensionar para consistencia (reduce ruido)
        crop_resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        
        features = []
        
        # 1. Histograma HSV
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
        
        hist_h = cv2.calcHist([hsv], [0], None, [self.bins_h], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.bins_s], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [self.bins_v], [0, 256])
        
        # Normalizar histogramas
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-8)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-8)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-8)
        
        features.extend(hist_h)
        features.extend(hist_s)
        features.extend(hist_v)
        
        # 2. Características de forma
        aspect_ratio = w / max(1, h)
        # Normalizar aspecto a rango ~[-1, 1] con sigmoid-like
        aspect_norm = (aspect_ratio - 1.0) / (1.0 + abs(aspect_ratio - 1.0))
        
        area_rel = (w * h) / (w_frame * h_frame)
        area_norm = min(1.0, area_rel * 10)  # Escalar, típicamente objetos < 10% del frame
        
        features.extend([aspect_norm, area_norm])
        
        # 3. Intensidad
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        mean_intensity = gray.mean() / 255.0
        std_intensity = gray.std() / 128.0  # Normalizar a ~[0, 2]
        
        features.extend([mean_intensity, std_intensity])
        
        # 4. Textura (gradientes)
        if self.usar_textura:
            # Gradientes Sobel
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            
            # Estadísticas de gradientes
            grad_x_mean = np.abs(grad_x).mean() / 255.0
            grad_x_std = grad_x.std() / 128.0
            grad_y_mean = np.abs(grad_y).mean() / 255.0
            grad_y_std = grad_y.std() / 128.0
            
            features.extend([grad_x_mean, grad_x_std, grad_y_mean, grad_y_std])
        
        embedding = np.array(features, dtype=np.float32)
        
        # Normalizar a norma unitaria
        if self.normalizar:
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
        
        return embedding
    
    def similitud(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcula similitud entre dos embeddings (coseno).
        
        Args:
            emb1: Primer embedding
            emb2: Segundo embedding
        
        Returns:
            Similitud en rango [-1, 1], donde 1 = idénticos
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def distancia(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcula distancia entre dos embeddings (euclidiana normalizada).
        
        Args:
            emb1: Primer embedding
            emb2: Segundo embedding
        
        Returns:
            Distancia en rango [0, 2] para embeddings normalizados
        """
        if emb1 is None or emb2 is None:
            return 2.0
        
        return float(np.linalg.norm(emb1 - emb2))


class ExtractorAparienciaMultiEscala(ExtractorAparienciaCPU):
    """
    Extractor que combina features de múltiples escalas para mayor robustez.
    
    Genera embeddings a 3 escalas (original, 0.75x, 0.5x) y los concatena,
    proporcionando mayor invarianza a escala.
    """
    
    def __init__(
        self,
        bins_h: int = 12,
        bins_s: int = 6,
        bins_v: int = 6,
        usar_textura: bool = True,
        normalizar: bool = True,
        escalas: Tuple[float, ...] = (1.0, 0.75, 0.5),
    ):
        # Bins reducidos porque concatenamos múltiples escalas
        super().__init__(
            bins_h=bins_h,
            bins_s=bins_s,
            bins_v=bins_v,
            usar_textura=usar_textura,
            normalizar=False,  # Normalizamos al final
        )
        self.escalas = escalas
        self._normalizar_final = normalizar
        self._dim_total = self._dim * len(escalas)
    
    @property
    def dimension(self) -> int:
        return self._dim_total
    
    def extraer(
        self,
        frame: np.ndarray,
        bbox: List[int],
        padding: float = 0.1,
    ) -> np.ndarray:
        """Extrae embedding multi-escala."""
        embeddings = []
        
        for escala in self.escalas:
            if escala == 1.0:
                frame_escala = frame
                bbox_escala = bbox
            else:
                # Redimensionar frame y ajustar bbox
                h, w = frame.shape[:2]
                new_w = int(w * escala)
                new_h = int(h * escala)
                frame_escala = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                bbox_escala = [int(b * escala) for b in bbox]
            
            emb = super().extraer(frame_escala, bbox_escala, padding)
            embeddings.append(emb)
        
        embedding = np.concatenate(embeddings)
        
        if self._normalizar_final:
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
        
        return embedding


def crear_extractor_apariencia(
    tipo: str = "simple",
    **kwargs,
) -> ExtractorAparienciaCPU:
    """
    Factory para crear extractores de apariencia.
    
    Args:
        tipo: "simple" o "multiescala"
        **kwargs: Argumentos para el extractor
    
    Returns:
        Instancia del extractor
    """
    if tipo == "multiescala":
        return ExtractorAparienciaMultiEscala(**kwargs)
    else:
        return ExtractorAparienciaCPU(**kwargs)


__all__ = [
    "ExtractorAparienciaCPU",
    "ExtractorAparienciaMultiEscala",
    "crear_extractor_apariencia",
]
