from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2

Logger = logging.Logger
Source = Union[str, int]


# =============================================================================
# SOPORTE PARA FUENTES EN DIRECTO (WEBCAMS)
# =============================================================================

@dataclass
class WebcamInfo:
    """Información sobre una webcam detectada."""
    index: int
    name: str
    resolution: Tuple[int, int]
    fps: float
    backend: str
    
    @property
    def display_name(self) -> str:
        """Nombre para mostrar en la UI."""
        return f"📹 {self.name} ({self.resolution[0]}x{self.resolution[1]} @ {self.fps:.0f}fps)"
    
    def __str__(self) -> str:
        return self.display_name


def enumerar_webcams(max_index: int = 10, logger: Optional[Logger] = None) -> List[WebcamInfo]:
    """
    Enumera las webcams disponibles en el sistema.
    
    Args:
        max_index: Máximo índice a probar (por defecto 10).
        logger: Logger opcional para mensajes de debug.
        
    Returns:
        Lista de WebcamInfo con las webcams encontradas.
    """
    logger = logger or logging.getLogger("sussy.fuentes")
    webcams: List[WebcamInfo] = []
    
    # Backends a probar en orden de preferencia
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),      # Windows - más info de dispositivos
        (cv2.CAP_MSMF, "Media Foundation"),  # Windows - mejor rendimiento
        (cv2.CAP_ANY, "Auto"),               # Cualquier backend disponible
    ]
    
    for idx in range(max_index):
        for backend_id, backend_name in backends:
            try:
                cap = cv2.VideoCapture(idx, backend_id)
                if cap.isOpened():
                    # Obtener info de la cámara
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    
                    # Intentar obtener nombre del dispositivo (solo DirectShow en Windows)
                    name = f"Cámara {idx}"
                    if backend_id == cv2.CAP_DSHOW:
                        # En DirectShow podemos intentar obtener el nombre real
                        # pero OpenCV no lo expone directamente, usamos el índice
                        name = f"Webcam {idx}"
                    
                    # Verificar que realmente podemos leer un frame
                    ret, _ = cap.read()
                    cap.release()
                    
                    if ret:
                        webcam = WebcamInfo(
                            index=idx,
                            name=name,
                            resolution=(width, height),
                            fps=fps,
                            backend=backend_name
                        )
                        webcams.append(webcam)
                        logger.debug("Webcam encontrada: %s", webcam)
                        break  # Ya encontramos esta cámara, no probar más backends
                    
            except Exception as e:
                logger.debug("Error probando índice %d con backend %s: %s", idx, backend_name, e)
                continue
    
    logger.info("Se encontraron %d webcams disponibles", len(webcams))
    return webcams


def abrir_webcam(
    index: int = 0,
    preferred_resolution: Optional[Tuple[int, int]] = None,
    preferred_fps: Optional[float] = None,
    logger: Optional[Logger] = None,
) -> cv2.VideoCapture:
    """
    Abre una webcam con configuración optimizada para procesamiento en tiempo real.
    
    Args:
        index: Índice de la webcam (0 = primera cámara).
        preferred_resolution: Resolución preferida (ancho, alto). Por defecto (1280, 720).
        preferred_fps: FPS preferidos. Por defecto 30.
        logger: Logger opcional.
        
    Returns:
        VideoCapture configurado para la webcam.
    """
    logger = logger or logging.getLogger("sussy.fuentes")
    
    # Valores por defecto optimizados para procesamiento de vídeo
    if preferred_resolution is None:
        preferred_resolution = (1280, 720)  # HD - buen balance calidad/rendimiento
    if preferred_fps is None:
        preferred_fps = 30.0
    
    # Probar backends en orden de preferencia para mejor rendimiento
    backends_to_try = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto"),
    ]
    
    for backend_id, backend_name in backends_to_try:
        try:
            logger.info("Abriendo webcam %d con backend %s...", index, backend_name)
            cap = cv2.VideoCapture(index, backend_id)
            
            if cap.isOpened():
                # Configurar resolución y FPS
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_resolution[1])
                cap.set(cv2.CAP_PROP_FPS, preferred_fps)
                
                # Configuraciones adicionales para mejor rendimiento
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Buffer pequeño para baja latencia
                
                # En DirectShow, desactivar autoexposición puede mejorar fps consistentes
                if backend_id == cv2.CAP_DSHOW:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode
                
                # Verificar que funciona
                ret, _ = cap.read()
                if ret:
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    logger.info(
                        "Webcam %d abierta: %dx%d @ %.1f fps (backend: %s)",
                        index, actual_width, actual_height, actual_fps, backend_name
                    )
                    return cap
                
                cap.release()
                
        except Exception as e:
            logger.debug("Error con backend %s: %s", backend_name, e)
            continue
    
    # Fallback: intentar sin especificar backend
    logger.warning("Usando fallback para webcam %d", index)
    return cv2.VideoCapture(index)


def es_fuente_live(source: Source) -> bool:
    """
    Determina si una fuente es en directo (webcam) o un archivo.
    
    Args:
        source: Fuente de vídeo (int para webcam, str para archivo/URL).
        
    Returns:
        True si es una fuente en directo.
    """
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        # URLs de streaming RTSP/HTTP también son live
        lower = source.lower()
        if lower.startswith(('rtsp://', 'http://', 'https://', 'rtmp://')):
            return True
        # Números como string también son webcams
        if source.strip().isdigit():
            return True
    return False


def normalizar_source(source: Optional[Union[str, int]]) -> Optional[Source]:
    """
    Convierte los argumentos CLI en un tipo aceptable por OpenCV:
      - "0" → 0 (webcam)
      - 0 (int) → 0 (ya es webcam, se devuelve tal cual)
      - resto → se devuelve tal cual
    """
    if source is None:
        return None
    # Si ya es un entero, es un índice de webcam válido
    if isinstance(source, int):
        return source
    # Es string, procesar normalmente
    texto = source.strip()
    if not texto:
        return None
    if texto.isdigit():
        return int(texto)
    return texto


def _intentar_abrir_con_hw_accel(
    source: Source,
    logger: Logger,
) -> Optional[cv2.VideoCapture]:
    """
    Intenta abrir el vídeo con decodificación por hardware (NVDEC/D3D11).
    Devuelve None si no se pudo o si la fuente es una webcam.
    """
    # Solo intentar HW accel para archivos de vídeo, no webcams
    if isinstance(source, int):
        return None
    
    # Lista de backends con aceleración por hardware a probar
    # En Windows con NVIDIA: D3D11VA es lo más compatible
    hw_backends = [
        # FFMPEG con aceleración CUDA/NVDEC
        (cv2.CAP_FFMPEG, "FFMPEG+HW"),
        # DirectShow con D3D11 (Windows)
        (cv2.CAP_MSMF, "MSMF/D3D11"),
    ]
    
    for backend, nombre in hw_backends:
        try:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                # Intentar habilitar decodificación por hardware si está disponible
                # CAP_PROP_HW_ACCELERATION: 0=ninguna, 1=cualquiera, específicos varían
                if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                    # Intentar VIDEO_ACCELERATION_ANY (1) o D3D11 específico
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)
                
                # Verificar que realmente funciona leyendo un frame
                ret, _ = cap.read()
                if ret:
                    # Rebobinar al inicio
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.info("Decodificación HW habilitada con backend %s", nombre)
                    return cap
                cap.release()
        except Exception as e:
            logger.debug("Backend %s no disponible: %s", nombre, e)
    
    return None


def abrir_fuente_video(
    source: Source,
    reintentos: int = 1,
    delay_segundos: float = 1.0,
    logger: Optional[Logger] = None,
    preferir_hw_accel: bool = True,
) -> cv2.VideoCapture:
    """
    Intenta abrir la fuente solicitada con varios reintentos controlados.
    
    Si preferir_hw_accel=True (por defecto), intenta primero usar decodificación
    por hardware (NVDEC/D3D11) para aprovechar la GPU y liberar CPU.
    
    Siempre devuelve un objeto VideoCapture; el caller debe comprobar isOpened().
    """
    logger = logger or logging.getLogger("sussy.fuentes")
    intentos = max(1, reintentos)

    # Configurar FFMPEG para preferir decodificación por hardware
    # Esto afecta al backend FFMPEG de OpenCV
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;d3d11va")
    
    for intento in range(1, intentos + 1):
        logger.info("Abriendo fuente (%s/%s): %s", intento, intentos, source)
        
        # Primero intentar con aceleración por hardware
        if preferir_hw_accel:
            cap = _intentar_abrir_con_hw_accel(source, logger)
            if cap is not None:
                return cap
        
        # Fallback: abrir sin aceleración por hardware
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            logger.info("Fuente abierta correctamente (decodificación por software).")
            return cap

        logger.warning("No se pudo abrir la fuente en el intento %s.", intento)
        cap.release()

        if intento < intentos:
            time.sleep(max(0.0, delay_segundos))

    # Como fallback devolvemos el último VideoCapture aunque esté cerrado;
    # el caller decidirá si aborta o si quiere reintentar manualmente.
    return cv2.VideoCapture(source)

