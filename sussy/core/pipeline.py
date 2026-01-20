import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from sussy.config import Config
from sussy.core.dataset_utils import guardar_crop
from sussy.core.deteccion import detectar, combinar_detecciones, analizar_recorte
from sussy.core.entorno import configurar_entorno_privado
from sussy.core.estabilidad_camara import MonitorEstabilidadCamara, DiagnosticoEstabilidad
from sussy.core.eventos import GestorEventos
from sussy.core.fuentes import normalizar_source, abrir_fuente_video
from sussy.core.movimiento import DetectorMovimiento
from sussy.core.prediccion import PredictorMovimiento
from sussy.core.presets import (
    aplicar_preset_camara,
    aplicar_preset_rendimiento,
    presets_disponibles,
    presets_rendimiento_disponibles,
)
from sussy.core.relevancia import EvaluadorRelevancia
from sussy.core.registro import RegistradorCSV
from sussy.core.seguimiento import TrackerSimple
from sussy.core.utilidades_iou import calcular_iou
from sussy.core.texto import dibujar_texto
from sussy.core.apariencia import crear_extractor_apariencia

LOGGER = configurar_entorno_privado(logging.getLogger("sussy.pipeline"))


@dataclass
class FrameResult:
    """
    Resultado de un paso del pipeline.
    frame: frame anotado si annotate=True; de lo contrario, frame original.
    tracks: lista de tracks/detecciones normalizados.
    eventos: eventos de alto nivel disparados en este frame (inicio/fin alerta).
    estado: métricas y banderas útiles para la UI (fps, idx, bloqueo por cámara, etc.).
    """

    frame: np.ndarray
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    eventos: List[Dict[str, Any]] = field(default_factory=list)
    estado: Dict[str, Any] = field(default_factory=dict)


class SussyPipeline:
    """
    Pipeline reutilizable y desacoplado de la UI. Recibe una fuente de vídeo,
    procesa frame a frame y devuelve resultados estructurados para ser consumidos
    por cualquier interfaz (CLI, Qt, etc.).

    Notas de diseño:
    - No crea ventanas ni maneja teclado/ratón.
    - Las configuraciones se aplican en caliente vía presets u overrides.
    - La anotación del frame (cajas y textos) es opcional para ahorrar coste si
      el consumidor solo quiere metadatos.
    """

    def __init__(
        self,
        annotate: bool = True,
        eventos_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.annotate = annotate
        self.eventos_callback = eventos_callback

        # Estado runtime
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.total_frames: int = -1
        self.indice_frame_actual: int = 0
        self.skip_frames: int = 1

        # Componentes
        self.tracker: Optional[TrackerSimple] = None
        self.crear_tracker: Optional[Callable[[], TrackerSimple]] = None
        self.predictor_movimiento: Optional[PredictorMovimiento] = None
        self.detector_movimiento: Optional[DetectorMovimiento] = None
        self.monitor_estabilidad: Optional[MonitorEstabilidadCamara] = None
        self.evaluador_relevancia = EvaluadorRelevancia()
        self.gestor_eventos = GestorEventos(
            backend_url=Config.BACKEND_URL,
            timeout=Config.BACKEND_TIMEOUT,
            simular_en_log=Config.SIMULAR_EVENTOS_EN_LOG,
        )
        self.gestor_estado: Optional["GestorEstadoDeteccion"] = None  # type: ignore
        self.logger: Optional[RegistradorCSV] = None
        self.media_inferencia_ms: float = 0.0

        # Flags de movimiento/estado de cámara
        self.camara_en_movimiento = False
        self.razon_movimiento = ""
        self.camara_mov_monitor = False
        self.camara_mov_rafaga = False
        self.camara_mov_rafaga_restante = 0
        self.rafaga_consecutiva = 0
        self.camara_mov_anomalia = False
        self.camara_mov_anomalia_restante = 0
        self.anomalia_consecutiva = 0
        self.anomalia_motivo_actual = "anomalia"
        self.numero_detecciones_anomalia = 0
        self.diagnostico_camara: Optional[DiagnosticoEstabilidad] = None
        self.ultimo_diag_monitor: Optional[DiagnosticoEstabilidad] = None
        self.motivo_monitor_actual = "estable"
        self.zoom_cooldown_restante = 0
        self.prev_frame_zoom_small: Optional[np.ndarray] = None

        # Persistencia para pausa/visualización
        self.ultimos_tracks: List[Dict[str, Any]] = []
        self.ultimo_frame: Optional[np.ndarray] = None

    # -------------------- Configuración / presets -------------------- #
    def set_config(self, overrides: Dict[str, Any]) -> None:
        """Aplica overrides directos sobre Config (uso avanzado)."""
        for attr, value in overrides.items():
            if not hasattr(Config, attr):
                raise AttributeError(f"Config no tiene el atributo '{attr}'")
            setattr(Config, attr, value)

    def aplicar_preset_camara(self, nombre: str, overrides: Optional[Dict[str, Any]] = None):
        return aplicar_preset_camara(nombre, overrides=overrides)

    def aplicar_preset_rendimiento(self, nombre: str, overrides: Optional[Dict[str, Any]] = None):
        return aplicar_preset_rendimiento(nombre, overrides=overrides)

    # --------------------------- Ciclo de vida ------------------------ #
    def start(
        self,
        source: Optional[str],
        *,
        cam_preset: Optional[str] = None,
        perf_preset: Optional[str] = None,
        skip_frames: Optional[int] = None,
        log_csv: Optional[str] = None,
    ) -> None:
        """
        Inicializa el pipeline:
        - Aplica presets si se piden.
        - Normaliza y abre la fuente.
        - Prepara módulos (tracker, detector de movimiento, monitor, predictor).
        """
        # Presets
        preset_aplicado = None
        if cam_preset:
            preset_aplicado = self.aplicar_preset_camara(
                cam_preset, overrides=getattr(Config, "CAMARA_PRESET_OVERRIDES", None)
            )
            LOGGER.info("Preset de cámara aplicado: %s", preset_aplicado.nombre)

        preset_rend_aplicado = None
        if perf_preset:
            preset_rend_aplicado = self.aplicar_preset_rendimiento(
                perf_preset, overrides=getattr(Config, "RENDIMIENTO_PRESET_OVERRIDES", None)
            )
            LOGGER.info("Preset de rendimiento aplicado: %s", preset_rend_aplicado.nombre)

        # Skip frames
        self.skip_frames = skip_frames if skip_frames is not None else getattr(Config, "SKIP_FRAMES_DEFECTO", 1)
        if self.skip_frames < 1:
            self.skip_frames = 1

        # Normalizar fuente (usar 'is None' porque 0 es un índice válido de webcam)
        fuente_normalizada = normalizar_source(source)
        if fuente_normalizada is None:
            fuente_normalizada = normalizar_source(Config.FUENTE_POR_DEFECTO)
        if fuente_normalizada is None:
            raise ValueError("No se especificó fuente (--source) ni Config.FUENTE_POR_DEFECTO")

        # Abrir fuente
        self.cap = abrir_fuente_video(
            fuente_normalizada,
            reintentos=Config.FUENTE_MAX_REINTENTOS,
            delay_segundos=Config.FUENTE_REINTENTO_DELAY,
            logger=LOGGER,
        )
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {fuente_normalizada}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.indice_frame_actual = 0

        # Componentes
        self._inicializar_componentes()

        # Logger opcional
        if log_csv:
            self.logger = RegistradorCSV(log_csv)

    def stop(self) -> None:
        """Libera recursos abiertos."""
        if self.cap:
            self.cap.release()
        self.cap = None
        if self.logger:
            self.logger.cerrar()
        self.logger = None

    def seek(self, frame_idx: int) -> None:
        """
        Coloca el puntero de la fuente en un frame concreto y reinicia
        componentes sensibles (tracker, predicción, detectores de movimiento).
        """
        if not self.cap:
            return
        frame_idx = max(0, frame_idx)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.indice_frame_actual = frame_idx
        self._reiniciar_componentes_movimiento(reset_tracker=True)
        self.ultimos_tracks = []
        self.ultimo_frame = None

    # -------------------------- Paso de proceso ----------------------- #
    def process_next(self) -> FrameResult:
        """
        Procesa el siguiente frame de la fuente.
        Devuelve FrameResult con frame anotado opcionalmente, tracks y estado.
        Si se alcanza el fin de la fuente, marca finished=True en estado.
        """
        if not self.cap:
            raise RuntimeError("Pipeline no inicializado. Llama a start().")

        ret, frame = self.cap.read()
        if not ret:
            estado = {"finished": True, "frame_idx": self.indice_frame_actual}
            # Devolvemos el último frame conocido o un placeholder negro
            frame_out = (
                self.ultimo_frame
                if self.ultimo_frame is not None
                else np.zeros((480, 640, 3), dtype=np.uint8)
            )
            return FrameResult(frame=frame_out, tracks=[], eventos=[], estado=estado)

        self.indice_frame_actual = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Gestión de estabilidad/zoom/movimiento de cámara
        self._actualizar_estado_camara(frame)

        tracks: List[Dict[str, Any]] = []
        eventos: List[Dict[str, Any]] = []

        if not self.camara_en_movimiento and (self.indice_frame_actual % self.skip_frames == 0):
            tracks, eventos = self._procesar_frame(frame)
            self.ultimos_tracks = tracks
        elif self.camara_en_movimiento:
            # Si la cámara se mueve, vaciamos tracks para evitar arrastres
            self.ultimos_tracks = []
        else:
            # Frame saltado por skip: mantenemos los últimos tracks para no perderlos visualmente
            tracks = self.ultimos_tracks

        # Anotar frame si se solicita
        frame_out = frame.copy()
        if self.annotate and Config.MOSTRAR_TRACKS:
            from sussy.core.visualizacion import dibujar_tracks

            dibujar_tracks(frame_out, tracks if tracks else self.ultimos_tracks)
            if self.camara_en_movimiento:
                self._dibujar_overlay_camara(frame_out)

        self.ultimo_frame = frame_out.copy()

        estado = {
            "fps": self.fps,
            "frame_idx": self.indice_frame_actual,
            "total_frames": self.total_frames,
            "camara_en_movimiento": self.camara_en_movimiento,
            "razon_movimiento": self.razon_movimiento,
            "motivo_monitor": self.motivo_monitor_actual,
            "diagnostico_camara": self.diagnostico_camara,
            "finished": False,
        }

        return FrameResult(frame=frame_out, tracks=tracks, eventos=eventos, estado=estado)

    # ------------------------ Lógica interna -------------------------- #
    def _inicializar_componentes(self) -> None:
        """Prepara tracker, detector de movimiento, monitor y predictor."""
        self.tracker = None
        self.predictor_movimiento = None
        self.crear_tracker = None
        self.extractor_apariencia = None

        if Config.USAR_TRACKER:
            # Configuración de estabilización de clases y Re-ID
            usar_estab = getattr(Config, "TRACKER_USAR_ESTABILIZACION_CLASES", True)
            usar_reid = getattr(Config, "TRACKER_USAR_REID", False)
            
            def _crear_tracker() -> TrackerSimple:
                tracker = TrackerSimple(
                    max_dist=Config.TRACKER_MATCH_DIST,
                    max_frames_lost=Config.TRACKER_MAX_FRAMES_LOST,
                    iou_threshold=Config.TRACKER_IOU_THRESHOLD,
                    usar_prediccion=getattr(Config, "TRACKER_USAR_PREDICCION", True),
                    prediccion_factor=getattr(Config, "TRACKER_PREDICCION_FACTOR", 0.8),
                    aceleracion_max=getattr(Config, "TRACKER_ACELERACION_MAX", 5.0),
                    # Nuevos parámetros v3
                    usar_estabilizacion_clases=usar_estab,
                    clase_decay=getattr(Config, "TRACKER_CLASE_DECAY", 0.92),
                    clase_min_frames=getattr(Config, "TRACKER_CLASE_MIN_FRAMES", 3),
                    usar_reid=usar_reid,
                    reid_max_edad=getattr(Config, "TRACKER_REID_MAX_EDAD", 300),
                    reid_umbral=getattr(Config, "TRACKER_REID_UMBRAL", 0.60),
                )
                return tracker

            self.crear_tracker = _crear_tracker
            self.tracker = self.crear_tracker()
            
            # Crear extractor de apariencia si Re-ID está activado
            if usar_reid:
                try:
                    tipo_extractor = getattr(Config, "TRACKER_APARIENCIA_TIPO", "simple")
                    self.extractor_apariencia = crear_extractor_apariencia(tipo=tipo_extractor)
                    self.tracker.set_extractor_apariencia(self.extractor_apariencia)
                    LOGGER.info(
                        "Re-ID activado con extractor '%s' (umbral=%.2f, max_edad=%d frames)",
                        tipo_extractor,
                        getattr(Config, "TRACKER_REID_UMBRAL", 0.60),
                        getattr(Config, "TRACKER_REID_MAX_EDAD", 300),
                    )
                except Exception as e:
                    LOGGER.warning("No se pudo inicializar extractor de apariencia: %s", e)
                    self.extractor_apariencia = None
            
            if usar_estab:
                LOGGER.info(
                    "Estabilización de clases activada (decay=%.2f, min_frames=%d)",
                    getattr(Config, "TRACKER_CLASE_DECAY", 0.92),
                    getattr(Config, "TRACKER_CLASE_MIN_FRAMES", 3),
                )

            if Config.USAR_PREDICCION_MOVIMIENTO:
                self.predictor_movimiento = PredictorMovimiento(
                    frames_adelante=Config.PREDICCION_FRAMES_ADELANTE,
                    padding_pct=Config.PREDICCION_PADDING_PCT,
                    vel_min=Config.PREDICCION_VEL_MIN,
                    max_zonas=Config.PREDICCION_MAX_ZONAS,
                )

        if Config.USAR_DETECTOR_MOVIMIENTO:
            self.detector_movimiento = DetectorMovimiento(
                umbral_diff=Config.MOVIMIENTO_UMBRAL,
                min_area=Config.MOVIMIENTO_AREA_MIN,
                contraste_min=Config.MOVIMIENTO_CONTRASTE_MIN,
                min_frames_vivos=Config.MOVIMIENTO_MIN_FRAMES,
                min_desplazamiento=Config.MOVIMIENTO_MIN_DESPLAZAMIENTO,
                max_detecciones=Config.MOVIMIENTO_MAX_DETECCIONES,
            )

        if Config.USAR_MONITOR_ESTABILIDAD:
            self.monitor_estabilidad = MonitorEstabilidadCamara(
                escala=Config.CAMARA_ESCALA_ANALISIS,
                max_desplazamiento_px=Config.CAMARA_MAX_DESPLAZAMIENTO_PX,
                min_puntos=Config.CAMARA_MIN_PUNTOS,
                max_ratio_perdidos=Config.CAMARA_MAX_RATIO_PERDIDOS,
                frames_inestables_para_disparar=Config.CAMARA_FRAMES_INESTABLES,
                frames_estables_para_recuperar=Config.CAMARA_FRAMES_ESTABLES,
                activar_inmediato=Config.CAMARA_ACTIVACION_INMEDIATA,
                max_cambio_escala=Config.CAMARA_MAX_CAMBIO_ESCALA,
                max_ratio_diferencia=Config.CAMARA_MAX_RATIO_DIFERENCIA,
            )

        from sussy.core.estado_alerta import GestorEstadoDeteccion
        self.gestor_estado = GestorEstadoDeteccion(
            frames_necesarios=Config.ALERTA_FRAMES_CONSECUTIVOS,
            tiempo_rearme=Config.ALERTA_TIEMPO_REARME,
            duracion_max=Config.ALERTA_DURACION_MAX,
        )

    def _reiniciar_componentes_movimiento(self, reset_tracker: bool = False) -> None:
        """Resetea contadores y módulos sensibles al movimiento/seek."""
        if self.detector_movimiento:
            self.detector_movimiento.resetear()
        if reset_tracker and self.crear_tracker:
            self.tracker = self.crear_tracker()
        if self.predictor_movimiento:
            self.predictor_movimiento.consumir_zonas()
        self.ultimos_tracks = []
        self.camara_mov_rafaga_restante = 0
        self.rafaga_consecutiva = 0
        self.camara_mov_anomalia_restante = 0
        self.anomalia_consecutiva = 0
        self.zoom_cooldown_restante = 0
        self.anomalia_motivo_actual = "anomalia"
        self.numero_detecciones_anomalia = 0
        self.camara_en_movimiento = False
        self.razon_movimiento = ""

    def _activar_bloqueo_movimiento(self, razon: str) -> None:
        if not self.camara_en_movimiento:
            self._reiniciar_componentes_movimiento()
        self.camara_en_movimiento = True
        self.razon_movimiento = razon
        if razon.startswith("monitor:zoom") and self.monitor_estabilidad:
            self.monitor_estabilidad.resetear()

    def _desactivar_bloqueo_movimiento(self) -> None:
        if self.camara_en_movimiento:
            self.camara_en_movimiento = False
            self.razon_movimiento = ""
            self.camara_mov_rafaga_restante = 0
            self.camara_mov_anomalia_restante = 0

    def _actualizar_estado_camara(self, frame: np.ndarray) -> None:
        """Gestiona zoom rápido, monitor de estabilidad y flags de bloqueo."""
        # Detección rápida de zoom
        zoom_fast_detectado = False
        if Config.CAMARA_ZOOM_FAST_RATIO > 0:
            gray_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.zoom_cooldown_restante == 0:
                gray_small = cv2.resize(
                    gray_small,
                    (max(32, int(gray_small.shape[1] * 0.1)), max(18, int(gray_small.shape[0] * 0.1))),
                    interpolation=cv2.INTER_AREA,
                )
                if self.prev_frame_zoom_small is not None:
                    diff_fast = cv2.absdiff(self.prev_frame_zoom_small, gray_small)
                    ratio_fast = float(np.count_nonzero(diff_fast > 25)) / float(diff_fast.size)
                    if ratio_fast >= Config.CAMARA_ZOOM_FAST_RATIO:
                        zoom_fast_detectado = True
                        self.zoom_cooldown_restante = Config.CAMARA_ZOOM_COOLDOWN_FRAMES
                        self._activar_bloqueo_movimiento("monitor:zoom_fast")
                        self.camara_mov_monitor = True
                        self.motivo_monitor_actual = "zoom_fast"
                        self.diagnostico_camara = None
                        self.ultimo_diag_monitor = None
                        self.prev_frame_zoom_small = gray_small
                        return
                self.prev_frame_zoom_small = gray_small

        # RAFAGA / ANOMALIA timers
        if self.camara_mov_rafaga_restante > 0:
            self.camara_mov_rafaga_restante -= 1
            self.camara_mov_rafaga = True
        else:
            self.camara_mov_rafaga = False

        if self.camara_mov_anomalia_restante > 0:
            self.camara_mov_anomalia_restante -= 1
            self.camara_mov_anomalia = True
        else:
            self.camara_mov_anomalia = False

        if self.zoom_cooldown_restante > 0:
            self.zoom_cooldown_restante -= 1
            self.camara_mov_monitor = True
            self.motivo_monitor_actual = "zoom_hold"
            self.diagnostico_camara = self.ultimo_diag_monitor
        elif self.monitor_estabilidad:
            self.camara_mov_monitor = self.monitor_estabilidad.actualizar(frame)
            self.diagnostico_camara = self.monitor_estabilidad.diagnostico
            self.ultimo_diag_monitor = self.diagnostico_camara
            self.motivo_monitor_actual = getattr(self.monitor_estabilidad, "motivo", "monitor")
            if self.camara_mov_monitor and self.motivo_monitor_actual.startswith("zoom"):
                self.zoom_cooldown_restante = Config.CAMARA_ZOOM_COOLDOWN_FRAMES
        else:
            self.camara_mov_monitor = False
            self.diagnostico_camara = None
            self.ultimo_diag_monitor = None
            self.motivo_monitor_actual = "sin_monitor"

        if self.camara_mov_monitor:
            self._activar_bloqueo_movimiento(f"monitor:{self.motivo_monitor_actual}")
        elif self.camara_mov_anomalia:
            self.diagnostico_camara = None
            self._activar_bloqueo_movimiento(self.anomalia_motivo_actual or "anomalia")
        elif self.camara_mov_rafaga:
            self.diagnostico_camara = None
            self._activar_bloqueo_movimiento("rafaga")
        else:
            self._desactivar_bloqueo_movimiento()

    def _procesar_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Ejecuta detección+tracking+eventos sobre un frame."""
        detecciones_yolo: List[Dict[str, Any]] = []
        if Config.USAR_YOLO:
            import time

            t0 = time.perf_counter()
            detecciones_yolo = detectar(
                frame,
                conf_umbral=Config.YOLO_CONF_UMBRAL,
                modelo_path=Config.YOLO_MODELO,
                clases_permitidas=Config.YOLO_CLASES_PERMITIDAS,
            ) or []
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if self.media_inferencia_ms == 0.0:
                self.media_inferencia_ms = elapsed_ms
            else:
                self.media_inferencia_ms = self.media_inferencia_ms * 0.8 + elapsed_ms * 0.2

        # Zonas anticipadas
        if self.predictor_movimiento:
            zonas_predichas = self.predictor_movimiento.consumir_zonas()
            detecciones_predichas: List[Dict[str, Any]] = []
            for zona in zonas_predichas:
                zx1, zy1, zx2, zy2 = zona.as_tuple()
                det_pred = analizar_recorte(
                    frame,
                    zx1,
                    zy1,
                    zx2,
                    zy2,
                    conf_umbral=max(0.1, Config.YOLO_CONF_UMBRAL - 0.2),
                    modelo_path=Config.YOLO_MODELO,
                    clases_permitidas=Config.YOLO_CLASES_PERMITIDAS,
                )
                if det_pred:
                    det_pred["descripcion"] = "Zona anticipada"
                    det_pred.setdefault("score", 0.4)
                    detecciones_predichas.append(det_pred)
            if detecciones_predichas:
                detecciones_yolo.extend(detecciones_predichas)

        # Movimiento clásico
        detecciones_mov: List[Dict[str, Any]] = []
        if self.detector_movimiento and not self.camara_en_movimiento:
            detecciones_mov = self.detector_movimiento.actualizar(frame)
            total_movimientos = getattr(self.detector_movimiento, "ultimo_conteo_detecciones", 0)
            if Config.MOVIMIENTO_RAFAGA_BLOBS:
                if total_movimientos >= Config.MOVIMIENTO_RAFAGA_BLOBS:
                    self.rafaga_consecutiva += 1
                else:
                    self.rafaga_consecutiva = 0
                activacion = max(1, Config.MOVIMIENTO_RAFAGA_FRAMES_ACTIVACION)
                if self.rafaga_consecutiva >= activacion:
                    self.diagnostico_camara = None
                    self._activar_bloqueo_movimiento("rafaga")
                    self.camara_mov_rafaga_restante = max(
                        self.camara_mov_rafaga_restante,
                        Config.MOVIMIENTO_RAFAGA_FRAMES,
                    )
                    self.camara_mov_rafaga = True
                    self.rafaga_consecutiva = 0
                    return [], []
            else:
                self.rafaga_consecutiva = 0

        # Filtro IA sobre blobs de movimiento huérfanos + guardado de crops
        nuevas_detecciones_ia: List[Dict[str, Any]] = []
        usar_filtro_mov = Config.USAR_FILTRO_IA_EN_MOVIMIENTO
        if (
            usar_filtro_mov
            and getattr(Config, "AUTO_DESACTIVAR_FILTRO_IA", False)
            and self.media_inferencia_ms > getattr(Config, "LIMITE_MS_INFERENCIA", 120.0)
        ):
            usar_filtro_mov = False

        for d_mov in detecciones_mov:
            solapa_yolo = any(calcular_iou(d_mov, d_yolo) > 0.1 for d_yolo in detecciones_yolo)
            if solapa_yolo:
                continue

            detectado_en_crop = None
            if usar_filtro_mov:
                detectado_en_crop = analizar_recorte(
                    frame,
                    d_mov["x1"],
                    d_mov["y1"],
                    d_mov["x2"],
                    d_mov["y2"],
                    conf_umbral=Config.YOLO_CONF_UMBRAL - 0.1,
                    modelo_path=Config.YOLO_MODELO,
                    clases_permitidas=Config.YOLO_CLASES_PERMITIDAS,
                    padding_pct=Config.MOVIMIENTO_CROP_PADDING_PCT,
                )
                if detectado_en_crop:
                    nuevas_detecciones_ia.append(detectado_en_crop)

            if Config.GUARDAR_CROPS_ENTRENAMIENTO:
                clase_guardar = detectado_en_crop["clase"] if detectado_en_crop else "unknown"
                # Usar detección validada si existe, sino la de movimiento
                det_para_guardar = detectado_en_crop if detectado_en_crop else d_mov
                guardar_crop(
                    frame,
                    d_mov["x1"],
                    d_mov["y1"],
                    d_mov["x2"],
                    d_mov["y2"],
                    clase_guardar,
                    Config.RUTA_DATASET_CAPTURE,
                    padding_pct=Config.MOVIMIENTO_CROP_PADDING_PCT,
                    det=det_para_guardar,
                    frame_idx=self.indice_frame_actual,
                    fuente=str(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.cap else "",
                )

        detecciones_yolo.extend(nuevas_detecciones_ia)

        detecciones = combinar_detecciones(detecciones_yolo, detecciones_mov)
        # Pasar tracks anteriores para análisis de trayectoria
        detecciones = self.evaluador_relevancia.filtrar(
            detecciones, frame.shape, tracks=self.ultimos_tracks
        )

        # Anomalías
        anomalia_detectada = False
        total_detecciones = len(detecciones)
        if Config.MOVIMIENTO_ANOMALIA_TOTAL and total_detecciones >= Config.MOVIMIENTO_ANOMALIA_TOTAL:
            anomalia_detectada = True
        if Config.MOVIMIENTO_ANOMALIA_POSIBLE_DRON:
            total_pd = sum(1 for d in detecciones if d.get("clase") == "posible_dron")
            if total_pd >= Config.MOVIMIENTO_ANOMALIA_POSIBLE_DRON:
                anomalia_detectada = True
        if anomalia_detectada:
            self.anomalia_consecutiva += 1
        else:
            self.anomalia_consecutiva = 0

        activacion_anomalia = max(1, Config.MOVIMIENTO_ANOMALIA_FRAMES_ACTIVACION)
        if self.anomalia_consecutiva >= activacion_anomalia:
            # Determinar tipo de anomalía y conteo apropiado
            total_pd = sum(1 for d in detecciones if d.get("clase") == "posible_dron")
            es_anomalia_pd = (
                Config.MOVIMIENTO_ANOMALIA_POSIBLE_DRON
                and total_pd >= Config.MOVIMIENTO_ANOMALIA_POSIBLE_DRON
            )
            if es_anomalia_pd:
                self.anomalia_motivo_actual = "anomalia_posible_dron"
                conteo_anomalia = total_pd
            else:
                self.anomalia_motivo_actual = "anomalia"
                conteo_anomalia = total_detecciones
            
            # IMPORTANTE: Activar bloqueo ANTES de asignar contador
            # porque _activar_bloqueo_movimiento puede resetear el contador
            self._activar_bloqueo_movimiento(self.anomalia_motivo_actual)
            self.numero_detecciones_anomalia = conteo_anomalia
            
            self.camara_mov_anomalia_restante = max(
                self.camara_mov_anomalia_restante,
                Config.MOVIMIENTO_ANOMALIA_FRAMES_ENFRIAMIENTO,
            )
            self.camara_mov_anomalia = True
            self.diagnostico_camara = None
            self.anomalia_consecutiva = 0
            return [], []

        # Tracking
        tracks: List[Dict[str, Any]] = []
        if self.tracker:
            # Pasar frame para Re-ID (extracción de embeddings visuales)
            tracks = self.tracker.actualizar(detecciones, frame=frame)
            if self.predictor_movimiento:
                self.predictor_movimiento.preparar_zonas(tracks, frame.shape)
        else:
            for d in detecciones:
                tracks.append(
                    {
                        "id": -1,
                        "box": [d["x1"], d["y1"], d["x2"], d["y2"]],
                        "clase": d.get("clase", "desc"),
                        "score": d.get("score", 0.0),
                    }
                )

        if self.logger is not None:
            self.logger.registrar(self.indice_frame_actual, tracks)

        eventos = self._procesar_eventos_alerta(tracks)
        return tracks, eventos

    def _procesar_eventos_alerta(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Traduce tracks a eventos de alto nivel y despacha callbacks."""
        if not self.gestor_estado:
            return []

        from sussy.core.estado_alerta import GestorEstadoDeteccion
        gestor_estado: GestorEstadoDeteccion = self.gestor_estado

        objetivos = [
            {
                "id": track.get("id", -1),
                "clase": track.get("clase", "desconocido"),
                "score": float(track.get("score", 0.0)),
            }
            for track in tracks
            if (not Config.CLASES_ALERTA) or (track.get("clase") in Config.CLASES_ALERTA)
        ]

        evento = gestor_estado.actualizar(bool(objetivos))
        if not evento:
            return []

        snapshot = gestor_estado.snapshot()
        contexto = {
            "frame": self.indice_frame_actual,
            "fps": self.fps,
            "objetivos": objetivos,
            "estado": {
                "activo": snapshot.activo,
                "frames_consecutivos": snapshot.frames_consecutivos,
                "ultimo_objetivo": snapshot.ultimo_objetivo,
                "inicio_alerta": snapshot.inicio_alerta,
            },
        }

        eventos_emitidos: List[Dict[str, Any]] = []
        if evento == "inicio":
            self.gestor_eventos.notificar_inicio(contexto)
            eventos_emitidos.append({"tipo": "inicio_alerta", **contexto})
        elif evento == "fin":
            self.gestor_eventos.notificar_fin(contexto)
            eventos_emitidos.append({"tipo": "fin_alerta", **contexto})

        if self.eventos_callback:
            for e in eventos_emitidos:
                self.eventos_callback(e)

        return eventos_emitidos

    def _dibujar_overlay_camara(self, frame: np.ndarray) -> None:
        """Overlay textual indicando bloqueo por movimiento/zoom."""
        motivo = self.razon_movimiento or ""
        if motivo.startswith("monitor"):
            sub = motivo.split(":", 1)[1] if ":" in motivo else "global"
            if sub.startswith("zoom"):
                msg = "CÁMARA EN MOVIMIENTO (zoom detectado)"
            else:
                msg = "CÁMARA EN MOVIMIENTO (monitor global)"
        elif motivo == "rafaga":
            msg = "CÁMARA EN MOVIMIENTO (ráfaga de blobs)"
        elif motivo == "anomalia":
            num_det = self.numero_detecciones_anomalia
            if not Config.USAR_MONITOR_ESTABILIDAD:
                msg = f"DETECCIONES MOVIMIENTO MASIVAS: {num_det}"
            else:
                msg = f"CÁMARA EN MOVIMIENTO (detecciones masivas: {num_det})"
        elif motivo == "anomalia_posible_dron":
            num_pd = self.numero_detecciones_anomalia
            msg = f"CÁMARA EN MOVIMIENTO (posible_dron: {num_pd})"
        else:
            msg = "CÁMARA EN MOVIMIENTO"

        font_path_ui = Config.UI_FONT_PATH
        dibujar_texto(
            frame,
            msg,
            (20, 40),
            color=(0, 0, 255),
            font_scale=0.6,
            thickness=2,
            font_path=font_path_ui,
        )

        if motivo.startswith("monitor") and self.diagnostico_camara:
            diag = self.diagnostico_camara
            sub = motivo.split(":", 1)[1] if ":" in motivo else "global"
            if sub.startswith("zoom"):
                diag_msg = f"Escala x{diag.factor_escala:.3f} diff={diag.ratio_pixeles:.2f}"
            else:
                diag_msg = f"Δpx={diag.desplazamiento_medio:.1f} pts={diag.puntos_validos}/{diag.total_puntos}"
            dibujar_texto(
                frame,
                diag_msg,
                (20, 65),
                color=(0, 0, 255),
                font_scale=0.55,
                thickness=1,
                font_path=font_path_ui,
            )


__all__ = [
    "SussyPipeline",
    "FrameResult",
    "presets_disponibles",
    "presets_rendimiento_disponibles",
]

