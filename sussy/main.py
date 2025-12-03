import argparse
import logging
from typing import Optional, List, Dict

import cv2
import numpy as np

from sussy.config import Config
from sussy.core.entorno import configurar_entorno_privado

LOGGER = configurar_entorno_privado(logging.getLogger("sussy.main"))

from sussy.core.deteccion import detectar, combinar_detecciones, analizar_recorte
from sussy.core.estado_alerta import GestorEstadoDeteccion
from sussy.core.eventos import GestorEventos
from sussy.core.fuentes import normalizar_source, abrir_fuente_video
from sussy.core.movimiento import DetectorMovimiento
from sussy.core.presets import aplicar_preset_camara, presets_disponibles
from sussy.core.relevancia import EvaluadorRelevancia
from sussy.core.seguimiento import TrackerSimple
from sussy.core.prediccion import PredictorMovimiento
from sussy.core.visualizacion import dibujar_tracks
from sussy.core.registro import RegistradorCSV
from sussy.core.dataset_utils import guardar_crop
from sussy.core.estabilidad_camara import MonitorEstabilidadCamara, DiagnosticoEstabilidad
from sussy.core.texto import dibujar_texto

PRESETS_CAMARA_DISPONIBLES = presets_disponibles()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema Sussy – Vista previa con pipeline básico (detección + tracking)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Fuente de vídeo (ruta, RTSP/HTTP o índice de webcam). Si no se indica se usa la de Config.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Alias legacy de --source para no romper scripts antiguos.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Procesar solo 1 de cada N fotogramas (por defecto 1 = todos).",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Ruta de archivo CSV para registrar los tracks (opcional).",
    )
    if PRESETS_CAMARA_DISPONIBLES:
        parser.add_argument(
            "--cam-preset",
            type=str,
            choices=PRESETS_CAMARA_DISPONIBLES,
            help=(
                "Preset rápido de cámara (fija/orientable/movil/movil_plus). "
                "Sobrescribe Config antes de iniciar el pipeline."
            ),
        )
    return parser.parse_args()


def dibujar_ui(frame, pausado, frame_actual, total_frames):
    if not Config.MOSTRAR_UI:
        return {}

    alto, ancho = frame.shape[:2]
    
    # Configuración UI
    alto_barra = 60
    y_barra = alto - alto_barra
    
    # 1. Fondo semitransparente
    capa = frame.copy()
    cv2.rectangle(capa, (0, y_barra), (ancho, alto), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(capa, alpha, frame, 1 - alpha, 0, frame)
    
    # 2. Botón Play/Pause (Izquierda)
    btn_x, btn_y = 20, y_barra + 10
    btn_tam = 40
    
    # Dibujar icono
    color_icono = (255, 255, 255)
    if pausado:
        # Icono Play (Triángulo)
        pts = np.array([
            [btn_x + 10, btn_y + 5],
            [btn_x + 10, btn_y + 35],
            [btn_x + 35, btn_y + 20]
        ], np.int32)
        cv2.fillPoly(frame, [pts], color_icono)
    else:
        # Icono Pause (Dos barras)
        cv2.rectangle(frame, (btn_x + 10, btn_y + 5), (btn_x + 18, btn_y + 35), color_icono, -1)
        cv2.rectangle(frame, (btn_x + 22, btn_y + 5), (btn_x + 30, btn_y + 35), color_icono, -1)
        
    # 3. Línea de tiempo
    timeline_x = btn_x + btn_tam + 20
    timeline_y = y_barra + 30
    timeline_w = ancho - timeline_x - 30
    
    # Barra fondo
    cv2.rectangle(frame, (timeline_x, timeline_y - 2), (timeline_x + timeline_w, timeline_y + 2), (100, 100, 100), -1)
    
    # Barra progreso
    if total_frames > 0:
        ancho_progreso = int((frame_actual / total_frames) * timeline_w)
        cv2.rectangle(frame, (timeline_x, timeline_y - 2), (timeline_x + ancho_progreso, timeline_y + 2), (0, 255, 0), -1)
        
        # Círculo indicador
        cv2.circle(frame, (timeline_x + ancho_progreso, timeline_y), 8, (255, 255, 255), -1)

    return {
        "btn": (btn_x, btn_y, btn_tam, btn_tam),
        "timeline": (timeline_x, y_barra, timeline_w, alto_barra)
    }


def procesar_eventos_alerta(
    gestor_estado: GestorEstadoDeteccion,
    gestor_eventos: GestorEventos,
    tracks: List[Dict],
    frame_idx: int,
    fps: float,
) -> None:
    """
    Traduce la lista de tracks en eventos de alto nivel. La lógica es
    deliberadamente independiente del resto del pipeline para facilitar
    pruebas unitarias o sustituciones (por ejemplo, otro orquestador).
    """
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
        return

    snapshot = gestor_estado.snapshot()
    contexto = {
        "frame": frame_idx,
        "fps": fps,
        "objetivos": objetivos,
        "estado": {
            "activo": snapshot.activo,
            "frames_consecutivos": snapshot.frames_consecutivos,
            "ultimo_objetivo": snapshot.ultimo_objetivo,
            "inicio_alerta": snapshot.inicio_alerta,
        },
    }

    if evento == "inicio":
        gestor_eventos.notificar_inicio(contexto)
    elif evento == "fin":
        gestor_eventos.notificar_fin(contexto)


def main() -> None:
    args = parse_args()

    preset_nombre = getattr(args, "cam_preset", None) or Config.CAMARA_PRESET_POR_DEFECTO
    preset_aplicado = None
    if preset_nombre:
        overrides_cfg = getattr(Config, "CAMARA_PRESET_OVERRIDES", None)
        overrides = dict(overrides_cfg) if isinstance(overrides_cfg, dict) and overrides_cfg else None
        try:
            preset_aplicado = aplicar_preset_camara(preset_nombre, overrides=overrides)
            print(f"Preset de cámara seleccionado: {preset_aplicado.nombre}")
            print(f" - {preset_aplicado.descripcion}")
            if overrides:
                claves = ", ".join(sorted(overrides.keys()))
                print(f"Overrides manuales aplicados: {claves}")
        except (ValueError, AttributeError) as exc:
            LOGGER.error("No se pudo aplicar el preset de cámara '%s': %s", preset_nombre, exc)
            return

    print("Sistema Sussy – pipeline básico (INGESTA → DETECCIÓN → TRACKING → VISUALIZACIÓN)")
    fuente_cli = args.source or args.video
    if args.video and not args.source:
        LOGGER.warning("El argumento --video quedará obsoleto; usa --source en su lugar.")

    fuente_normalizada = normalizar_source(fuente_cli) or normalizar_source(Config.FUENTE_POR_DEFECTO)
    if fuente_normalizada is None:
        print("No se especificó ninguna fuente (--source) ni valor por defecto en Config.")
        return

    print(f"Fuente seleccionada: {fuente_normalizada}")
    print("Controles: [Espacio] Pausa/Play, [Click Botón] Pausa/Play, [Timeline] Buscar")
    print("Pulsa 'q' para salir.")

    cap = abrir_fuente_video(
        fuente_normalizada,
        reintentos=Config.FUENTE_MAX_REINTENTOS,
        delay_segundos=Config.FUENTE_REINTENTO_DELAY,
        logger=LOGGER,
    )
    if not cap.isOpened():
        print(f"Error al abrir la fuente: {fuente_normalizada}")
        return

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    if total_frames <= 0:
        LOGGER.info("Fuente sin conteo total (probablemente streaming). Se mostrará timeline relativo.")
    print(f"Resolución: {ancho}x{alto}, Frames: {total_frames if total_frames > 0 else 'desconocido'}, FPS: {fps:.2f}")

    # --- INICIALIZACIÓN MODULAR ---
    tracker = None
    predictor_movimiento = None
    crear_tracker = None
    if Config.USAR_TRACKER:
        def crear_tracker() -> TrackerSimple:
            return TrackerSimple(
                max_dist=Config.TRACKER_MATCH_DIST,
                max_frames_lost=Config.TRACKER_MAX_FRAMES_LOST,
                iou_threshold=Config.TRACKER_IOU_THRESHOLD,
            )

        tracker = crear_tracker()
        print(f"Tracker: ACTIVADO (Paciencia={Config.TRACKER_MAX_FRAMES_LOST}, IoU={Config.TRACKER_IOU_THRESHOLD})")

        if Config.USAR_PREDICCION_MOVIMIENTO:
            predictor_movimiento = PredictorMovimiento(
                frames_adelante=Config.PREDICCION_FRAMES_ADELANTE,
                padding_pct=Config.PREDICCION_PADDING_PCT,
                vel_min=Config.PREDICCION_VEL_MIN,
                max_zonas=Config.PREDICCION_MAX_ZONAS,
            )
            print(
                "Predicción Movimiento: ACTIVADA "
                f"(adelante={Config.PREDICCION_FRAMES_ADELANTE}, "
                f"padding={Config.PREDICCION_PADDING_PCT})"
            )
    else:
        print("Tracker: DESACTIVADO")

    detector_movimiento = None
    if Config.USAR_DETECTOR_MOVIMIENTO:
        detector_movimiento = DetectorMovimiento(
            umbral_diff=Config.MOVIMIENTO_UMBRAL,
            min_area=Config.MOVIMIENTO_AREA_MIN,
            contraste_min=Config.MOVIMIENTO_CONTRASTE_MIN,
            min_frames_vivos=Config.MOVIMIENTO_MIN_FRAMES,
            min_desplazamiento=Config.MOVIMIENTO_MIN_DESPLAZAMIENTO,
            max_detecciones=Config.MOVIMIENTO_MAX_DETECCIONES,
        )
        print(f"Detector Movimiento: ACTIVADO (Umbral={Config.MOVIMIENTO_UMBRAL}, AreaMin={Config.MOVIMIENTO_AREA_MIN})")
    else:
        print("Detector Movimiento: DESACTIVADO")

    monitor_estabilidad = None
    camara_en_movimiento = False
    diagnostico_camara = None
    camara_mov_monitor = False
    camara_mov_rafaga = False
    camara_mov_rafaga_restante = 0
    rafaga_consecutiva = 0
    camara_mov_anomalia = False
    camara_mov_anomalia_restante = 0
    anomalia_consecutiva = 0
    razon_movimiento = ""
    font_path_ui = Config.UI_FONT_PATH
    zoom_cooldown_restante = 0
    ultimo_diag_monitor: Optional[DiagnosticoEstabilidad] = None
    anomalia_motivo_actual = "anomalia"
    prev_frame_zoom_small: Optional[np.ndarray] = None
    motivo_monitor_actual = "estable"
    if Config.USAR_MONITOR_ESTABILIDAD:
        monitor_estabilidad = MonitorEstabilidadCamara(
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
        print(
            "Monitor Estabilidad: ACTIVADO "
            f"(max_shift={Config.CAMARA_MAX_DESPLAZAMIENTO_PX}px, "
            f"min_pts={Config.CAMARA_MIN_PUNTOS})"
        )
    else:
        print("Monitor Estabilidad: DESACTIVADO")

    def reiniciar_componentes_movimiento() -> None:
        nonlocal tracker, ultimos_tracks, camara_mov_rafaga_restante, rafaga_consecutiva, camara_mov_anomalia_restante, anomalia_consecutiva, zoom_cooldown_restante, anomalia_motivo_actual
        if detector_movimiento:
            detector_movimiento.resetear()
        if crear_tracker:
            tracker = crear_tracker()
        if predictor_movimiento:
            predictor_movimiento.consumir_zonas()
        ultimos_tracks = []
        camara_mov_rafaga_restante = 0
        rafaga_consecutiva = 0
        camara_mov_anomalia_restante = 0
        anomalia_consecutiva = 0
        zoom_cooldown_restante = 0
        anomalia_motivo_actual = "anomalia"

    def activar_bloqueo_movimiento(razon: str) -> None:
        nonlocal camara_en_movimiento, razon_movimiento
        if not camara_en_movimiento:
            reiniciar_componentes_movimiento()
        camara_en_movimiento = True
        razon_movimiento = razon
        if razon.startswith("monitor:zoom") and monitor_estabilidad:
            monitor_estabilidad.resetear()

    def desactivar_bloqueo_movimiento() -> None:
        nonlocal camara_en_movimiento, razon_movimiento, camara_mov_rafaga_restante, camara_mov_anomalia_restante
        if camara_en_movimiento:
            camara_en_movimiento = False
            razon_movimiento = ""
            camara_mov_rafaga_restante = 0
            camara_mov_anomalia_restante = 0

    if Config.USAR_YOLO:
        print(f"Detector YOLO: ACTIVADO (Modelo={Config.YOLO_MODELO})")
    else:
        print("Detector YOLO: DESACTIVADO")

    gestor_estado = GestorEstadoDeteccion(
        frames_necesarios=Config.ALERTA_FRAMES_CONSECUTIVOS,
        tiempo_rearme=Config.ALERTA_TIEMPO_REARME,
        duracion_max=Config.ALERTA_DURACION_MAX,
    )
    gestor_eventos = GestorEventos(
        backend_url=Config.BACKEND_URL,
        timeout=Config.BACKEND_TIMEOUT,
        simular_en_log=Config.SIMULAR_EVENTOS_EN_LOG,
    )
    evaluador_relevancia = EvaluadorRelevancia()

    logger: Optional[RegistradorCSV] = None
    if args.log_csv:
        logger = RegistradorCSV(args.log_csv)
        print(f"Registrando tracks en: {logger.ruta}")

    ventana = "Sussy - Vista previa"
    if Config.MOSTRAR_UI or Config.MOSTRAR_TRACKS:
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    
    # Estado
    pausado = False
    indice_frame_actual = 0
    areas_ui = {}
    peticion_busqueda = -1
    
    # Variables para persistencia en pausa
    ultimos_tracks = []
    
    def callback_raton(event, x, y, flags, param):
        nonlocal pausado, peticion_busqueda
        if not Config.MOSTRAR_UI:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Chequear botón Play/Pause
            bx, by, bw, bh = areas_ui.get("btn", (0,0,0,0))
            if bx <= x <= bx + bw and by <= y <= by + bh:
                pausado = not pausado
                return

            # Chequear Timeline
            tx, ty, tw, th = areas_ui.get("timeline", (0,0,0,0))
            if tx <= x <= tx + tw and ty <= y <= ty + th:
                # Calcular porcentaje
                rel_x = x - tx
                pct = max(0.0, min(1.0, rel_x / tw))
                peticion_busqueda = int(pct * total_frames)

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            # Arrastrar en timeline
            tx, ty, tw, th = areas_ui.get("timeline", (0,0,0,0))
            if tx <= x <= tx + tw and ty <= y <= ty + th:
                rel_x = x - tx
                pct = max(0.0, min(1.0, rel_x / tw))
                peticion_busqueda = int(pct * total_frames)

    if Config.MOSTRAR_UI:
        cv2.setMouseCallback(ventana, callback_raton)

    while True:
        # Gestionar Seek (Búsqueda)
        if peticion_busqueda >= 0:
            indice_frame_actual = peticion_busqueda
            cap.set(cv2.CAP_PROP_POS_FRAMES, indice_frame_actual)
            if crear_tracker:
                tracker = crear_tracker()  # Reiniciar tracker al saltar
            ultimos_tracks = []       # Limpiar tracks visuales
            peticion_busqueda = -1
            
            # Si estaba pausado, forzamos lectura para actualizar la vista
            if pausado:
                ret, frame = cap.read()
                if ret:
                    # Retroceder puntero porque read() avanza
                    cap.set(cv2.CAP_PROP_POS_FRAMES, indice_frame_actual) 
                    
                    # Al hacer seek en pausa, ejecutamos detección para ver qué hay
                    detecciones_yolo = []
                    if Config.USAR_YOLO:
                        detecciones_yolo = detectar(
                            frame, 
                            conf_umbral=Config.YOLO_CONF_UMBRAL, 
                            modelo_path=Config.YOLO_MODELO,
                            clases_permitidas=Config.YOLO_CLASES_PERMITIDAS
                        ) or []
                    
                    detecciones_mov = []
                    if detector_movimiento and not camara_en_movimiento:
                        detecciones_mov = detector_movimiento.actualizar(frame)
                    
                    detecciones = combinar_detecciones(detecciones_yolo, detecciones_mov)
                    detecciones = evaluador_relevancia.filtrar(detecciones, frame.shape)
                    
                    # Dibujamos detecciones como si fueran tracks (sin ID)
                    tracks_visuales = []
                    for i, d in enumerate(detecciones):
                        t = {
                            "id": -1, # ID dummy
                            "box": [d['x1'], d['y1'], d['x2'], d['y2']],
                            "clase": d.get("clase", "desc"),
                            "score": d.get("score", 0.0)
                        }
                        tracks_visuales.append(t)
                    
                    if Config.MOSTRAR_TRACKS:
                        dibujar_tracks(frame, tracks_visuales)
                    ultimos_tracks = tracks_visuales # Guardar para el loop de pausa
                else:
                    frame = np.zeros((alto, ancho, 3), dtype=np.uint8)
        
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                print("Fin del vídeo.")
                pausado = True
                continue
            
            indice_frame_actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Detección rápida de zoom basada en diff global reducido
            if Config.CAMARA_ZOOM_FAST_RATIO > 0:
                gray_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Solo ejecutar detector rápido si NO estamos ya en cooldown de zoom
                if zoom_cooldown_restante == 0:
                    gray_small = cv2.resize(
                        gray_small,
                        (max(32, int(gray_small.shape[1] * 0.1)), max(18, int(gray_small.shape[0] * 0.1))),
                        interpolation=cv2.INTER_AREA,
                    )
                    if prev_frame_zoom_small is not None:
                        diff_fast = cv2.absdiff(prev_frame_zoom_small, gray_small)
                        ratio_fast = float(np.count_nonzero(diff_fast > 25)) / float(diff_fast.size)
                        if ratio_fast >= Config.CAMARA_ZOOM_FAST_RATIO:
                            zoom_fast_detectado = True
                            zoom_cooldown_restante = Config.CAMARA_ZOOM_COOLDOWN_FRAMES
                            activar_bloqueo_movimiento("monitor:zoom_fast")
                            camara_mov_monitor = True
                            motivo_monitor_actual = "zoom_fast"
                            diagnostico_camara = None
                            ultimo_diag_monitor = None
                            prev_frame_zoom_small = gray_small
                            continue
                    prev_frame_zoom_small = gray_small
            if camara_mov_rafaga_restante > 0:
                camara_mov_rafaga_restante -= 1
                camara_mov_rafaga = True
            else:
                camara_mov_rafaga = False

            if camara_mov_anomalia_restante > 0:
                camara_mov_anomalia_restante -= 1
                camara_mov_anomalia = True
            else:
                camara_mov_anomalia = False

            if zoom_cooldown_restante > 0:
                zoom_cooldown_restante -= 1
                camara_mov_monitor = True
                motivo_monitor_actual = "zoom_hold"
                diagnostico_camara = ultimo_diag_monitor
            elif monitor_estabilidad:
                camara_mov_monitor = monitor_estabilidad.actualizar(frame)
                diagnostico_camara = monitor_estabilidad.diagnostico
                ultimo_diag_monitor = diagnostico_camara
                motivo_monitor_actual = getattr(monitor_estabilidad, "motivo", "monitor")
                if camara_mov_monitor and motivo_monitor_actual.startswith("zoom"):
                    zoom_cooldown_restante = Config.CAMARA_ZOOM_COOLDOWN_FRAMES
            else:
                camara_mov_monitor = False
                diagnostico_camara = None
                ultimo_diag_monitor = None
                motivo_monitor_actual = "sin_monitor"

            if camara_mov_monitor:
                activar_bloqueo_movimiento(f"monitor:{motivo_monitor_actual}")
            elif camara_mov_anomalia:
                diagnostico_camara = None
                activar_bloqueo_movimiento(anomalia_motivo_actual or "anomalia")
            elif camara_mov_rafaga:
                diagnostico_camara = None
                activar_bloqueo_movimiento("rafaga")
            else:
                desactivar_bloqueo_movimiento()
            
            # --- PROCESAMIENTO ---
            tracks = []
            if camara_en_movimiento:
                ultimos_tracks = []
                rafaga_consecutiva = 0
                anomalia_consecutiva = 0
            elif indice_frame_actual % args.skip == 0:
                detecciones_yolo = []
                if Config.USAR_YOLO:
                    detecciones_yolo = detectar(
                        frame, 
                        conf_umbral=Config.YOLO_CONF_UMBRAL, 
                        modelo_path=Config.YOLO_MODELO,
                        clases_permitidas=Config.YOLO_CLASES_PERMITIDAS
                    ) or []

                if predictor_movimiento:
                    zonas_predichas = predictor_movimiento.consumir_zonas()
                    detecciones_predichas = []
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

                if Config.MOVIMIENTO_ANOMALIA_POSIBLE_DRON:
                    total_pd_yolo = sum(
                        1 for d in detecciones_yolo if d.get("clase") == "posible_dron"
                    )
                    if total_pd_yolo > Config.MOVIMIENTO_ANOMALIA_POSIBLE_DRON:
                        anomalia_motivo_actual = "anomalia_posible_dron"
                        activar_bloqueo_movimiento(anomalia_motivo_actual)
                        camara_mov_anomalia_restante = max(
                            camara_mov_anomalia_restante,
                            Config.MOVIMIENTO_ANOMALIA_FRAMES_ENFRIAMIENTO,
                        )
                        camara_mov_anomalia = True
                        diagnostico_camara = None
                        anomalia_consecutiva = 0
                        continue
                
                detecciones_mov = []
                if detector_movimiento and not camara_en_movimiento:
                    detecciones_mov = detector_movimiento.actualizar(frame)
                    total_movimientos = getattr(
                        detector_movimiento, "ultimo_conteo_detecciones", 0
                    )
                    if Config.MOVIMIENTO_RAFAGA_BLOBS:
                        if total_movimientos >= Config.MOVIMIENTO_RAFAGA_BLOBS:
                            rafaga_consecutiva += 1
                        else:
                            rafaga_consecutiva = 0

                        activacion = max(1, Config.MOVIMIENTO_RAFAGA_FRAMES_ACTIVACION)
                        if rafaga_consecutiva >= activacion:
                            diagnostico_camara = None
                            activar_bloqueo_movimiento("rafaga")
                            camara_mov_rafaga_restante = max(
                                camara_mov_rafaga_restante,
                                Config.MOVIMIENTO_RAFAGA_FRAMES,
                            )
                            camara_mov_rafaga = True
                            rafaga_consecutiva = 0
                            continue
                    else:
                        rafaga_consecutiva = 0
                else:
                    rafaga_consecutiva = 0
                    total_movimientos = 0
                
                # --- NUEVA LÓGICA: FILTRO IA + DATASET ---
                # Analizamos los objetos en movimiento que YOLO no vio
                from sussy.core.utilidades_iou import calcular_iou # Import local para evitar lios
                
                nuevas_detecciones_ia = []
                
                for d_mov in detecciones_mov:
                    # 1. Chequear si ya solapa con una detección YOLO existente
                    solapa_yolo = False
                    for d_yolo in detecciones_yolo:
                        if calcular_iou(d_mov, d_yolo) > 0.1: # Un poco de margen
                            solapa_yolo = True
                            break
                    
                    if solapa_yolo:
                        # Ya lo tiene YOLO, no hacemos nada extra (quizás guardar crop etiquetado si quisiéramos)
                        continue
                        
                    # 2. Si es movimiento "huérfano", intentamos clasificarlo con IA
                    detectado_en_crop = None
                    if Config.USAR_FILTRO_IA_EN_MOVIMIENTO:
                        detectado_en_crop = analizar_recorte(
                            frame, 
                            d_mov['x1'], d_mov['y1'], d_mov['x2'], d_mov['y2'],
                            conf_umbral=Config.YOLO_CONF_UMBRAL - 0.1, # Un poco más permisivo
                            modelo_path=Config.YOLO_MODELO,
                            clases_permitidas=Config.YOLO_CLASES_PERMITIDAS,
                            padding_pct=Config.MOVIMIENTO_CROP_PADDING_PCT,
                        )
                        
                        if detectado_en_crop:
                            # ¡ÉXITO! La IA encontró algo en el recorte
                            nuevas_detecciones_ia.append(detectado_en_crop)
                            # Nota: Al añadirlo a detecciones_yolo (o lista similar), 
                            # combinar_detecciones lo priorizará sobre el bloque de movimiento.
                    
                    # 3. Guardar crop para entrenamiento
                    if Config.GUARDAR_CROPS_ENTRENAMIENTO:
                        clase_guardar = "unknown"
                        if detectado_en_crop:
                            clase_guardar = detectado_en_crop['clase']
                        
                        guardar_crop(
                            frame, 
                            d_mov['x1'], d_mov['y1'], d_mov['x2'], d_mov['y2'],
                            clase_guardar,
                            Config.RUTA_DATASET_CAPTURE,
                            padding_pct=Config.MOVIMIENTO_CROP_PADDING_PCT,
                        )

                # Añadimos lo nuevo encontrado a la lista "oficial" de YOLO para que el tracker lo pille
                detecciones_yolo.extend(nuevas_detecciones_ia)

                detecciones = combinar_detecciones(detecciones_yolo, detecciones_mov)
                detecciones = evaluador_relevancia.filtrar(detecciones, frame.shape)

                anomalia_detectada = False
                total_detecciones = len(detecciones)
                if (
                    Config.MOVIMIENTO_ANOMALIA_TOTAL
                    and total_detecciones >= Config.MOVIMIENTO_ANOMALIA_TOTAL
                ):
                    anomalia_detectada = True

                if Config.MOVIMIENTO_ANOMALIA_POSIBLE_DRON:
                    total_posible_dron = sum(
                        1 for d in detecciones if d.get("clase") == "posible_dron"
                    )
                    if total_posible_dron >= Config.MOVIMIENTO_ANOMALIA_POSIBLE_DRON:
                        anomalia_detectada = True

                if anomalia_detectada:
                    anomalia_consecutiva += 1
                else:
                    anomalia_consecutiva = 0

                activacion_anomalia = max(
                    1, Config.MOVIMIENTO_ANOMALIA_FRAMES_ACTIVACION
                )
                if anomalia_consecutiva >= activacion_anomalia:
                    anomalia_motivo_actual = "anomalia"
                    activar_bloqueo_movimiento(anomalia_motivo_actual)
                    camara_mov_anomalia_restante = max(
                        camara_mov_anomalia_restante,
                        Config.MOVIMIENTO_ANOMALIA_FRAMES_ENFRIAMIENTO,
                    )
                    camara_mov_anomalia = True
                    diagnostico_camara = None
                    anomalia_consecutiva = 0
                    continue

                if tracker:
                    tracks = tracker.actualizar(detecciones)
                    if predictor_movimiento:
                        predictor_movimiento.preparar_zonas(tracks, frame.shape)
                else:
                    # Si no hay tracker, pasamos las detecciones crudas como "tracks" para visualizar
                    for d in detecciones:
                        tracks.append({
                            "id": -1,
                            "box": [d['x1'], d['y1'], d['x2'], d['y2']],
                            "clase": d.get("clase", "desc"),
                            "score": d.get("score", 0.0)
                        })

                ultimos_tracks = tracks # Guardar para pausa
                
                if logger is not None:
                    logger.registrar(indice_frame_actual, tracks)

            procesar_eventos_alerta(
                gestor_estado,
                gestor_eventos,
                tracks,
                indice_frame_actual,
                fps,
            )
            
            # --- VISUALIZACIÓN ---
            if Config.MOSTRAR_TRACKS:
                dibujar_tracks(frame, tracks)
            
            # Info overlay
            if Config.MOSTRAR_UI:
                texto = f"Frame {indice_frame_actual}/{total_frames} - objs: {len(tracks)}"
                dibujar_texto(
                    frame,
                    texto,
                    (ancho - 350, 40),
                    color=(0, 255, 0),
                    font_scale=0.7,
                    thickness=2,
                    font_path=font_path_ui,
                )
                if camara_en_movimiento:
                    motivo_overlay = razon_movimiento or ""
                    if motivo_overlay.startswith("monitor"):
                        sub = motivo_overlay.split(":", 1)[1] if ":" in motivo_overlay else "global"
                        if sub.startswith("zoom"):
                            msg = "CÁMARA EN MOVIMIENTO (zoom detectado)"
                        else:
                            msg = "CÁMARA EN MOVIMIENTO (monitor global)"
                    elif motivo_overlay == "rafaga":
                        msg = "CÁMARA EN MOVIMIENTO (ráfaga de blobs)"
                    elif motivo_overlay == "anomalia":
                        msg = "CÁMARA EN MOVIMIENTO (detecciones masivas)"
                    elif motivo_overlay == "anomalia_posible_dron":
                        msg = "CÁMARA EN MOVIMIENTO (exceso posible_dron)"
                    else:
                        msg = "CÁMARA EN MOVIMIENTO"
                    dibujar_texto(
                        frame,
                        msg,
                        (20, 40),
                        color=(0, 0, 255),
                        font_scale=0.6,
                        thickness=2,
                        font_path=font_path_ui,
                    )
                    if motivo_overlay.startswith("monitor") and diagnostico_camara:
                        sub = motivo_overlay.split(":", 1)[1] if ":" in motivo_overlay else "global"
                        if sub.startswith("zoom"):
                            diag_msg = (
                                f"Escala x{diagnostico_camara.factor_escala:.3f} "
                                f"diff={diagnostico_camara.ratio_pixeles:.2f}"
                            )
                        else:
                            diag_msg = (
                                f"Δpx={diagnostico_camara.desplazamiento_medio:.1f} "
                                f"pts={diagnostico_camara.puntos_validos}/{diagnostico_camara.total_puntos}"
                            )
                        dibujar_texto(
                            frame,
                            diag_msg,
                            (20, 65),
                            color=(0, 0, 255),
                            font_scale=0.55,
                            thickness=1,
                            font_path=font_path_ui,
                        )

        else:
            # Pausado: mostrar frame estático
            cap.set(cv2.CAP_PROP_POS_FRAMES, indice_frame_actual) # Asegurar posición
            ret, frame = cap.read()
            if ret:
                 # Retroceder porque read avanzó
                 cap.set(cv2.CAP_PROP_POS_FRAMES, indice_frame_actual)
                 
                 # DIBUJAR LOS ÚLTIMOS TRACKS CONOCIDOS
                 if Config.MOSTRAR_TRACKS:
                     dibujar_tracks(frame, ultimos_tracks)
                 
                 # Info overlay
                 if Config.MOSTRAR_UI:
                     texto = f"Frame {indice_frame_actual}/{total_frames} - objs: {len(ultimos_tracks)} (PAUSA)"
                     dibujar_texto(
                         frame,
                         texto,
                         (ancho - 450, 40),
                         color=(0, 255, 255),
                         font_scale=0.7,
                         thickness=2,
                         font_path=font_path_ui,
                     )
                     if camara_en_movimiento:
                         motivo_overlay = razon_movimiento or ""
                         if motivo_overlay.startswith("monitor"):
                             sub = motivo_overlay.split(":", 1)[1] if ":" in motivo_overlay else "global"
                             if sub.startswith("zoom"):
                                 msg = "CÁMARA EN MOVIMIENTO (zoom detectado)"
                             else:
                                 msg = "CÁMARA EN MOVIMIENTO (monitor global)"
                         elif motivo_overlay == "rafaga":
                             msg = "CÁMARA EN MOVIMIENTO (ráfaga de blobs)"
                         elif motivo_overlay == "anomalia":
                             msg = "CÁMARA EN MOVIMIENTO (detecciones masivas)"
                         elif motivo_overlay == "anomalia_posible_dron":
                             msg = "CÁMARA EN MOVIMIENTO (exceso posible_dron)"
                         else:
                             msg = "CÁMARA EN MOVIMIENTO"
                         dibujar_texto(
                             frame,
                             msg,
                             (20, 40),
                             color=(0, 0, 255),
                             font_scale=0.6,
                             thickness=2,
                             font_path=font_path_ui,
                         )

            else:
                frame = np.zeros((alto, ancho, 3), dtype=np.uint8)

        # Dibujar UI Custom
        areas_ui = dibujar_ui(frame, pausado, indice_frame_actual, total_frames)

        cv2.imshow(ventana, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            pausado = not pausado

    cap.release()
    print(f"Vídeo terminado. Frames procesados: {indice_frame_actual}")
    if logger is not None:
        logger.cerrar()
        print(f"CSV de tracks guardado en: {logger.ruta}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

