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
from sussy.core.relevancia import EvaluadorRelevancia
from sussy.core.seguimiento import TrackerSimple
from sussy.core.prediccion import PredictorMovimiento
from sussy.core.visualizacion import dibujar_tracks
from sussy.core.registro import RegistradorCSV
from sussy.core.dataset_utils import guardar_crop


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
    if Config.USAR_TRACKER:
        tracker = TrackerSimple(
            max_dist=Config.TRACKER_MATCH_DIST,
            max_frames_lost=Config.TRACKER_MAX_FRAMES_LOST,
            iou_threshold=Config.TRACKER_IOU_THRESHOLD
        )
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
            min_desplazamiento=Config.MOVIMIENTO_MIN_DESPLAZAMIENTO
        )
        print(f"Detector Movimiento: ACTIVADO (Umbral={Config.MOVIMIENTO_UMBRAL}, AreaMin={Config.MOVIMIENTO_AREA_MIN})")
    else:
        print("Detector Movimiento: DESACTIVADO")

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
            if tracker:
                tracker = TrackerSimple() # Reiniciar tracker al saltar
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
                    if detector_movimiento:
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
            
            # --- PROCESAMIENTO ---
            if indice_frame_actual % args.skip == 0:
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
                
                detecciones_mov = []
                if detector_movimiento:
                    detecciones_mov = detector_movimiento.actualizar(frame)
                
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
                
                tracks = []
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
            else:
                tracks = []

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
                cv2.putText(frame, texto, (ancho - 350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
                     cv2.putText(frame, texto, (ancho - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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

