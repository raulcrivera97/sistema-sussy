import argparse
from typing import Optional, List

import cv2
import numpy as np

from sussy.core.deteccion import detectar, combinar_detecciones
from sussy.core.movimiento import DetectorMovimiento
from sussy.core.seguimiento import TrackerSimple
from sussy.core.visualizacion import dibujar_tracks
from sussy.core.registro import RegistradorCSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema Sussy – Vista previa con pipeline básico (detección + tracking)."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Ruta al archivo de vídeo de entrada (por ejemplo, C:/videos/test.mp4)",
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


def main() -> None:
    args = parse_args()

    print("Sistema Sussy – pipeline básico (INGESTA → DETECCIÓN → TRACKING → VISUALIZACIÓN)")
    print(f"Abriendo vídeo: {args.video}")
    print("Controles: [Espacio] Pausa/Play, [Click Botón] Pausa/Play, [Timeline] Buscar")
    print("Pulsa 'q' para salir.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error al abrir vídeo: {args.video}")
        return

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"Resolución: {ancho}x{alto}, Frames: {total_frames}, FPS: {fps:.2f}")

    tracker = TrackerSimple()
    detector_movimiento = DetectorMovimiento()
    logger: Optional[RegistradorCSV] = None
    if args.log_csv:
        logger = RegistradorCSV(args.log_csv)
        print(f"Registrando tracks en: {logger.ruta}")

    ventana = "Sussy - Vista previa"
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

    cv2.setMouseCallback(ventana, callback_raton)

    while True:
        # Gestionar Seek (Búsqueda)
        if peticion_busqueda >= 0:
            indice_frame_actual = peticion_busqueda
            cap.set(cv2.CAP_PROP_POS_FRAMES, indice_frame_actual)
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
                    # (aunque no actualizamos tracker para no ensuciar historia, o sí?
                    #  Mejor solo detección visual)
                    detecciones_yolo = detectar(frame) or []
                    detecciones_mov = detector_movimiento.actualizar(frame)
                    detecciones = combinar_detecciones(detecciones_yolo, detecciones_mov)
                    
                    # Dibujamos detecciones como si fueran tracks (sin ID)
                    # Para que se vean "verdes" y consistentes
                    # Creamos tracks falsos para visualización
                    tracks_visuales = []
                    for i, d in enumerate(detecciones):
                        # Estructura compatible con dibujar_tracks
                        t = {
                            "id": -1, # ID dummy
                            "box": [d['x1'], d['y1'], d['x2'], d['y2']],
                            "clase": d.get("clase", "desc"),
                            "score": d.get("score", 0.0)
                        }
                        tracks_visuales.append(t)
                    
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
                detecciones_yolo = detectar(frame) or []
                detecciones_mov = detector_movimiento.actualizar(frame)
                detecciones = combinar_detecciones(detecciones_yolo, detecciones_mov)
                tracks = tracker.actualizar(detecciones)
                ultimos_tracks = tracks # Guardar para pausa
                
                if logger is not None:
                    logger.registrar(indice_frame_actual, tracks)
            else:
                tracks = []
            
            # --- VISUALIZACIÓN ---
            dibujar_tracks(frame, tracks)
            
            # Info overlay
            texto = f"Frame {indice_frame_actual}/{total_frames} – objs: {len(tracks)}"
            cv2.putText(frame, texto, (ancho - 350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Pausado: mostrar frame estático
            # Para evitar parpadeos y recálculos, idealmente mostraríamos el último frame procesado.
            # Pero como 'frame' se sobrescribe en cada iteración del while, necesitamos recuperarlo.
            # Estrategia: Leer el frame actual (sin avanzar) y dibujar 'ultimos_tracks'.
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, indice_frame_actual) # Asegurar posición
            ret, frame = cap.read()
            if ret:
                 # Retroceder porque read avanzó
                 cap.set(cv2.CAP_PROP_POS_FRAMES, indice_frame_actual)
                 
                 # DIBUJAR LOS ÚLTIMOS TRACKS CONOCIDOS
                 # Esto mantiene los "cuadraditos verdes"
                 dibujar_tracks(frame, ultimos_tracks)
                 
                 # Info overlay
                 texto = f"Frame {indice_frame_actual}/{total_frames} – objs: {len(ultimos_tracks)} (PAUSA)"
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
