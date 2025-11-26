import argparse
from typing import Optional, List

import cv2

from sussy.core.ingesta import frames_desde_video
from sussy.core.deteccion import detectar, combinar_detecciones
from sussy.core.movimiento import MotionDetector
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


def main() -> None:
    args = parse_args()

    print("Sistema Sussy – pipeline básico (INGESTA → DETECCIÓN → TRACKING → VISUALIZACIÓN)")
    print(f"Abriendo vídeo: {args.video}")
    print("Pulsa 'q' en la ventana de vídeo para salir.")

    tracker = TrackerSimple()
    motion_detector = MotionDetector()
    logger: Optional[RegistradorCSV] = None
    if args.log_csv:
        logger = RegistradorCSV(args.log_csv)
        print(f"Registrando tracks en: {logger.ruta}")

    num_frames_mostrados = 0
    ventana = "Sussy – Vista previa"

    for idx, frame in frames_desde_video(args.video):
        # Saltar frames según 'skip'
        if idx % args.skip != 0:
            continue

        if num_frames_mostrados == 0:
            alto, ancho, canales = frame.shape
            print(f"Resolución: {ancho}x{alto}, canales: {canales}")

        # 1) Detección con YOLO
        detecciones_yolo = detectar(frame) or []
        
        # 2) Detección de movimiento (para objetos pequeños/lejanos)
        detecciones_mov = motion_detector.actualizar(frame)

        # 3) Combinar ambas
        detecciones = combinar_detecciones(detecciones_yolo, detecciones_mov)


        # 2) Tracking (asigna IDs a las detecciones)
        tracks = tracker.actualizar(detecciones)

        # 3) Registro (si está activado)
        if logger is not None:
            logger.registrar(idx, tracks)

        # 4) Visualización de tracks
        dibujar_tracks(frame, tracks)

        # 5) Info overlay básica (arriba a la derecha)
        texto = f"Frame {idx} – objs: {len(tracks)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(
            texto, font, font_scale, thickness
        )

        margin_x = 10
        margin_y = 10

        x = frame.shape[1] - text_width - margin_x
        y = margin_y + text_height

        cv2.putText(
            frame,
            texto,
            (x, y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

        # Mostrar
        cv2.imshow(ventana, frame)
        key: int = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Salida solicitada por el usuario (q).")
            break

        num_frames_mostrados += 1

        if idx % 50 == 0:
            print(f"Procesado frame {idx} (tracks: {len(tracks)})")

    print(f"Vídeo terminado o salida manual. Frames mostrados: {num_frames_mostrados}")
    if logger is not None:
        logger.cerrar()
        print(f"CSV de tracks guardado en: {logger.ruta}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
