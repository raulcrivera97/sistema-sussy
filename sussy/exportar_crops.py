import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from sussy.core.ingesta import frames_desde_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exporta recortes (crops) de objetos a partir de un vídeo y un CSV de tracks.\n"
            "Se puede filtrar por clases (por ejemplo: --clases bird,airplane,car)."
        )
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Ruta al vídeo original (el mismo que se usó para generar el CSV).",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Ruta al CSV de tracks generado por Sussy (--log-csv).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Carpeta de salida donde se guardarán las imágenes recortadas.",
    )
    parser.add_argument(
        "--cada",
        type=int,
        default=3,
        help="Guardar solo 1 de cada N frames (por defecto 3) para no generar miles de casi iguales.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=16,
        help="Tamaño mínimo (en píxeles) del lado más corto del recorte. Recortes más pequeños se descartan.",
    )
    parser.add_argument(
        "--clases",
        type=str,
        default=None,
        help=(
            "Lista de clases a incluir, separadas por comas. "
            "Ejemplo: --clases bird,airplane,car. "
            "Si no se indica, se exportan todas las clases."
        ),
    )
    return parser.parse_args()


def cargar_tracks_por_frame(
    csv_path: Path, clases_filtradas: Optional[set[str]]
) -> Dict[int, List[dict]]:
    """
    Carga el CSV de tracks y agrupa las filas por frame.

    Si 'clases_filtradas' no es None, solo se guardan las filas cuya 'clase'
    esté en ese conjunto. Si es None, se guardan todas.
    """
    tracks_por_frame: Dict[int, List[dict]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame"])
            clase = row["clase"]

            if clases_filtradas is not None and clase not in clases_filtradas:
                continue

            track_id = int(row["id"])
            score = float(row["score"])
            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])

            tracks_por_frame.setdefault(frame_idx, []).append(
                {
                    "id": track_id,
                    "clase": clase,
                    "score": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    return tracks_por_frame


def asegurar_out_dirs(base: Path) -> Path:
    """
    Crea una subcarpeta 'sin_clasificar' dentro de out-dir
    y la devuelve. Ahí irán todos los recortes.
    """
    base.mkdir(parents=True, exist_ok=True)
    sin_clasificar = base / "sin_clasificar"
    sin_clasificar.mkdir(parents=True, exist_ok=True)
    return sin_clasificar


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.csv)
    out_base = Path(args.out_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo no encontrado: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")

    clases_filtradas: Optional[set[str]] = None
    if args.clases:
        clases_filtradas = {
            c.strip() for c in args.clases.split(",") if c.strip()
        }
        print(f"Filtrando solo clases: {sorted(clases_filtradas)}")
    else:
        print("Sin filtro de clases: se exportarán todas las clases presentes en el CSV.")

    tracks_por_frame = cargar_tracks_por_frame(csv_path, clases_filtradas)
    out_dir = asegurar_out_dirs(out_base)

    print(f"Frames con tracks relevantes: {len(tracks_por_frame)}")
    print(f"Guardando recortes en: {out_dir}")

    num_crops = 0

    cada = max(1, args.cada)
    min_size = max(1, args.min_size)

    for frame_idx, frame in frames_desde_video(str(video_path)):
        # Solo guardar 1 de cada N frames globalmente
        if frame_idx % cada != 0:
            continue

        if frame_idx not in tracks_por_frame:
            continue

        alto, ancho, _ = frame.shape

        for t in tracks_por_frame[frame_idx]:
            x1 = max(0, min(ancho - 1, t["x1"]))
            y1 = max(0, min(alto - 1, t["y1"]))
            x2 = max(0, min(ancho, t["x2"]))
            y2 = max(0, min(alto, t["y2"]))

            w = x2 - x1
            h = y2 - y1

            if w < min_size or h < min_size:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Nombre de archivo: frame_id_clase_score_id.jpg
            nombre = f"f{frame_idx:06d}_id{t['id']}_cls{t['clase']}_s{t['score']:.2f}.jpg"
            out_path = out_dir / nombre

            cv2.imwrite(str(out_path), crop)
            num_crops += 1

    print(f"Total de recortes guardados: {num_crops}")


if __name__ == "__main__":
    main()
