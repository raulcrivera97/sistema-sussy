from typing import List, Dict, Any, Optional, Union

import logging
import os
import numpy as np
from ultralytics import YOLO

from sussy.core.backends import BackendInfo, detectar_backends, describir_backends
from sussy.core.onnx_inference import ONNXDetector

# Modelo global (se carga solo una vez por configuración)
_MODEL: Optional[Union[YOLO, ONNXDetector]] = None
_WARMED_UP = False
_BACKEND_INFO: Optional[BackendInfo] = None
_CURRENT_MODEL_PATH: Optional[str] = None  # Para detectar cambios de modelo
_CURRENT_IMG_SIZE: Optional[int] = None    # Para detectar cambios de tamaño

# Tamaño de entrada para YOLO: más grande = mejor para objetos pequeños (a costa de CPU)
_IMG_SIZE = 1280

# Tipo de detección que usamos en todo el sistema
Detection = Dict[str, Any]

LOGGER = logging.getLogger("sussy.deteccion")


def invalidar_modelo():
    """
    Invalida el modelo cargado para forzar recarga en la siguiente inferencia.
    Llamar cuando cambia el preset de rendimiento o el modelo YOLO.
    """
    global _MODEL, _WARMED_UP, _BACKEND_INFO, _CURRENT_MODEL_PATH, _CURRENT_IMG_SIZE
    
    if _MODEL is not None:
        LOGGER.info("Invalidando modelo cargado para permitir recarga")
        try:
            if hasattr(_MODEL, 'session'):
                del _MODEL
            elif hasattr(_MODEL, 'model'):
                del _MODEL
        except Exception:
            pass
    
    _MODEL = None
    _WARMED_UP = False
    _BACKEND_INFO = None
    _CURRENT_MODEL_PATH = None
    _CURRENT_IMG_SIZE = None


def _cargar_modelo(ruta_pesos: str = "yolo11n.pt"):
    """
    Carga el modelo YOLO usando el mejor backend disponible según las preferencias.
    Recarga automáticamente si cambia el modelo o el tamaño de imagen.
    """
    global _MODEL, _WARMED_UP, _BACKEND_INFO, _CURRENT_MODEL_PATH, _CURRENT_IMG_SIZE
    from sussy.config import Config
    
    current_modelo = getattr(Config, "YOLO_MODELO", ruta_pesos)
    current_imgsz = getattr(Config, "YOLO_IMG_SIZE", _IMG_SIZE)
    
    necesita_recarga = False
    if _MODEL is not None:
        if _CURRENT_MODEL_PATH != current_modelo:
            LOGGER.info("Modelo cambió de %s a %s - recargando", _CURRENT_MODEL_PATH, current_modelo)
            necesita_recarga = True
        elif _CURRENT_IMG_SIZE != current_imgsz:
            LOGGER.info("Tamaño de imagen cambió de %s a %s - recargando", _CURRENT_IMG_SIZE, current_imgsz)
            necesita_recarga = True
    
    if necesita_recarga:
        invalidar_modelo()

    if _MODEL is None:
        preferencias = getattr(Config, "BACKENDS_PREFERIDOS", ["cuda", "onnx", "cpu"])
        backend_forzado = getattr(Config, "BACKEND_FORZADO", None)
        if backend_forzado:
            preferencias = [backend_forzado] + [p for p in preferencias if p != backend_forzado]

        backend_preferido, backends_disponibles = detectar_backends(
            preferencias=preferencias,
            modelo_pt=ruta_pesos,
            modelo_onnx_cfg=getattr(Config, "YOLO_MODELO_ONNX", None),
        )

        if backend_preferido is None:
            msg = f"No hay backend de inferencia disponible. Revisar dependencias. Detectados: {describir_backends(backends_disponibles)}"
            LOGGER.error(msg)
            raise RuntimeError(msg)

        # Ordenar candidatos según preferencias para permitir fallback
        ordenados: List[BackendInfo] = []
        for pref in preferencias:
            ordenados.extend([c for c in backends_disponibles if c.nombre == pref and c not in ordenados])
        ordenados.extend([c for c in backends_disponibles if c not in ordenados])

        errores = []

        for backend in ordenados:
            try:
                LOGGER.info(
                    "Intentando backend: %s (%s). Otros disponibles: %s",
                    backend.nombre,
                    backend.descripcion,
                    describir_backends([c for c in ordenados if c != backend]),
                )

                imgsz = getattr(Config, "YOLO_IMG_SIZE", _IMG_SIZE)

                if backend.tipo == "onnx":
                    # Usar nuestro detector ONNX directo para forzar DML
                    modelo = ONNXDetector(
                        model_path=backend.modelo_path,
                        input_size=imgsz,
                        providers=None,  # Auto-detecta DML/CUDA/CPU
                    )
                    LOGGER.info("ONNX cargado con provider: %s", modelo.active_provider)
                else:
                    # Backend Torch (CUDA/MPS/CPU)
                    modelo = YOLO(backend.modelo_path)

                    if backend.tipo == "torch":
                        import torch  # type: ignore

                        modelo.to(backend.dispositivo)

                        if getattr(Config, "YOLO_HALF", False) and str(backend.dispositivo).startswith("cuda"):
                            try:
                                modelo.model.half()  # type: ignore[attr-defined]
                                LOGGER.info("Modelo en FP16")
                            except Exception as exc:  # pragma: no cover
                                LOGGER.warning("No se pudo activar FP16: %s", exc)

                        try:
                            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                        except Exception:
                            pass

                _MODEL = modelo
                _BACKEND_INFO = backend
                _CURRENT_MODEL_PATH = current_modelo
                _CURRENT_IMG_SIZE = current_imgsz

                if backend.tipo == "onnx":
                    # ONNXDetector ya hace warmup internamente
                    _WARMED_UP = True
                else:
                    _ejecutar_warmup(_MODEL, strict=True)
                    _WARMED_UP = True
                
                LOGGER.info("Modelo cargado: %s (imgsz=%s) con backend %s", 
                           current_modelo, current_imgsz, backend.nombre)
                break
            except Exception as exc:  # pragma: no cover - queremos seguir probando
                errores.append(f"{backend.nombre}: {exc}")
                LOGGER.warning("Fallo al inicializar backend %s -> %s. Probando siguiente.", backend.nombre, exc)
                _MODEL = None
                _BACKEND_INFO = None
                _WARMED_UP = False

        if _MODEL is None or _BACKEND_INFO is None:
            msg = f"No se pudo inicializar ningún backend. Errores: {errores}"
            LOGGER.error(msg)
            raise RuntimeError(msg)

    return _MODEL


def precargar_modelo(ruta_pesos: str = "yolo11n.pt") -> None:
    """Precarga el modelo YOLO (descarga y warmup)."""
    _cargar_modelo(ruta_pesos)


def exportar_modelo_onnx(
    ruta_pesos: str = "yolo11n.pt",
    imgsz: Optional[int] = None,
    ruta_salida: Optional[str] = None,
) -> str:
    """
    Exporta el modelo YOLO a ONNX para habilitar runtimes alternativos (ORT/DirectML).
    - imgsz: si es None, usa Config.YOLO_IMG_SIZE.
    - ruta_salida: opcional para mover el archivo resultante.
    Retorna la ruta final del archivo .onnx generado.
    """
    from sussy.config import Config

    imgsz = imgsz or getattr(Config, "YOLO_IMG_SIZE", _IMG_SIZE)

    LOGGER.info("Exportando modelo %s a ONNX (imgsz=%s)...", ruta_pesos, imgsz)
    modelo = YOLO(ruta_pesos)
    salida_generada = modelo.export(format="onnx", imgsz=imgsz, half=getattr(Config, "YOLO_HALF", False))

    if ruta_salida and salida_generada and salida_generada != ruta_salida:
        try:
            os.replace(salida_generada, ruta_salida)
            salida_generada = ruta_salida
        except Exception as exc:
            LOGGER.warning("No se pudo mover el ONNX a %s: %s", ruta_salida, exc)

    LOGGER.info("ONNX generado en: %s", salida_generada)
    return salida_generada


def backend_activo() -> Optional[BackendInfo]:
    """Devuelve la información del backend actualmente en uso."""
    return _BACKEND_INFO


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Utilidades de detección Sussy (YOLO).")
    parser.add_argument("--export-onnx", action="store_true", help="Exporta el modelo a ONNX y termina.")
    parser.add_argument("--pesos", type=str, default="yolo11n.pt", help="Ruta al modelo .pt")
    parser.add_argument("--imgsz", type=int, default=None, help="Resolución cuadrada para export (ej. 640/1280)")
    parser.add_argument("--salida", type=str, default=None, help="Ruta destino para el .onnx (opcional)")

    args = parser.parse_args()

    if args.export_onnx:
        ruta = exportar_modelo_onnx(args.pesos, imgsz=args.imgsz, ruta_salida=args.salida)
        print(f"ONNX exportado en: {ruta}")
        sys.exit(0)

    print("Nada que hacer. Usa --export-onnx para exportar el modelo.")


def _ejecutar_warmup(modelo: YOLO, strict: bool = False) -> None:
    """
    Realiza una inferencia sobre un frame vacío para inicializar kernels.
    Cualquier error en esta fase se informa; si strict=True se relanza para permitir fallback.
    """
    try:
        from sussy.config import Config

        imgsz = getattr(Config, "YOLO_IMG_SIZE", _IMG_SIZE)
        frame_dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        modelo.predict(
            frame_dummy,
            imgsz=imgsz,
            conf=0.01,
            iou=0.25,
            verbose=False,
        )
        LOGGER.debug("Warmup YOLO completado.")
    except Exception as exc:  # pragma: no cover - solo informativo
        LOGGER.warning("No se pudo ejecutar el warmup de YOLO: %s", exc)
        if strict:
            raise


def detectar(frame: np.ndarray, conf_umbral: float = 0.5, modelo_path: str = "yolo11n.pt", clases_permitidas: List[str] = None) -> List[Detection]:
    """
    Detección "cruda" con YOLO:
    - Sin filtros por tamaño.
    - Solo conf mínima e IoU por defecto.
    - Filtrado opcional por lista de nombres de clases (clases_permitidas).

    Devolvemos SIEMPRE una lista de dicts con:
      x1, y1, x2, y2, clase (string), score (float)
    """
    from sussy.config import Config

    model = _cargar_modelo(modelo_path)

    # Si es ONNXDetector, usar su método detect() directamente
    if isinstance(model, ONNXDetector):
        return model.detect(
            frame,
            conf_threshold=conf_umbral,
            iou_threshold=0.50,
            classes_filter=clases_permitidas,
        )

    # Backend Torch (Ultralytics YOLO)
    imgsz = getattr(Config, "YOLO_IMG_SIZE", _IMG_SIZE)
    max_det = getattr(Config, "YOLO_MAX_DET", 300)
    vid_stride = getattr(Config, "YOLO_VID_STRIDE", 1)

    # Llamada a YOLO
    results = model(
        frame,
        imgsz=imgsz,
        conf=conf_umbral,
        iou=0.50,
        verbose=False,
        max_det=max_det,
        vid_stride=vid_stride,
    )

    detecciones: List[Detection] = []

    if not results:
        return detecciones

    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return detecciones

    names = model.names  # diccionario id -> nombre de clase

    for box in boxes:
        cls_id = int(box.cls.item())
        score = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        clase = names.get(cls_id, str(cls_id))

        # Filtrado por clase
        if clases_permitidas and len(clases_permitidas) > 0:
            if clase not in clases_permitidas:
                continue

        det: Detection = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "clase": clase,
            "score": score,
        }
        detecciones.append(det)

    return detecciones




def combinar_detecciones(
    dets_yolo: List[Detection],
    dets_mov: List[Detection],
    iou_thresh: float = 0.3
) -> List[Detection]:
    """
    Combina detecciones de YOLO y Movimiento.
    - Prioridad a YOLO: si un objeto de movimiento solapa con uno de YOLO,
      nos quedamos con el de YOLO (que tiene clase específica).
    - Si Movimiento no solapa con nada de YOLO, lo añadimos como posible objeto de interés.
    """
    # Importar aquí para evitar ciclos o simplemente usar la utilidad
    from sussy.core.utilidades_iou import calcular_iou
    
    finales = list(dets_yolo)  # Copia superficial

    for d_mov in dets_mov:
        solapa = False
        for d_yolo in dets_yolo:
            iou = calcular_iou(d_mov, d_yolo)
            if iou > iou_thresh:
                solapa = True
                break
        
        if not solapa:
            # Es un objeto en movimiento que YOLO no ha visto
            finales.append(d_mov)
            
    return finales


def analizar_recorte(
    frame: np.ndarray, 
    x1: int, y1: int, x2: int, y2: int, 
    conf_umbral: float = 0.3, # Umbral más bajo para segunda pasada
    modelo_path: str = "yolo11n.pt",
    clases_permitidas: List[str] = None,
    padding_pct: float = 0.0,
) -> Optional[Detection]:
    """
    Recorta la región indicada y pasa YOLO solo a ese trozo.
    Útil para verificar qué es un objeto en movimiento pequeño.
    Retorna la mejor detección encontrada en el recorte (ajustada a coordenadas globales), o None.
    """
    # Validar coordenadas
    alto, ancho = frame.shape[:2]

    if padding_pct > 0.0:
        width = x2 - x1
        height = y2 - y1
        pad_x = int(max(2, round(width * padding_pct)))
        pad_y = int(max(2, round(height * padding_pct)))
        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_y
        y2 += pad_y

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(ancho, x2), min(alto, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Detectar en el crop
    # NOTA: Usamos un umbral un poco más bajo por defecto porque se supone que "algo se mueve"
    # y queremos saber qué es.
    dets_crop = detectar(crop, conf_umbral=conf_umbral, modelo_path=modelo_path, clases_permitidas=clases_permitidas)
    
    if not dets_crop:
        return None
    
    # Nos quedamos con la detección con mayor score
    mejor_det = max(dets_crop, key=lambda d: d['score'])
    
    # Ajustar coordenadas del crop a globales
    mejor_det['x1'] += x1
    mejor_det['x2'] += x1
    mejor_det['y1'] += y1
    mejor_det['y2'] += y1
    
    return mejor_det

