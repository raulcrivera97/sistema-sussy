"""
Inferencia ONNX directa con control explícito del ExecutionProvider.
Evita que Ultralytics seleccione CPUExecutionProvider por defecto.
Optimizado para máximo rendimiento con prefetching y paralelismo.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np

LOGGER = logging.getLogger("sussy.onnx_inference")


class ONNXDetector:
    """
    Detector YOLO usando ONNX Runtime directamente con DmlExecutionProvider.
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        input_size: int = 1280,
    ):
        import onnxruntime as ort
        import os

        self.input_size = input_size

        # Priorizar DML si está disponible
        available = ort.get_available_providers()
        if providers is None:
            providers = []
            if "DmlExecutionProvider" in available:
                providers.append("DmlExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

        LOGGER.info("Creando sesión ONNX con providers: %s", providers)

        # Opciones de sesión optimizadas para máximo rendimiento
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Paralelismo: usar todos los cores disponibles
        num_threads = os.cpu_count() or 4
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        
        # Modo de ejecución paralelo para operadores independientes
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        # Optimizaciones de memoria
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        sess_options.enable_cpu_mem_arena = True

        # Configuración específica para DML
        provider_options = []
        for prov in providers:
            if prov == "DmlExecutionProvider":
                # Usar GPU 1 (NVIDIA RTX) en lugar de GPU 0 (Intel integrada)
                # device_id=1 selecciona la segunda GPU del sistema
                provider_options.append({
                    "device_id": 1,
                    "disable_metacommands": False,
                    "enable_dynamic_graph_fusion": True,
                })
            else:
                provider_options.append({})

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )

        # Info del provider activo
        active_providers = self.session.get_providers()
        LOGGER.info("Providers activos en la sesión: %s", active_providers)
        self.active_provider = active_providers[0] if active_providers else "Unknown"

        # Obtener info de inputs/outputs
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Detectar el tamaño de entrada real del modelo ONNX
        # El shape suele ser [batch, channels, height, width] o similar
        model_input_size = None
        if self.input_shape and len(self.input_shape) >= 4:
            # Intentar obtener height/width del shape
            h, w = self.input_shape[2], self.input_shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                model_input_size = h  # Asumimos cuadrado
                if model_input_size != input_size:
                    LOGGER.warning(
                        "El modelo ONNX espera entrada %dx%d pero se solicitó %d. "
                        "Usando el tamaño del modelo ONNX.",
                        model_input_size, model_input_size, input_size
                    )
                    self.input_size = model_input_size
        
        LOGGER.info("ONNXDetector inicializado: input_size=%d, shape=%s", self.input_size, self.input_shape)

        # Preparar IO binding para reducir copias CPU<->GPU
        self._use_io_binding = "DmlExecutionProvider" in active_providers
        self._io_binding = None
        if self._use_io_binding:
            try:
                self._io_binding = self.session.io_binding()
                LOGGER.info("IO Binding habilitado para DML (reduce overhead de transferencia)")
            except Exception as e:
                LOGGER.warning("No se pudo habilitar IO Binding: %s", e)
                self._use_io_binding = False

        # Cargar nombres de clases COCO (YOLO usa estos por defecto)
        self.names = self._get_coco_names()

        # Pool de threads para preprocesamiento paralelo
        self._preprocess_executor = ThreadPoolExecutor(max_workers=2)
        self._pending_preprocess = None
        self._preprocess_lock = threading.Lock()

    def _get_coco_names(self) -> Dict[int, str]:
        """Nombres de clases COCO estándar."""
        return {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
            10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
            14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
            20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
            35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
            39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
            44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
            49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
            54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
            59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
            64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
            69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
            74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
            79: "toothbrush",
        }

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocesa el frame para YOLO:
        - Resize manteniendo aspect ratio con letterbox
        - Normaliza a [0, 1]
        - Convierte a NCHW
        Optimizado para máximo rendimiento.
        """
        import cv2
        
        h, w = frame.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize con interpolación rápida
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox (padding) - preallocado para eficiencia
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        
        # Usar copyMakeBorder es más rápido que crear array y copiar
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, self.input_size - new_h - pad_h,
            pad_w, self.input_size - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        # Normalizar y convertir a NCHW float32 - operaciones vectorizadas
        # Usar np.ascontiguousarray para memoria contigua (más rápido para DML)
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[np.newaxis], dtype=np.float32) / 255.0

        return blob, scale, (pad_w, pad_h)

    def postprocess(
        self,
        outputs: np.ndarray,
        scale: float,
        pad: Tuple[int, int],
        original_shape: Tuple[int, int],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Postprocesa la salida de YOLO:
        - Filtra por confianza
        - Aplica NMS
        - Convierte coordenadas a la imagen original
        Optimizado con operaciones vectorizadas.
        """
        # YOLO11 output shape: [1, num_classes+4, num_boxes] o [1, num_boxes, num_classes+4]
        if len(outputs.shape) == 3:
            if outputs.shape[1] < outputs.shape[2]:
                outputs = outputs.transpose(0, 2, 1)
            outputs = outputs[0]

        # Filtrar por confianza primero (reduce trabajo posterior)
        scores = outputs[:, 4:]
        max_scores = scores.max(axis=1)
        mask = max_scores > conf_threshold
        
        if not mask.any():
            return []
        
        outputs = outputs[mask]
        scores = scores[mask]
        
        # Obtener clase y score máximo
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Convertir xywh a xyxy (vectorizado)
        boxes = outputs[:, :4]
        half_wh = boxes[:, 2:4] / 2
        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, :2] = boxes[:, :2] - half_wh  # x1, y1
        boxes_xyxy[:, 2:] = boxes[:, :2] + half_wh  # x2, y2

        # NMS
        indices = self._nms(boxes_xyxy, confidences, iou_threshold)
        
        if len(indices) == 0:
            return []
        
        boxes_xyxy = boxes_xyxy[indices]
        class_ids = class_ids[indices]
        confidences = confidences[indices]

        # Convertir coordenadas a imagen original (vectorizado)
        pad_w, pad_h = pad
        orig_h, orig_w = original_shape
        
        # Quitar padding y escalar
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale
        
        # Clamp a límites
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
        
        # Convertir a enteros
        boxes_int = boxes_xyxy.astype(np.int32)

        # Construir lista de detecciones
        detections = [
            {
                "x1": int(boxes_int[i, 0]),
                "y1": int(boxes_int[i, 1]),
                "x2": int(boxes_int[i, 2]),
                "y2": int(boxes_int[i, 3]),
                "clase": self.names.get(int(class_ids[i]), str(class_ids[i])),
                "score": float(confidences[i]),
            }
            for i in range(len(indices))
        ]

        return detections

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> List[int]:
        """Non-Maximum Suppression simple."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def _run_inference(self, blob: np.ndarray) -> np.ndarray:
        """Ejecuta inferencia en la GPU."""
        if self._use_io_binding and self._io_binding is not None:
            try:
                self._io_binding.bind_cpu_input(self.input_name, blob)
                for out_name in self.output_names:
                    self._io_binding.bind_output(out_name, "cpu")
                self.session.run_with_iobinding(self._io_binding)
                outputs = self._io_binding.copy_outputs_to_cpu()
                self._io_binding.clear_binding_inputs()
                self._io_binding.clear_binding_outputs()
                return outputs[0]
            except Exception:
                pass
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return outputs[0]

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        classes_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta detección en un frame.
        Usa IO Binding cuando está disponible para máximo rendimiento GPU.
        """
        original_shape = frame.shape[:2]  # (h, w)

        # Preprocess
        blob, scale, pad = self.preprocess(frame)

        # Inference
        output = self._run_inference(blob)

        # Postprocess
        detections = self.postprocess(
            output,
            scale,
            pad,
            original_shape,
            conf_threshold,
            iou_threshold,
        )

        # Filtrar por clases si se especifica
        if classes_filter:
            detections = [d for d in detections if d["clase"] in classes_filter]

        return detections

    def preprocess_async(self, frame: np.ndarray) -> None:
        """Inicia preprocesamiento en background para el siguiente frame."""
        self._pending_preprocess = self._preprocess_executor.submit(
            self._preprocess_with_shape, frame
        )

    def _preprocess_with_shape(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
        """Preprocesa y devuelve también el shape original."""
        blob, scale, pad = self.preprocess(frame)
        return blob, scale, pad, frame.shape[:2]

    def detect_with_prefetch(
        self,
        frame: np.ndarray,
        next_frame: Optional[np.ndarray],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        classes_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detecta en frame actual mientras preprocesa el siguiente en paralelo.
        Llama con next_frame=None en el último frame.
        """
        # Si hay preprocesamiento pendiente del frame anterior, usarlo
        if self._pending_preprocess is not None:
            try:
                blob, scale, pad, original_shape = self._pending_preprocess.result()
            except Exception:
                blob, scale, pad = self.preprocess(frame)
                original_shape = frame.shape[:2]
            self._pending_preprocess = None
        else:
            blob, scale, pad = self.preprocess(frame)
            original_shape = frame.shape[:2]

        # Iniciar preprocesamiento del siguiente frame en paralelo con inferencia
        if next_frame is not None:
            self.preprocess_async(next_frame)

        # Inference
        output = self._run_inference(blob)

        # Postprocess
        detections = self.postprocess(
            output,
            scale,
            pad,
            original_shape,
            conf_threshold,
            iou_threshold,
        )

        if classes_filter:
            detections = [d for d in detections if d["clase"] in classes_filter]

        return detections

    def warmup(self) -> None:
        """Ejecuta una inferencia dummy para calentar el modelo."""
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.detect(dummy, conf_threshold=0.01)
        LOGGER.debug("Warmup ONNX completado con %s", self.active_provider)


# Instancia global para reutilizar
_ONNX_DETECTOR: Optional[ONNXDetector] = None


def get_onnx_detector(
    model_path: str,
    input_size: int = 1280,
    providers: Optional[List[str]] = None,
) -> ONNXDetector:
    """
    Obtiene o crea el detector ONNX global.
    """
    global _ONNX_DETECTOR
    if _ONNX_DETECTOR is None:
        _ONNX_DETECTOR = ONNXDetector(model_path, providers=providers, input_size=input_size)
        _ONNX_DETECTOR.warmup()
    return _ONNX_DETECTOR

