class Config:
    """
    Archivo de configuración centralizado para el Sistema Sussy.
    Cada bloque agrupa parámetros pensados para poder activar o desactivar
    módulos completos según el dispositivo o escenario de despliegue.
    """

    # ==========================================
    # ENTORNO Y TELEMETRÍA
    # ==========================================
    DESACTIVAR_TELEMETRIA = True  # Evita envíos a servicios externos por defecto
    NIVEL_LOG = "INFO"            # INFO/DEBUG/WARNING
    TELEMETRIA_VARS = {
        "WANDB_MODE": "disabled",
        "COMET_MODE": "disabled",
        "CLEARML_MODE": "disabled",
        "YOLO_VERBOSE": "False",
    }

    # ==========================================
    # MÓDULOS GENERALES (True = Activado)
    # ==========================================
    USAR_DETECTOR_MOVIMIENTO = True
    USAR_YOLO = True
    USAR_TRACKER = True

    # ==========================================
    # PRESETS DE CÁMARA
    # ==========================================
    CAMARA_PRESET_POR_DEFECTO = None  # Ej: "fija", "orientable", "movil", "movil_plus"
    CAMARA_PRESET_OVERRIDES = {}      # Dict opcional para forzar ajustes aun usando preset

    # ==========================================
    # PRESETS DE RENDIMIENTO
    # ==========================================
    RENDIMIENTO_PRESET_POR_DEFECTO = "minimo"  # Ej: "minimo", "equilibrado", "maximo"
    RENDIMIENTO_PRESET_OVERRIDES = {}      # Ajustes manuales sobre el preset elegido
    SKIP_FRAMES_DEFECTO = 3                # N de frames a saltar por defecto (se puede forzar por preset)

    # ==========================================
    # FUENTES DE VÍDEO / INGESTA
    # ==========================================
    FUENTE_POR_DEFECTO = None  # Ruta, RTSP, HTTP o índice de webcam. Se puede sobreescribir con --source
    FUENTE_MAX_REINTENTOS = 3  # Reintentos antes de abortar una fuente inestable
    FUENTE_REINTENTO_DELAY = 2.0  # Segundos entre reintentos

    # ==========================================
    # VISUALIZACIÓN / UI
    # ==========================================
    MOSTRAR_UI = True        # Botones y timeline
    MOSTRAR_TRACKS = True    # Cajas y etiquetas en pantalla

    # ==========================================
    # DETECTOR DE MOVIMIENTO
    # ==========================================
    MOVIMIENTO_UMBRAL = 25
    MOVIMIENTO_AREA_MIN = 10
    MOVIMIENTO_CONTRASTE_MIN = 5.0
    MOVIMIENTO_MIN_FRAMES = 3
    MOVIMIENTO_MIN_DESPLAZAMIENTO = 5
    MOVIMIENTO_CROP_PADDING_PCT = 0.3  # Aumenta recortes de movimiento para contexto adicional
    MOVIMIENTO_MAX_DETECCIONES = 120    # Límite de blobs por frame para evitar saturación
    MOVIMIENTO_RAFAGA_BLOBS = 80        # Si se supera, asumimos sacudida (auto pausa)
    MOVIMIENTO_RAFAGA_FRAMES = 2        # Frames de enfriamiento tras detectar ráfaga
    MOVIMIENTO_RAFAGA_FRAMES_ACTIVACION = 2  # Frames consecutivos con ráfaga antes de pausar
    MOVIMIENTO_ANOMALIA_TOTAL = 160     # Si hay más detecciones totales, se considera anomalía
    MOVIMIENTO_ANOMALIA_POSIBLE_DRON = 8   # Nº máximo de posible_dron simultáneos
    MOVIMIENTO_ANOMALIA_FRAMES_ACTIVACION = 1
    MOVIMIENTO_ANOMALIA_FRAMES_ENFRIAMIENTO = 3

    # ==========================================
    # ESTABILIDAD DE LA CÁMARA
    # ==========================================
    USAR_MONITOR_ESTABILIDAD = True
    CAMARA_ESCALA_ANALISIS = 0.5
    CAMARA_MAX_DESPLAZAMIENTO_PX = 7.0
    CAMARA_MIN_PUNTOS = 60
    CAMARA_MAX_RATIO_PERDIDOS = 0.55
    CAMARA_FRAMES_INESTABLES = 3
    CAMARA_FRAMES_ESTABLES = 4
    CAMARA_ACTIVACION_INMEDIATA = True
    CAMARA_MAX_CAMBIO_ESCALA = 0.04
    CAMARA_MAX_RATIO_DIFERENCIA = 0.35
    CAMARA_ZOOM_COOLDOWN_FRAMES = 12
    CAMARA_ZOOM_FAST_RATIO = 0.25

    # ==========================================
    # UI / TEXTO
    # ==========================================
    UI_FONT_PATH = None  # Ruta opcional a un .ttf; si es None se auto-detecta (ej. Arial en Windows)

    # ==========================================
    # DETECCIÓN (YOLO / IA)
    # ==========================================
    BACKENDS_PREFERIDOS = ["onnx", "cuda", "cpu"]  # ONNX primero para RTX 5080 sin binarios CUDA
    BACKEND_FORZADO = "onnx"                       # Fuerza ONNX Runtime (evita PyTorch CUDA sm_120)
    YOLO_MODELO_ONNX = None                        # Ruta opcional a .onnx (fallback ONNX Runtime)
    YOLO_MODELO = "yolo11x.pt"
    YOLO_IMG_SIZE = 960           # Tamaño de entrada para YOLO (px lado mayor) - 960 es buen balance velocidad/precisión
    YOLO_DEVICE = None            # "cuda", "cpu", "mps", "auto"/None
    YOLO_HALF = False             # FP16 si la GPU lo soporta
    YOLO_MAX_DET = 300            # Límite de detecciones por frame
    YOLO_VID_STRIDE = 1           # stride para vídeo; >1 baja coste
    YOLO_CONF_UMBRAL = 0.35
    YOLO_CLASES_PERMITIDAS = [
        "person",
        "bicycle", "car", "motorcycle", "bus", "truck",
        "airplane", "bird", "drone"
    ]
    CLASES_ALERTA = [
        "drone",
        "bird",
        "airplane",
        "posible_dron",
    ]  # Clases que gatillan estados de interés en el orquestador

    # ==========================================
    # TRACKER
    # ==========================================
    TRACKER_MAX_FRAMES_LOST = 10
    TRACKER_IOU_THRESHOLD = 0.3
    TRACKER_MATCH_DIST = 100

    # ==========================================
    # RELEVANCIA DE OBJETOS EN MOVIMIENTO
    # ==========================================
    RELEVANCIA_AREA_DRON_MAX = 0.002    # Proporción del frame; por debajo se considera minúsculo (posible dron)
    RELEVANCIA_AREA_RAMA_MIN = 0.02     # Por encima suelen ser ramas/paneles que ocupan demasiado
    RELEVANCIA_VEL_MIN = 1.2            # Pixels por frame para considerar que realmente se desplaza
    RELEVANCIA_BORDE_PCT = 0.08         # Porcentaje de margen: movimiento pegado al borde suele ser vegetación
    RELEVANCIA_ASPECTO_RAMAS = 6.0      # Relaciones de aspecto muy alargadas suelen corresponder a ramas/paneles

    # ==========================================
    # PREDICCIÓN DE MOVIMIENTO / ZONAS DE INTERÉS
    # ==========================================
    USAR_PREDICCION_MOVIMIENTO = True
    PREDICCION_FRAMES_ADELANTE = 3       # Cuántos frames proyectamos hacia delante
    PREDICCION_PADDING_PCT = 0.45        # Margen extra alrededor de la zona predicha
    PREDICCION_VEL_MIN = 0.8             # Velocidad mínima (px/frame) para ser candidato
    PREDICCION_MAX_ZONAS = 8             # Límite para evitar excesos de recortes

    # ==========================================
    # ORQUESTADOR DE ALERTAS
    # ==========================================
    ALERTA_FRAMES_CONSECUTIVOS = 6   # Frames seguidos con objetivo antes de activar una sesión
    ALERTA_TIEMPO_REARME = 1.0       # Segundos sin objetivos para reiniciar contadores
    ALERTA_DURACION_MAX = 60.0       # Máximo de segundos de una sesión antes de cortarla

    # ==========================================
    # EVENTOS / BACKEND
    # ==========================================
    BACKEND_URL = None               # URL opcional para notificaciones REST
    BACKEND_TIMEOUT = 3.0            # Timeout de peticiones REST
    SIMULAR_EVENTOS_EN_LOG = True    # Si no hay backend, al menos dejar constancia en logs


    # ==========================================
    # MEJORAS (filtro IA + dataset)
    # ==========================================
    USAR_FILTRO_IA_EN_MOVIMIENTO = True
    GUARDAR_CROPS_ENTRENAMIENTO = False  # Se configura manualmente según necesidad
    RUTA_DATASET_CAPTURE = "data/dataset_capture"

    # Auto-tuning según coste observado
    AUTO_DESACTIVAR_FILTRO_IA = True
    LIMITE_MS_INFERENCIA = 120.0  # desactiva filtro IA en movimiento si la media supera este valor

    # ==========================================
    # CALIDAD DE DATASET (filtros para crops)
    # ==========================================
    CROP_GUARDAR_SOLO_VALIDADOS = False   # True = solo guardar si YOLO confirmó algo
    CROP_MIN_SCORE = 0.35                  # Score mínimo para guardar crop validado
    CROP_MIN_AREA_PX = 100                 # Área mínima en píxeles para guardar
    CROP_MAX_AREA_PCT = 0.15               # Área máxima como % del frame (evitar fondos)
    CROP_GUARDAR_METADATOS = True          # Guardar JSON con info del crop
    CROP_FILTRAR_VEGETACION = True         # Aplicar filtro anti-vegetación

    # ==========================================
    # FILTRO ANTI-VEGETACIÓN
    # ==========================================
    VEGETACION_RATIO_VERDE_MIN = 0.25      # Si >25% es verde, probable vegetación
    VEGETACION_RATIO_MARRON_MIN = 0.30     # Si >30% es marrón, probable rama/tronco
    VEGETACION_DESVIACION_COLOR_MIN = 15   # Desviación estándar mínima (objetos uniformes = artificial)
    VEGETACION_BORDES_MIN_RATIO = 0.02     # Ratio mínimo de bordes (drones tienen bordes definidos)

    # ==========================================
    # HEURÍSTICAS AVANZADAS DRON
    # ==========================================
    DRON_ASPECTO_MIN = 0.5                 # Ratio mínimo ancho/alto (drones son compactos)
    DRON_ASPECTO_MAX = 2.5                 # Ratio máximo ancho/alto
    DRON_TRAYECTORIA_FRAMES = 5            # Frames para evaluar linealidad de trayectoria
    DRON_TRAYECTORIA_LINEALIDAD_MIN = 0.7  # 0-1, cuán lineal es el movimiento (1=perfectamente recto)
    DRON_PERSISTENCIA_CLASE_FRAMES = 3     # Frames como posible_dron antes de confirmar

    # ==========================================
    # TRACKER MEJORADO
    # ==========================================
    TRACKER_USAR_PREDICCION = True         # Usar velocidad para predecir posición
    TRACKER_PREDICCION_FACTOR = 0.8        # Factor de confianza en predicción (0-1)
    TRACKER_ACELERACION_MAX = 5.0          # Cambio máximo de velocidad por frame

    # ==========================================
    # ESTABILIZACIÓN DE CLASES (v3)
    # ==========================================
    # Evita que la clase de un objeto cambie constantemente entre frames.
    # Usa votación ponderada con decay temporal para determinar la clase "estable".
    TRACKER_USAR_ESTABILIZACION_CLASES = True   # Activar estabilización de clases
    TRACKER_CLASE_DECAY = 0.92                   # Decay para clases no observadas (0.9-0.95 recomendado)
                                                 # Valores altos = más memoria, más estable
                                                 # Valores bajos = más reactivo a cambios reales
    TRACKER_CLASE_MIN_FRAMES = 3                 # Frames mínimos antes de usar clase estabilizada

    # ==========================================
    # RE-IDENTIFICACIÓN (Re-ID) VISUAL
    # ==========================================
    # Permite re-identificar objetos que salen y vuelven a entrar en el frame.
    # Usa embeddings visuales (histograma color, textura) que funcionan en CPU.
    TRACKER_USAR_REID = True                     # Activar Re-ID visual
    TRACKER_REID_MAX_EDAD = 300                  # Frames máximos para recordar tracks (~10s a 30fps)
    TRACKER_REID_UMBRAL = 0.60                   # Similitud mínima para Re-ID (0.5-0.7 recomendado)
    TRACKER_REID_MAX_GUARDADOS = 50              # Máximo de tracks en la galería de Re-ID

    # Tipo de extractor de apariencia: "simple" o "multiescala"
    # - simple: más rápido, suficiente para la mayoría de casos
    # - multiescala: más robusto pero ~3x más lento
    TRACKER_APARIENCIA_TIPO = "simple"


