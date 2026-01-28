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
    MOVIMIENTO_UMBRAL = 30              # Umbral de diferencia entre frames (más alto = menos sensible)
    MOVIMIENTO_AREA_MIN = 80            # Área mínima en píxeles (subido para evitar ruido)
    MOVIMIENTO_CONTRASTE_MIN = 15.0     # Contraste mínimo con entorno (subido para menos falsos positivos)
    MOVIMIENTO_MIN_FRAMES = 4           # Frames mínimos para confirmar movimiento
    MOVIMIENTO_MIN_DESPLAZAMIENTO = 15  # Desplazamiento mínimo en píxeles
    MOVIMIENTO_CROP_PADDING_PCT = 0.3   # Aumenta recortes de movimiento para contexto adicional
    MOVIMIENTO_MAX_DETECCIONES = 30     # Límite de blobs por frame (reducido para rendimiento)
    MOVIMIENTO_RAFAGA_BLOBS = 25        # Si se supera, asumimos sacudida
    MOVIMIENTO_RAFAGA_FRAMES = 3        # Frames de enfriamiento tras detectar ráfaga
    MOVIMIENTO_RAFAGA_FRAMES_ACTIVACION = 2  # Frames consecutivos con ráfaga antes de pausar
    MOVIMIENTO_ANOMALIA_TOTAL = 40      # Si hay más detecciones totales, se considera anomalía
    MOVIMIENTO_ANOMALIA_POSIBLE_DRON = 5   # Nº máximo de posible_dron simultáneos
    MOVIMIENTO_ANOMALIA_FRAMES_ACTIVACION = 2  # Subido de 1 a 2 para evitar falsos positivos
    MOVIMIENTO_ANOMALIA_FRAMES_ENFRIAMIENTO = 5  # Subido de 3 a 5
    
    # Filtro de contención: descartar blobs dentro de objetos YOLO
    # Para modelo PREENTRENADO (COCO):
    # CLASES_CON_MOVIMIENTO_INTERNO = [
    #     "person", "car", "motorcycle", "bus", "truck", "bicycle",
    #     "bird", "dog", "cat", "horse", "cow", "elephant", "bear"
    # ]
    # Para modelo Fase 1.5 (ACTIVA):
    CLASES_CON_MOVIMIENTO_INTERNO = ["persona", "vehiculo_civil", "vehiculo_militar", "pajaro", "avion"]
    MOVIMIENTO_MARGEN_CONTENCION = 0.15   # Margen extra alrededor del objeto YOLO (15%)
    MOVIMIENTO_UMBRAL_CONTENCION = 0.6    # Proporción del blob que debe estar contenido (60%)
    
    # Control de blobs sin validar
    # False = Solo las detecciones validadas por YOLO pasan al tracker
    # True = Los blobs de movimiento relevantes también pasan como "movimiento" (para detectar objetos no identificados)
    INCLUIR_MOVIMIENTO_SIN_VALIDAR = True

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
    
    # =====================================================
    # SELECCIÓN DE MODELO
    # =====================================================
    # Opción 1: Modelo PREENTRENADO (detecta 80 clases COCO - personas, coches, pájaros, etc.)
    #           Mejor para uso general, webcam, vigilancia variada
    # YOLO_MODELO = "yolo11n.pt"
    # YOLO_MODELO_ONNX = None
    
    # Opción 2: Modelo PERSONALIZADO Fase 1 (solo 5 clases - NO RECOMENDADO)
    #           *persona entrenada solo en contexto militar - NO funciona bien en webcam/oficina
    # YOLO_MODELO = "../Entrenamiento/modelos/fase1_v1/weights/best.pt"
    # YOLO_MODELO_ONNX = "../Entrenamiento/exports/fase1_v1.onnx"
    
    # Opción 3: Modelo PERSONALIZADO Fase 1.5 (6 clases: persona, dron, vehiculo_civil, vehiculo_militar, pajaro, avion)
    #           Entrenado con COCO + datos propios - funciona en webcam Y contexto militar
    # YOLO_MODELO = "../Entrenamiento/modelos/fase1_5_v1/weights/best.pt"
    # YOLO_MODELO_ONNX = "../Entrenamiento/exports/fase1_5_v1.onnx"

    # Opción 4: Modelo PERSONALIZADO Fase 2 (6 clases, entrenado con datos Fase 2 + Fase 1.5)
    #           Incluye videos anotados de drones y vehículos militares
    YOLO_MODELO = "../Entrenamiento/modelos/fase2_v1/weights/best.pt"
    YOLO_MODELO_ONNX = "../Entrenamiento/exports/fase2_v1.onnx"
    
    YOLO_IMG_SIZE = 640           # Tamaño de entrada para YOLO
    YOLO_DEVICE = None            # "cuda", "cpu", "mps", "auto"/None
    YOLO_HALF = False             # FP16 si la GPU lo soporta
    YOLO_MAX_DET = 300            # Límite de detecciones por frame
    YOLO_VID_STRIDE = 1           # stride para vídeo; >1 baja coste
    YOLO_CONF_UMBRAL = 0.35
    
    # Clases para modelo PREENTRENADO (COCO) - Opción 1
    # YOLO_CLASES_PERMITIDAS = [
    #     "person",
    #     "bicycle", "car", "motorcycle", "bus", "truck",
    #     "airplane", "bird"
    # ]
    # CLASES_ALERTA = ["bird", "airplane"]
    
    # Clases para modelo Fase 1.5 - Opción 3 (ACTIVA)
    YOLO_CLASES_PERMITIDAS = ["persona", "dron", "vehiculo_civil", "vehiculo_militar", "pajaro", "avion"]
    CLASES_ALERTA = ["dron", "pajaro", "avion"]

    # ==========================================
    # TRACKER
    # ==========================================
    TRACKER_MAX_FRAMES_LOST = 10
    TRACKER_IOU_THRESHOLD = 0.3
    TRACKER_MATCH_DIST = 100

    # ==========================================
    # RELEVANCIA DE OBJETOS EN MOVIMIENTO
    # ==========================================
    RELEVANCIA_AREA_DRON_MAX = 0.003    # Proporción del frame; por debajo se considera minúsculo (posible dron)
    RELEVANCIA_AREA_DRON_MIN = 0.0001   # Área mínima relativa para no ser ruido (nuevo)
    RELEVANCIA_AREA_RAMA_MIN = 0.015    # Por encima suelen ser ramas/paneles que ocupan demasiado (bajado)
    RELEVANCIA_VEL_MIN = 2.0            # Pixels por frame para considerar movimiento real (subido de 1.2)
    RELEVANCIA_VEL_DRON_MIN = 3.0       # Velocidad mínima específica para clasificar como dron (nuevo)
    RELEVANCIA_BORDE_PCT = 0.10         # Porcentaje de margen: movimiento pegado al borde (subido de 0.08)
    RELEVANCIA_ASPECTO_RAMAS = 5.0      # Relaciones de aspecto muy alargadas (bajado de 6.0)

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
    DRON_ASPECTO_MIN = 0.6                 # Ratio mínimo ancho/alto (drones son compactos) - más estricto
    DRON_ASPECTO_MAX = 2.0                 # Ratio máximo ancho/alto - más estricto
    DRON_TRAYECTORIA_FRAMES = 8            # Frames para evaluar linealidad de trayectoria (subido)
    DRON_TRAYECTORIA_LINEALIDAD_MIN = 0.75 # 0-1, cuán lineal es el movimiento (subido de 0.7)
    DRON_PERSISTENCIA_CLASE_FRAMES = 6     # Frames como posible_dron antes de confirmar (subido de 3)
    DRON_AREA_MIN_PX = 100                 # Área mínima en píxeles para considerar como dron (nuevo)
    DRON_SCORE_BASE = 0.3                  # Score base para posible_dron (nuevo)

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


