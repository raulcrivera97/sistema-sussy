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

    # ==========================================
    # DETECCIÓN (YOLO / IA)
    # ==========================================
    YOLO_MODELO = "yolo11x.pt"
    YOLO_CONF_UMBRAL = 0.5
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
    GUARDAR_CROPS_ENTRENAMIENTO = True
    RUTA_DATASET_CAPTURE = "data/dataset_capture"


