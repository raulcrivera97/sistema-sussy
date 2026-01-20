# Sistema de Entrenamiento Jerárquico - Sussy

## Visión General

El sistema de entrenamiento de Sussy utiliza una **arquitectura jerárquica en cascada** para detectar y clasificar objetos con alto nivel de detalle.

```
                    ┌─────────────────────────────────────┐
                    │         IMAGEN DE ENTRADA           │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │   FASE 1: DETECTOR BASE (YOLO)      │
                    │   Detecta: Persona, Animal,         │
                    │   VehiculoAereo, VehiculoTerrestre, │
                    │   VehiculoTerrestreMilitar          │
                    └─────────────────┬───────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
    ┌───────▼───────┐         ┌───────▼───────┐         ┌───────▼───────┐
    │  Clasificador │         │  Clasificador │         │  Clasificador │
    │   Personas    │         │ VehiculoAereo │         │   Militares   │
    │   ─────────   │         │   ─────────   │         │   ─────────   │
    │ adulto, niño  │         │ Heli, Avion,  │         │ Tanque, VCI   │
    │   anciano     │         │    Dron       │         │   APC, ...    │
    └───────┬───────┘         └───────┬───────┘         └───────┬───────┘
            │                         │                         │
    ┌───────▼───────┐         ┌───────▼───────┐         ┌───────▼───────┐
    │   Atributos   │         │   Subtipos    │         │   Modelos     │
    │   ─────────   │         │   ─────────   │         │   ─────────   │
    │ rol, arma,    │         │ cuadricoptero │         │ M1 Abrams,    │
    │ acción, etc.  │         │ hexacoptero   │         │ Leopard 2...  │
    └───────────────┘         └───────────────┘         └───────────────┘
```

## Fases de Entrenamiento

### FASE 1: Detección Base

**Objetivo:** Detectar objetos y clasificarlos en 5 categorías principales.

| ID | Categoría | Descripción |
|----|-----------|-------------|
| 0 | Persona | Ser humano |
| 10 | Animal | Cualquier animal |
| 20 | VehiculoAereo | Helicópteros, aviones, drones |
| 30 | VehiculoTerrestre | Coches, motos, camiones civiles |
| 40 | VehiculoTerrestreMilitar | Tanques, blindados, APCs |

**Modelo:** YOLO11 (detección de objetos)
**Dataset:** Imágenes con bounding boxes etiquetados
**Épocas sugeridas:** 100

### FASE 2: Clasificación de Subcategorías

**Objetivo:** Para cada detección de Fase 1, clasificar en subcategoría.

Ejemplo para `VehiculoAereo`:

| ID | Subcategoría |
|----|--------------|
| 21 | Helicoptero |
| 22 | Avion |
| 23 | Dron |

**Modelo:** YOLO11-cls (clasificación)
**Dataset:** Crops de las detecciones de Fase 1
**Épocas sugeridas:** 50 por categoría

### FASE 3: Clasificación de Subtipos

**Objetivo:** Clasificación fina dentro de cada subcategoría.

Ejemplo para `VehiculoAereo.Dron`:

| ID | Subtipo |
|----|---------|
| 231 | cuadricoptero |
| 232 | hexacoptero |
| 233 | ala_fija |
| 234 | vtol |

**Modelo:** YOLO11-cls (clasificación)
**Épocas sugeridas:** 30 por subcategoría

### FASE 4: Predicción de Atributos

**Objetivo:** Predecir atributos adicionales para cada detección.

Ejemplo de atributos para `Persona`:

| Atributo | Tipo | Valores |
|----------|------|---------|
| genero | enum | masculino, femenino, indeterminado |
| rol | enum | civil, militar, policia, sanitario, bomberos |
| mochila | bool | true/false |
| casco | bool | true/false |
| armamento | enum | sin_armas, arma_corta, fusil_asalto, ... |
| accion | enum | caminando, corriendo, apuntando, ... |

**Modelo:** Clasificador multi-label o heads separados
**Épocas sugeridas:** 30 por tipo de atributo

## Estructura de Carpetas del Dataset

```
datasets/
├── fase1_categorias/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│
├── fase2_vehiculoaereo/
│   ├── data.yaml
│   ├── train/
│   │   ├── Helicoptero/
│   │   ├── Avion/
│   │   └── Dron/
│   └── val/
│
├── fase3_vehiculoaereo_dron/
│   ├── data.yaml
│   ├── train/
│   │   ├── cuadricoptero/
│   │   ├── hexacoptero/
│   │   ├── ala_fija/
│   │   └── vtol/
│   └── val/
│
└── atributos/
    ├── persona_rol/
    ├── persona_armamento/
    └── dron_carga/
```

## Uso del Sistema

### 1. Inicializar Proyecto

```python
from sussy.training.entrenamiento import iniciar_proyecto_entrenamiento

entrenador = iniciar_proyecto_entrenamiento("mi_proyecto")
entrenador.imprimir_plan()
```

### 2. Preparar Datasets

```python
from sussy.training.dataset import crear_dataset_desde_carpeta

# Importar imágenes ya organizadas
gestor = crear_dataset_desde_carpeta(
    path_imagenes="imagenes_raw",
    path_salida="datasets/fase1",
    nombre="detector_base",
)

# Ver estadísticas
gestor.imprimir_resumen()

# Exportar a formato YOLO
gestor.exportar_yolo()
```

### 3. Entrenar Modelos

```python
# Fase 1: Detector base
resultado = entrenador.entrenar_fase1_deteccion()
print(f"Modelo guardado en: {resultado.path_modelo}")

# Fase 2: Clasificadores por categoría
resultado = entrenador.entrenar_fase2_clasificacion("VehiculoAereo")

# Fase 3: Subtipos
resultado = entrenador.entrenar_fase3_subtipos("VehiculoAereo", "Dron")
```

### 4. Integrar en el Pipeline

Una vez entrenados, los modelos se integran en el pipeline de Sussy:

```python
# En config.py
YOLO_MODELO = "models/fase1/weights/best.pt"

# Clasificadores adicionales se cargan dinámicamente
CLASIFICADORES = {
    "VehiculoAereo": "models/fase2_vehiculoaereo/weights/best.pt",
    "VehiculoAereo.Dron": "models/fase3_vehiculoaereo_dron/weights/best.pt",
}
```

## Consideraciones para RTX 5080 (sm_120)

Dado que PyTorch no tiene kernels para la arquitectura Blackwell (sm_120), el entrenamiento debe hacerse de una de estas formas:

### Opción A: Entrenar en la nube
- Google Colab (GPU gratis)
- Kaggle Notebooks
- AWS/GCP/Azure con GPUs compatibles

### Opción B: Entrenar en CPU local
- Más lento pero funciona
- Usar batch_size pequeño (4-8)
- Épocas reducidas con early stopping

### Opción C: Usar otro PC con GPU compatible
- Entrenar en máquina con GPU CUDA compatible
- Exportar modelos a ONNX
- Ejecutar inferencia en RTX 5080 con ONNX Runtime

```python
# Forzar entrenamiento en CPU
config = ConfiguracionEntrenamiento(
    ...
    device="cpu",
    batch_size=4,
    workers=2,
)
```

## Formato de Etiquetas

### Formato YOLO (básico)

```
# clase_id cx cy w h
0 0.5 0.5 0.3 0.4
23 0.2 0.3 0.1 0.1
```

### Formato JSON (extendido con atributos)

```json
{
  "imagen": "frame_001.jpg",
  "etiquetas": [
    {
      "clase_id": 0,
      "clase_nombre": "Persona",
      "clase_completa": "Persona.adulto",
      "bbox": [0.35, 0.3, 0.65, 0.9],
      "confianza": 0.95,
      "atributos": {
        "genero": "masculino",
        "rol": "militar",
        "casco": true,
        "armamento": "fusil_asalto",
        "accion": "caminando"
      }
    },
    {
      "clase_id": 231,
      "clase_nombre": "cuadricoptero",
      "clase_completa": "VehiculoAereo.Dron.cuadricoptero",
      "bbox": [0.1, 0.1, 0.25, 0.25],
      "confianza": 0.87,
      "atributos": {
        "rol": "militar",
        "carga": true,
        "tipo_carga": "camara"
      }
    }
  ]
}
```

## Herramientas de Etiquetado Recomendadas

1. **CVAT** - Etiquetado web con soporte jerárquico
2. **Label Studio** - Flexible, soporta atributos
3. **Labelme** - Simple, JSON nativo
4. **Roboflow** - Cloud, augmentación automática

## Métricas de Evaluación

Para cada fase se evalúan métricas específicas:

| Fase | Métricas Principales |
|------|---------------------|
| 1 (Detección) | mAP@50, mAP@50-95, Precision, Recall |
| 2 (Clasificación) | Top-1 Accuracy, Top-5 Accuracy |
| 3 (Subtipos) | Accuracy, F1-Score por clase |
| 4 (Atributos) | Accuracy por atributo, F1 multi-label |

## Próximos Pasos

1. **Recopilar datos** - Imágenes representativas de cada clase
2. **Etiquetar** - Usar herramientas recomendadas
3. **Organizar** - Estructura de carpetas según taxonomía
4. **Entrenar Fase 1** - Detector base
5. **Iterar** - Evaluar, añadir datos, re-entrenar
6. **Entrenar Fases 2-4** - Una vez Fase 1 sea sólida

---

*Sistema Sussy - Entrenamiento Jerárquico v1.0*
