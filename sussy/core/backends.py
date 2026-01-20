import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

LOGGER = logging.getLogger("sussy.backends")


@dataclass
class BackendInfo:
    """
    Describe un backend de inferencia disponible.
    nombre: etiqueta corta (cuda/mps/onnx/cpu)
    tipo: pila principal (torch u onnx)
    dispositivo: identificador para mover el modelo (p.ej. cuda:0, cpu)
    modelo_path: ruta del modelo a cargar con ese backend
    descripcion: texto informativo para logs/UI
    """

    nombre: str
    tipo: str
    dispositivo: str
    modelo_path: str
    descripcion: str


def _inferir_onnx_path(modelo_pt: str, modelo_onnx_cfg: Optional[str]) -> Optional[str]:
    """
    Busca el modelo ONNX correspondiente en múltiples ubicaciones:
    1. Ruta configurada explícitamente
    2. Junto al modelo .pt (si tiene ruta)
    3. En el directorio de trabajo actual
    4. En el directorio del módulo sussy
    5. Junto al ejecutable (para PyInstaller)
    """
    if modelo_onnx_cfg and os.path.isfile(modelo_onnx_cfg):
        return modelo_onnx_cfg

    if not modelo_pt:
        return None
    
    # Obtener el nombre base del modelo (sin extensión)
    nombre_base = os.path.splitext(os.path.basename(modelo_pt))[0]
    nombre_onnx = f"{nombre_base}.onnx"
    
    # Lista de directorios donde buscar
    directorios_busqueda = []
    
    # 1. Directorio del modelo .pt (si tiene ruta)
    dir_modelo = os.path.dirname(modelo_pt)
    if dir_modelo:
        directorios_busqueda.append(os.path.abspath(dir_modelo))
    
    # 2. Directorio de trabajo actual
    directorios_busqueda.append(os.getcwd())
    
    # 3. Directorio del módulo sussy
    try:
        import sussy
        dir_sussy = os.path.dirname(os.path.dirname(os.path.abspath(sussy.__file__)))
        directorios_busqueda.append(dir_sussy)
    except Exception:
        pass
    
    # 4. Directorio del ejecutable (PyInstaller)
    import sys
    if getattr(sys, 'frozen', False):
        directorios_busqueda.append(os.path.dirname(sys.executable))
    
    # 5. Directorio del script principal
    if hasattr(sys, 'argv') and sys.argv[0]:
        dir_script = os.path.dirname(os.path.abspath(sys.argv[0]))
        directorios_busqueda.append(dir_script)
    
    # Buscar en todos los directorios
    for directorio in directorios_busqueda:
        candidato = os.path.join(directorio, nombre_onnx)
        if os.path.isfile(candidato):
            LOGGER.debug("Modelo ONNX encontrado: %s", candidato)
            return candidato
    
    LOGGER.debug("No se encontró %s en: %s", nombre_onnx, directorios_busqueda)
    return None


def detectar_backends(
    preferencias: List[str],
    modelo_pt: str,
    modelo_onnx_cfg: Optional[str] = None,
) -> Tuple[Optional[BackendInfo], List[BackendInfo]]:
    """
    Construye una lista de backends disponibles y devuelve el primero
    que coincida con las preferencias (o el primero disponible).
    """
    candidatos: List[BackendInfo] = []

    torch_mod = None
    try:
        import torch as _torch  # type: ignore

        torch_mod = _torch
    except Exception:
        torch_mod = None

    # Backends Torch (CUDA/MPS/CPU)
    if torch_mod:
        if torch_mod.cuda.is_available():
            try:
                props = torch_mod.cuda.get_device_properties(0)
                desc = f"CUDA {props.name} (cc {props.major}.{props.minor})"
            except Exception:
                desc = "CUDA disponible"
            candidatos.append(
                BackendInfo(
                    nombre="cuda",
                    tipo="torch",
                    dispositivo="cuda:0",
                    modelo_path=modelo_pt,
                    descripcion=desc,
                )
            )

        if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():  # type: ignore
            candidatos.append(
                BackendInfo(
                    nombre="mps",
                    tipo="torch",
                    dispositivo="mps",
                    modelo_path=modelo_pt,
                    descripcion="Apple MPS",
                )
            )

        candidatos.append(
            BackendInfo(
                nombre="cpu",
                tipo="torch",
                dispositivo="cpu",
                modelo_path=modelo_pt,
                descripcion="CPU (PyTorch)",
            )
        )

    # Backend ONNX (requiere onnxruntime y modelo .onnx existente)
    onnx_path = _inferir_onnx_path(modelo_pt, modelo_onnx_cfg)
    if onnx_path:
        try:
            import onnxruntime as ort  # type: ignore

            providers = ort.get_available_providers()
            desc = f"ONNX Runtime ({', '.join(providers)})" if providers else "ONNX Runtime"
            candidatos.append(
                BackendInfo(
                    nombre="onnx",
                    tipo="onnx",
                    dispositivo="onnxruntime",
                    modelo_path=onnx_path,
                    descripcion=desc,
                )
            )
        except Exception:
            LOGGER.debug("onnxruntime no disponible; se omite backend ONNX.")

    seleccionado: Optional[BackendInfo] = None
    prefs = preferencias or []
    prefs = prefs if isinstance(prefs, list) else [prefs]

    for pref in prefs:
        for cand in candidatos:
            if cand.nombre == pref:
                seleccionado = cand
                break
        if seleccionado:
            break

    if not seleccionado and candidatos:
        seleccionado = candidatos[0]

    return seleccionado, candidatos


def describir_backends(disponibles: List[BackendInfo]) -> str:
    """Texto legible con los backends detectados."""
    if not disponibles:
        return "sin backends disponibles"
    return "; ".join(f"{b.nombre} -> {b.descripcion}" for b in disponibles)


