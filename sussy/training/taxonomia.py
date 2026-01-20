"""
Sistema de Taxonomía Jerárquica para Sussy.

Define la estructura completa de clases, subtipos y atributos para
el entrenamiento de modelos de detección y clasificación.

La taxonomía sigue una estructura jerárquica:
  Categoría > Subcategoría > Subtipo > Atributos

Esto permite:
- Detección de alto nivel (Persona, Vehículo, Animal)
- Clasificación detallada (tipo de vehículo, subtipo)
- Atributos adicionales (armamento, rol, color, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum, auto
import json
import yaml
from pathlib import Path


# ==============================================================================
# ENUMERACIONES BASE
# ==============================================================================

class Rol(Enum):
    CIVIL = "civil"
    MILITAR = "militar"
    POLICIA = "policia"
    SANITARIO = "sanitario"
    BOMBEROS = "bomberos"
    DESCONOCIDO = "desconocido"


class TipoCarga(Enum):
    SIN_CARGA = "sin_carga"
    PAQUETE = "paquete"
    CAMARA = "camara"
    SENSOR = "sensor"
    SUMINISTROS = "suministros"
    ARMAMENTO = "armamento"
    DESCONOCIDO = "desconocido"


class Armamento(Enum):
    SIN_ARMAS = "sin_armas"
    ARMA_CORTA = "arma_corta"
    ARMA_LARGA = "arma_larga"
    ESCOPETA = "escopeta"
    FUSIL_ASALTO = "fusil_asalto"
    SUBFUSIL = "subfusil"
    FRANCOTIRADOR = "francotirador"
    ARMA_BLANCA = "arma_blanca"
    EXPLOSIVO = "explosivo"
    OBJETO_SOSPECHOSO = "objeto_sospechoso"


class AccionPersona(Enum):
    CAMINANDO = "caminando"
    CORRIENDO = "corriendo"
    AGACHADO = "agachado"
    TUMBADO = "tumbado"
    CARGANDO_OBJETO = "cargando_objeto"
    APUNTANDO = "apuntando"
    MANIOBRANDO = "maniobrando"
    AGRESION = "agresion"
    ESTATICO = "estatico"


# ==============================================================================
# DEFINICIÓN DE TAXONOMÍA
# ==============================================================================

TAXONOMIA = {
    # =========================================================================
    # PERSONA
    # =========================================================================
    "Persona": {
        "id_base": 0,
        "descripcion": "Ser humano detectado",
        "subtipos": {
            "adulto": {"id": 1, "descripcion": "Persona adulta"},
            "nino": {"id": 2, "descripcion": "Menor de edad"},
            "anciano": {"id": 3, "descripcion": "Persona mayor"},
        },
        "atributos": {
            "genero": {
                "tipo": "enum",
                "valores": ["masculino", "femenino", "indeterminado"],
            },
            "rol": {
                "tipo": "enum",
                "valores": ["civil", "militar", "policia", "sanitario", "bomberos"],
            },
            "mochila": {"tipo": "bool"},
            "casco": {"tipo": "bool"},
            "chaleco": {"tipo": "bool"},
            "equipo_tactico": {"tipo": "bool"},
            "armamento": {
                "tipo": "enum",
                "valores": [
                    "sin_armas", "arma_corta", "arma_larga", "escopeta",
                    "fusil_asalto", "subfusil", "francotirador",
                    "arma_blanca", "explosivo", "objeto_sospechoso"
                ],
            },
            "pose": {
                "tipo": "enum",
                "valores": ["de_pie", "sentado", "agachado", "tumbado"],
            },
            "accion": {
                "tipo": "enum",
                "valores": [
                    "caminando", "corriendo", "agachado", "tumbado",
                    "cargando_objeto", "apuntando", "maniobrando", "agresion"
                ],
            },
        },
    },

    # =========================================================================
    # ANIMAL
    # =========================================================================
    "Animal": {
        "id_base": 10,
        "descripcion": "Animal detectado",
        "subtipos": {
            "perro": {"id": 11},
            "gato": {"id": 12},
            "pajaro": {"id": 13},
            "caballo": {"id": 14},
            "ganado": {"id": 15},
            "fauna_silvestre": {"id": 16},
        },
        "atributos": {
            "tamano": {
                "tipo": "enum",
                "valores": ["pequeno", "mediano", "grande"],
            },
            "color": {"tipo": "texto"},
            "pose": {
                "tipo": "enum",
                "valores": ["de_pie", "sentado", "tumbado", "volando", "nadando"],
            },
            "domestico": {"tipo": "bool"},
        },
    },

    # =========================================================================
    # VEHÍCULO AÉREO
    # =========================================================================
    "VehiculoAereo": {
        "id_base": 20,
        "descripcion": "Vehículo aéreo",
        "subcategorias": {
            "Helicoptero": {
                "id_base": 21,
                "subtipos": {
                    "ligero": {"id": 211},
                    "transporte": {"id": 212},
                    "ataque": {"id": 213},
                    "utilitario": {"id": 214},
                },
                "atributos": {
                    "rol": {"tipo": "enum", "valores": ["civil", "militar", "policia", "sanitario"]},
                    "modelo": {"tipo": "texto"},
                    "pais": {"tipo": "texto"},
                    "armamento": {"tipo": "bool"},
                    "carga_externa": {"tipo": "bool"},
                },
            },
            "Avion": {
                "id_base": 22,
                "subtipos": {
                    "jet": {"id": 221},
                    "helice": {"id": 222},
                    "caza": {"id": 223},
                    "transporte": {"id": 224},
                    "carga": {"id": 225},
                    "anfibio": {"id": 226},
                },
                "atributos": {
                    "rol": {"tipo": "enum", "valores": ["civil", "militar", "privado"]},
                    "modelo": {"tipo": "texto"},
                    "pais": {"tipo": "texto"},
                    "armamento": {"tipo": "bool"},
                },
            },
            "Dron": {
                "id_base": 23,
                "subtipos": {
                    "cuadricoptero": {"id": 231},
                    "hexacoptero": {"id": 232},
                    "ala_fija": {"id": 233},
                    "vtol": {"id": 234},
                },
                "atributos": {
                    "rol": {"tipo": "enum", "valores": ["civil", "militar", "comercial", "hobby"]},
                    "carga": {"tipo": "bool"},
                    "tipo_carga": {
                        "tipo": "enum",
                        "valores": ["sin_carga", "camara", "sensor", "paquete", "armamento"],
                    },
                    "tamano": {"tipo": "enum", "valores": ["mini", "pequeno", "mediano", "grande"]},
                },
            },
        },
    },

    # =========================================================================
    # VEHÍCULO TERRESTRE
    # =========================================================================
    "VehiculoTerrestre": {
        "id_base": 30,
        "descripcion": "Vehículo terrestre civil o comercial",
        "subcategorias": {
            "Coche": {
                "id_base": 31,
                "subtipos": {
                    "sedan": {"id": 311},
                    "suv": {"id": 312},
                    "furgoneta": {"id": 313},
                    "pickup": {"id": 314},
                    "deportivo": {"id": 315},
                    "industrial": {"id": 316},
                },
                "atributos": {
                    "rol": {"tipo": "enum", "valores": ["civil", "policia", "taxi", "emergencias"]},
                    "modelo": {"tipo": "texto"},
                    "marca": {"tipo": "texto"},
                    "matricula": {"tipo": "texto"},
                    "color": {"tipo": "texto"},
                },
            },
            "Moto": {
                "id_base": 32,
                "subtipos": {
                    "carretera": {"id": 321},
                    "trail": {"id": 322},
                    "cross": {"id": 323},
                    "scooter": {"id": 324},
                },
                "atributos": {
                    "rol": {"tipo": "enum", "valores": ["civil", "policia", "delivery"]},
                    "modelo": {"tipo": "texto"},
                    "marca": {"tipo": "texto"},
                    "matricula": {"tipo": "texto"},
                },
            },
            "Bicicleta": {
                "id_base": 33,
                "subtipos": {
                    "urbana": {"id": 331},
                    "montana": {"id": 332},
                    "carretera": {"id": 333},
                    "electrica": {"id": 334},
                },
                "atributos": {},
            },
            "Camion": {
                "id_base": 34,
                "subtipos": {
                    "rigido": {"id": 341},
                    "articulado": {"id": 342},
                    "cisterna": {"id": 343},
                    "militar_logistico": {"id": 344},
                    "pickup_pesado": {"id": 345},
                },
                "atributos": {
                    "rol": {"tipo": "enum", "valores": ["civil", "militar", "emergencias"]},
                    "matricula": {"tipo": "texto"},
                    "modelo": {"tipo": "texto"},
                    "pais": {"tipo": "texto"},
                    "tipo_carga": {
                        "tipo": "enum",
                        "valores": ["general", "liquidos", "contenedor", "militar", "especial"],
                    },
                },
            },
        },
    },

    # =========================================================================
    # VEHÍCULO TERRESTRE MILITAR
    # =========================================================================
    "VehiculoTerrestreMilitar": {
        "id_base": 40,
        "descripcion": "Vehículo militar terrestre",
        "subcategorias": {
            "CarroCombate": {
                "id_base": 41,
                "subtipos": {
                    "MBT": {"id": 411, "descripcion": "Main Battle Tank"},
                    "tanque_ligero": {"id": 412},
                },
                "atributos": {
                    "modelo": {"tipo": "texto"},
                    "pais": {"tipo": "texto"},
                    "matricula_tactica": {"tipo": "texto"},
                    "camuflaje": {"tipo": "enum", "valores": ["verde", "desierto", "artico", "urbano", "sin_camuflaje"]},
                    "insignia": {"tipo": "bool"},
                },
            },
            "VCI": {
                "id_base": 42,
                "descripcion": "Vehículo de Combate de Infantería",
                "subtipos": {
                    "IFV": {"id": 421, "descripcion": "Infantry Fighting Vehicle"},
                    "VCI_ligero": {"id": 422},
                },
                "atributos": {
                    "modelo": {"tipo": "texto"},
                    "pais": {"tipo": "texto"},
                    "matricula_tactica": {"tipo": "texto"},
                    "camuflaje": {"tipo": "enum", "valores": ["verde", "desierto", "artico", "urbano"]},
                },
            },
            "BlindadoLigero": {
                "id_base": 43,
                "subtipos": {
                    "Humvee": {"id": 431},
                    "VAMTAC": {"id": 432},
                    "JLTV": {"id": 433},
                    "otro": {"id": 439},
                },
                "atributos": {
                    "modelo": {"tipo": "texto"},
                    "pais": {"tipo": "texto"},
                    "armamento_montado": {"tipo": "bool"},
                },
            },
            "APC": {
                "id_base": 44,
                "descripcion": "Armoured Personnel Carrier",
                "subtipos": {
                    "M113": {"id": 441},
                    "BTR": {"id": 442},
                    "Pandur": {"id": 443},
                    "otro": {"id": 449},
                },
                "atributos": {
                    "modelo": {"tipo": "texto"},
                    "pais": {"tipo": "texto"},
                },
            },
            "ArtilleriaAutopropulsada": {
                "id_base": 45,
                "subtipos": {
                    "M109": {"id": 451},
                    "PzH2000": {"id": 452},
                    "otro": {"id": 459},
                },
                "atributos": {
                    "modelo": {"tipo": "texto"},
                    "pais": {"tipo": "texto"},
                    "calibre": {"tipo": "texto"},
                },
            },
        },
    },
}


# ==============================================================================
# CLASES DE SOPORTE
# ==============================================================================

@dataclass
class ClaseDeteccion:
    """Representa una clase de detección en la taxonomía."""
    id: int
    nombre: str
    nombre_completo: str  # Ej: "VehiculoAereo.Dron.cuadricoptero"
    categoria: str
    subcategoria: Optional[str] = None
    subtipo: Optional[str] = None
    nivel: int = 1  # 1=categoría, 2=subcategoría, 3=subtipo
    atributos_disponibles: Dict[str, Dict] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)


@dataclass  
class EtiquetaDeteccion:
    """Etiqueta completa para una detección con atributos."""
    clase: ClaseDeteccion
    bbox: tuple  # (x1, y1, x2, y2) normalizado [0-1]
    confianza: float = 1.0
    atributos: Dict[str, Any] = field(default_factory=dict)
    
    def to_yolo_line(self, include_attrs: bool = False) -> str:
        """Convierte a formato YOLO: class_id cx cy w h [attrs...]"""
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        line = f"{self.clase.id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        
        if include_attrs and self.atributos:
            attrs_str = json.dumps(self.atributos, ensure_ascii=False)
            line += f" # {attrs_str}"
        
        return line
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario para JSON."""
        return {
            "clase_id": self.clase.id,
            "clase_nombre": self.clase.nombre,
            "clase_completa": self.clase.nombre_completo,
            "categoria": self.clase.categoria,
            "subcategoria": self.clase.subcategoria,
            "subtipo": self.clase.subtipo,
            "bbox": list(self.bbox),
            "confianza": self.confianza,
            "atributos": self.atributos,
        }


class GestorTaxonomia:
    """
    Gestiona la taxonomía jerárquica de clases.
    
    Permite:
    - Obtener clases por ID o nombre
    - Navegar la jerarquía
    - Generar archivos de configuración para YOLO
    - Validar etiquetas
    """
    
    def __init__(self, taxonomia: Dict = None):
        self.taxonomia = taxonomia or TAXONOMIA
        self._clases: Dict[int, ClaseDeteccion] = {}
        self._clases_por_nombre: Dict[str, ClaseDeteccion] = {}
        self._construir_indice()
    
    def _construir_indice(self):
        """Construye índices de clases desde la taxonomía."""
        for cat_nombre, cat_data in self.taxonomia.items():
            id_base = cat_data.get("id_base", 0)
            
            # Clase de categoría
            clase_cat = ClaseDeteccion(
                id=id_base,
                nombre=cat_nombre,
                nombre_completo=cat_nombre,
                categoria=cat_nombre,
                nivel=1,
                atributos_disponibles=cat_data.get("atributos", {}),
            )
            self._registrar_clase(clase_cat)
            
            # Subtipos directos
            for subtipo_nombre, subtipo_data in cat_data.get("subtipos", {}).items():
                clase_subtipo = ClaseDeteccion(
                    id=subtipo_data.get("id", id_base + 1),
                    nombre=subtipo_nombre,
                    nombre_completo=f"{cat_nombre}.{subtipo_nombre}",
                    categoria=cat_nombre,
                    subtipo=subtipo_nombre,
                    nivel=2,
                    atributos_disponibles=cat_data.get("atributos", {}),
                )
                self._registrar_clase(clase_subtipo)
            
            # Subcategorías (para estructuras anidadas como VehiculoAereo)
            for subcat_nombre, subcat_data in cat_data.get("subcategorias", {}).items():
                subcat_id = subcat_data.get("id_base", id_base + 10)
                
                clase_subcat = ClaseDeteccion(
                    id=subcat_id,
                    nombre=subcat_nombre,
                    nombre_completo=f"{cat_nombre}.{subcat_nombre}",
                    categoria=cat_nombre,
                    subcategoria=subcat_nombre,
                    nivel=2,
                    atributos_disponibles=subcat_data.get("atributos", {}),
                )
                self._registrar_clase(clase_subcat)
                
                # Subtipos de la subcategoría
                for subtipo_nombre, subtipo_data in subcat_data.get("subtipos", {}).items():
                    clase_subtipo = ClaseDeteccion(
                        id=subtipo_data.get("id", subcat_id + 1),
                        nombre=subtipo_nombre,
                        nombre_completo=f"{cat_nombre}.{subcat_nombre}.{subtipo_nombre}",
                        categoria=cat_nombre,
                        subcategoria=subcat_nombre,
                        subtipo=subtipo_nombre,
                        nivel=3,
                        atributos_disponibles=subcat_data.get("atributos", {}),
                    )
                    self._registrar_clase(clase_subtipo)
    
    def _registrar_clase(self, clase: ClaseDeteccion):
        """Registra una clase en los índices."""
        self._clases[clase.id] = clase
        self._clases_por_nombre[clase.nombre_completo] = clase
        self._clases_por_nombre[clase.nombre] = clase  # También por nombre corto
    
    def obtener_clase(self, id_o_nombre) -> Optional[ClaseDeteccion]:
        """Obtiene una clase por ID o nombre."""
        if isinstance(id_o_nombre, int):
            return self._clases.get(id_o_nombre)
        return self._clases_por_nombre.get(id_o_nombre)
    
    def obtener_todas_clases(self, nivel: Optional[int] = None) -> List[ClaseDeteccion]:
        """Obtiene todas las clases, opcionalmente filtradas por nivel."""
        if nivel is None:
            return list(self._clases.values())
        return [c for c in self._clases.values() if c.nivel == nivel]
    
    def obtener_categorias(self) -> List[ClaseDeteccion]:
        """Obtiene solo las categorías de nivel 1."""
        return self.obtener_todas_clases(nivel=1)
    
    def obtener_hijos(self, clase: ClaseDeteccion) -> List[ClaseDeteccion]:
        """Obtiene las subclases directas de una clase."""
        hijos = []
        prefijo = clase.nombre_completo + "."
        for c in self._clases.values():
            if c.nombre_completo.startswith(prefijo):
                # Solo hijos directos (un nivel más)
                resto = c.nombre_completo[len(prefijo):]
                if "." not in resto:
                    hijos.append(c)
        return hijos
    
    def obtener_ancestros(self, clase: ClaseDeteccion) -> List[ClaseDeteccion]:
        """Obtiene los ancestros de una clase (de más general a más específico)."""
        ancestros = []
        partes = clase.nombre_completo.split(".")
        for i in range(1, len(partes)):
            nombre_ancestro = ".".join(partes[:i])
            ancestro = self._clases_por_nombre.get(nombre_ancestro)
            if ancestro:
                ancestros.append(ancestro)
        return ancestros
    
    def generar_yaml_yolo(self, path: Path, nivel_max: int = 3) -> Path:
        """
        Genera archivo YAML de clases para YOLO.
        
        Args:
            path: Ruta donde guardar el archivo
            nivel_max: Nivel máximo de detalle a incluir
        
        Returns:
            Path del archivo generado
        """
        clases = [c for c in self._clases.values() if c.nivel <= nivel_max]
        clases.sort(key=lambda c: c.id)
        
        # Crear mapeo id -> nombre
        names = {c.id: c.nombre_completo for c in clases}
        
        data = {
            "path": ".",
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(names),
            "names": names,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        
        return path
    
    def generar_json_taxonomia(self, path: Path) -> Path:
        """Exporta la taxonomía completa a JSON para referencia."""
        data = {
            "version": "1.0",
            "descripcion": "Taxonomía jerárquica de clases para Sistema Sussy",
            "clases": [
                {
                    "id": c.id,
                    "nombre": c.nombre,
                    "nombre_completo": c.nombre_completo,
                    "categoria": c.categoria,
                    "subcategoria": c.subcategoria,
                    "subtipo": c.subtipo,
                    "nivel": c.nivel,
                    "atributos": c.atributos_disponibles,
                }
                for c in sorted(self._clases.values(), key=lambda x: x.id)
            ],
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return path
    
    def validar_etiqueta(self, etiqueta: EtiquetaDeteccion) -> List[str]:
        """
        Valida una etiqueta contra la taxonomía.
        
        Returns:
            Lista de errores (vacía si es válida)
        """
        errores = []
        
        # Verificar que la clase existe
        if etiqueta.clase.id not in self._clases:
            errores.append(f"Clase ID {etiqueta.clase.id} no existe en taxonomía")
            return errores
        
        clase = self._clases[etiqueta.clase.id]
        
        # Verificar atributos
        for attr_nombre, attr_valor in etiqueta.atributos.items():
            if attr_nombre not in clase.atributos_disponibles:
                errores.append(f"Atributo '{attr_nombre}' no válido para clase {clase.nombre}")
                continue
            
            attr_def = clase.atributos_disponibles[attr_nombre]
            if attr_def.get("tipo") == "enum":
                valores_validos = attr_def.get("valores", [])
                if attr_valor not in valores_validos:
                    errores.append(
                        f"Valor '{attr_valor}' no válido para atributo '{attr_nombre}'. "
                        f"Valores válidos: {valores_validos}"
                    )
            elif attr_def.get("tipo") == "bool":
                if not isinstance(attr_valor, bool):
                    errores.append(f"Atributo '{attr_nombre}' debe ser booleano")
        
        return errores
    
    def __len__(self) -> int:
        return len(self._clases)
    
    def __iter__(self):
        return iter(self._clases.values())


# ==============================================================================
# INSTANCIA GLOBAL
# ==============================================================================

_gestor_taxonomia: Optional[GestorTaxonomia] = None


def obtener_taxonomia() -> GestorTaxonomia:
    """Obtiene la instancia global del gestor de taxonomía."""
    global _gestor_taxonomia
    if _gestor_taxonomia is None:
        _gestor_taxonomia = GestorTaxonomia()
    return _gestor_taxonomia


def resetear_taxonomia():
    """Resetea la instancia global (útil para tests)."""
    global _gestor_taxonomia
    _gestor_taxonomia = None


# ==============================================================================
# UTILIDADES
# ==============================================================================

def imprimir_taxonomia():
    """Imprime la taxonomía de forma legible."""
    gestor = obtener_taxonomia()
    
    print("\n" + "=" * 60)
    print("TAXONOMÍA DE CLASES - SISTEMA SUSSY")
    print("=" * 60)
    
    for cat in gestor.obtener_categorias():
        print(f"\n[{cat.id:3d}] {cat.nombre}")
        
        for hijo in gestor.obtener_hijos(cat):
            print(f"  [{hijo.id:3d}] {hijo.nombre}")
            
            for nieto in gestor.obtener_hijos(hijo):
                print(f"    [{nieto.id:3d}] {nieto.nombre}")
                
                if nieto.atributos_disponibles:
                    attrs = ", ".join(nieto.atributos_disponibles.keys())
                    print(f"          Atributos: {attrs}")


if __name__ == "__main__":
    imprimir_taxonomia()
    
    # Generar archivos de configuración
    gestor = obtener_taxonomia()
    
    print(f"\nTotal de clases: {len(gestor)}")
    print(f"Categorías: {len(gestor.obtener_categorias())}")
