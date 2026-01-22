def _get_coords(box):
    """Extrae coordenadas de una caja (dict o lista/tupla)."""
    if isinstance(box, dict):
        return box["x1"], box["y1"], box["x2"], box["y2"]
    return box[0], box[1], box[2], box[3]


def calcular_iou(boxA, boxB):
    """
    Calcula la Intersección sobre Unión (IoU) entre dos cajas.
    Soporta input como diccionario {'x1':..., 'y1':...} o lista/tupla [x1, y1, x2, y2].
    """
    xA1, yA1, xA2, yA2 = _get_coords(boxA)
    xB1, yB1, xB2, yB2 = _get_coords(boxB)

    xA = max(xA1, xB1)
    yA = max(yA1, yB1)
    xB = min(xA2, xB2)
    yB = min(yA2, yB2)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (xA2 - xA1) * (yA2 - yA1)
    boxBArea = (xB2 - xB1) * (yB2 - yB1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def centro_contenido_en_caja(box_pequeno, box_grande, margen_pct: float = 0.1):
    """
    Verifica si el centro del box_pequeno está contenido dentro del box_grande.
    Útil para detectar blobs de movimiento dentro de objetos YOLO (personas, coches, etc.).
    
    Args:
        box_pequeno: Caja pequeña (blob de movimiento)
        box_grande: Caja grande (detección YOLO)
        margen_pct: Margen extra alrededor del box_grande (0.1 = 10%)
    
    Returns:
        True si el centro del box_pequeno está dentro del box_grande expandido
    """
    x1_p, y1_p, x2_p, y2_p = _get_coords(box_pequeno)
    x1_g, y1_g, x2_g, y2_g = _get_coords(box_grande)
    
    # Centro del box pequeño
    cx = (x1_p + x2_p) / 2
    cy = (y1_p + y2_p) / 2
    
    # Expandir el box grande con margen
    w_g = x2_g - x1_g
    h_g = y2_g - y1_g
    
    x1_exp = x1_g - w_g * margen_pct
    y1_exp = y1_g - h_g * margen_pct
    x2_exp = x2_g + w_g * margen_pct
    y2_exp = y2_g + h_g * margen_pct
    
    return (x1_exp <= cx <= x2_exp) and (y1_exp <= cy <= y2_exp)


def box_contenido_en_caja(box_pequeno, box_grande, umbral_contencion: float = 0.7):
    """
    Verifica si una proporción significativa del box_pequeno está contenida en box_grande.
    
    Args:
        box_pequeno: Caja pequeña (blob de movimiento)
        box_grande: Caja grande (detección YOLO)
        umbral_contencion: Proporción mínima del área del box_pequeno que debe estar dentro (0.7 = 70%)
    
    Returns:
        True si >= umbral_contencion del área del box_pequeno está dentro del box_grande
    """
    x1_p, y1_p, x2_p, y2_p = _get_coords(box_pequeno)
    x1_g, y1_g, x2_g, y2_g = _get_coords(box_grande)
    
    # Área de intersección
    x1_i = max(x1_p, x1_g)
    y1_i = max(y1_p, y1_g)
    x2_i = min(x2_p, x2_g)
    y2_i = min(y2_p, y2_g)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return False
    
    area_interseccion = (x2_i - x1_i) * (y2_i - y1_i)
    area_pequeno = max(1, (x2_p - x1_p) * (y2_p - y1_p))
    
    return (area_interseccion / area_pequeno) >= umbral_contencion
