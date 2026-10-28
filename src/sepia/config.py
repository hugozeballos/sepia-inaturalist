"""Configuracion de una corrida. Se carga desde un YAML en configs/."""
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

from dataclasses import dataclass, asdict, field

API_URL = "https://api.inaturalist.org/v1/observations"
SEPIA_OFFICINALIS = 151429  # taxon_id de Sepia officinalis en iNaturalist

# iNaturalist pide no superar ~1 consulta por segundo a la API y no bajar
# mas de ~5 GB de fotos por hora. Estos limites quedan por debajo de eso.
API_PAUSE_S = 1.1
PHOTO_PAUSE_S = 0.5
MAX_GB_PER_HOUR = 4.0

PHOTO_SIZES = ("square", "small", "medium", "large", "original")


# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------

@dataclass
class Config:
    """Parametros de la descarga. Se guardan en manifest.json."""
    base_dir: str
    taxon_id: int = SEPIA_OFFICINALIS
    # Rectangulo amplio (sur, oeste, norte, este). Pendiente: definirlo
    # desde el PPT de sectores de Martin. None = sin filtro espacial.
    bbox: tuple[float, float, float, float] | None = None
    d1: str | None = None              # fecha minima 'YYYY-MM-DD'
    d2: str | None = None              # fecha maxima 'YYYY-MM-DD'
    quality_grade: str | None = None   # 'research', 'needs_id' o None (todas)
    photo_size: str = "original"
    per_page: int = 200                # maximo que permite la API
    user_agent: str = "sepia-inat-research (contact: hzeballosmeneses@gmail.com)"
    # Opcionales para scripts/download.py
    ids_file: str | None = None        # lista de IDs (uno por linea): descarga solo esos (piloto)
    ids_excel: str | None = None       # alternativa: IDs desde la columna url de un Excel
    resize: tuple[int, ...] = (512, 1024)  # versiones reducidas a generar
    # Triaje con modelo de vision (ver src/sepia/triage.py)
    triage: dict | None = None

    def query_params(self) -> dict:
        """Parametros de la consulta a la API (sin paginacion)."""
        p = {"taxon_id": self.taxon_id, "photos": "true",
             "order_by": "id", "order": "asc", "per_page": self.per_page}
        if self.bbox:
            p.update(dict(zip(("swlat", "swlng", "nelat", "nelng"), self.bbox)))
        if self.d1:
            p["d1"] = self.d1
        if self.d2:
            p["d2"] = self.d2
        if self.quality_grade:
            p["quality_grade"] = self.quality_grade
        return p




def load_config(path: str) -> Config:
    """Lee un YAML de configs/ y devuelve un Config."""
    import yaml
    data = yaml.safe_load(open(path, encoding="utf-8"))
    if data.get("bbox") is not None:
        data["bbox"] = tuple(data["bbox"])
    if data.get("resize") is not None:
        data["resize"] = tuple(data["resize"])
    return Config(**data)
