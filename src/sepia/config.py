"""Configuracion de una corrida. Se carga desde un YAML en configs/."""
from __future__ import annotations

from dataclasses import dataclass

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
    # Nombre de la hoja de registros de esta corrida en el Excel (como 'A1', 'A2' en la tabla
    # de Martin). Sin valor: el nombre de la carpeta de la corrida.
    sheet_name: str | None = None
    # Corridas anteriores cuyos registros NO se vuelven a bajar (p. ej. ["pilot_a1"]): nombre de
    # carpeta junto a base_dir, o ruta completa. Se excluyen al armar las tablas (antes de las fotos).
    skip_ids_from: list[str] | None = None
    # Sector de Martin de esta corrida ('A'...'E', ver configs/sectors.yaml). Si se da y no hay
    # bbox, la descarga usa el rectangulo del sector. Los registros de OTRO sector no entran a las
    # tablas de esta corrida (quedan en logs/other_sector.jsonl y se bajan en la corrida de su
    # sector). Sin sector (corridas viejas): se asigna 'site' a cada registro, sin filtrar.
    sector: str | None = None
    sectors_file: str | None = None    # None = configs/sectors.yaml del repo
    resize: tuple[int, ...] = (512, 1024)  # versiones reducidas a generar
    # Triaje con modelo de vision (ver src/sepia/triage.py)
    triage: dict | None = None
    # Rasgos por foto: lado y erosion (ver src/sepia/photo_traits.py)
    photo_traits: dict | None = None
    # Marcas de picoteo por foto (ver src/sepia/bitemarks.py). Sin seccion: usa la de photo_traits
    bitemarks: dict | None = None
    # Recuadro del hueso por foto (ver src/sepia/bonebox.py). Sin seccion: usa la de photo_traits
    bonebox: dict | None = None

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
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data.get("bbox") is not None:
        data["bbox"] = tuple(data["bbox"])
    if data.get("resize") is not None:
        data["resize"] = tuple(data["resize"])
    if data.get("sector") and data.get("bbox") is None:
        from .sectors import get_sector
        data["bbox"] = get_sector(data["sector"], data.get("sectors_file"))["bbox"]
    return Config(**data)
