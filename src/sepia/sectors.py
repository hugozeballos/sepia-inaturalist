"""
Sectores A-E de Martin (configs/sectors.yaml) y asignacion de cada registro a un sector.

Regla fija, con las coordenadas de la API (nunca modificadas):
  1. Se buscan los sectores cuyo rectangulo contiene el punto (bordes incluidos).
  2. Se descartan los sectores en cuyas zonas excluidas cae el punto.
  3. Queda 1 sector -> site = ese sector.
     Quedan 2 (solapes chicos entre rectangulos) -> el primero en el orden de sectors.yaml,
     y se anota en 'site_overlap' (PENDIENTE con Martin).
     No queda ninguno -> excluido: 'outside_sectors' (fuera de todo rectangulo) o
     'excluded_zone:<sector>:<zona>' (dentro de un rectangulo, pero en una zona excluida).
Sin dependencias externas: punto en poligono por el metodo del rayo (ray casting).
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_SECTORS_FILE = Path(__file__).resolve().parents[2] / "configs" / "sectors.yaml"


def load_sectors(path: str | Path | None = None) -> list[dict]:
    """Lee configs/sectors.yaml. Devuelve la lista de sectores en su orden."""
    import yaml
    path = Path(path) if path else DEFAULT_SECTORS_FILE
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sectors = data["sectors"]
    for s in sectors:
        s["bbox"] = tuple(float(v) for v in s["bbox"])
        s["exclude"] = s.get("exclude") or {}
        south, west, north, east = s["bbox"]
        if not (south < north and west < east):
            raise ValueError(f"sector {s['name']}: bbox must be [south, west, north, east]")
    return sectors


def get_sector(name: str, path: str | Path | None = None) -> dict:
    """Un sector por su nombre ('A'...'E')."""
    for s in load_sectors(path):
        if s["name"] == name:
            return s
    raise KeyError(f"sector {name!r} is not in sectors.yaml")


def in_bbox(lat: float, lon: float, bbox) -> bool:
    """Punto dentro del rectangulo [sur, oeste, norte, este], bordes incluidos."""
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


def in_polygon(lat: float, lon: float, polygon) -> bool:
    """Punto dentro de un poligono [[lon, lat], ...] (metodo del rayo)."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                inside = not inside
    return inside


def assign_sector(lat, lon, sectors: list[dict]) -> dict:
    """
    Sector de un punto. Devuelve {'site', 'site_overlap', 'sector_reason'}:
    site = 'A'...'E' o None; sector_reason = None si tiene sector, si no el motivo de exclusion.
    """
    if lat is None or lon is None or lat != lat or lon != lon:      # sin coordenadas o NaN
        return {"site": None, "site_overlap": None, "sector_reason": "no_coordinates"}
    hits, zones = [], []
    for s in sectors:
        if not in_bbox(lat, lon, s["bbox"]):
            continue
        zone = next((z for z, poly in s["exclude"].items() if in_polygon(lat, lon, poly)), None)
        if zone:
            zones.append(f"excluded_zone:{s['name']}:{zone}")
        else:
            hits.append(s["name"])
    if hits:
        return {"site": hits[0], "site_overlap": ",".join(hits) if len(hits) > 1 else None,
                "sector_reason": None}
    return {"site": None, "site_overlap": None,
            "sector_reason": zones[0] if zones else "outside_sectors"}


def assign_sectors(obs, sectors: list[dict]):
    """Agrega site, site_overlap y sector_reason a la tabla de registros (copia)."""
    import pandas as pd
    out = obs.copy()
    res = [assign_sector(a, b, sectors) for a, b in zip(out["latitude"], out["longitude"])]
    res = pd.DataFrame(res, index=out.index)
    for c in res.columns:
        out[c] = res[c]
    return out


def overlaps(sectors: list[dict]) -> list[tuple[str, str, tuple]]:
    """Pares de sectores cuyos rectangulos se cruzan, con el rectangulo comun (para revisar)."""
    out = []
    for i, a in enumerate(sectors):
        for b in sectors[i + 1:]:
            s = max(a["bbox"][0], b["bbox"][0]); w = max(a["bbox"][1], b["bbox"][1])
            n = min(a["bbox"][2], b["bbox"][2]); e = min(a["bbox"][3], b["bbox"][3])
            if s < n and w < e:
                out.append((a["name"], b["name"], (s, w, n, e)))
    return out
