"""Descarga de registros y fotos de iNaturalist (fase 1)."""
from __future__ import annotations

import json
import re
import time
import datetime as dt
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import Config, API_URL, API_PAUSE_S, PHOTO_PAUSE_S, MAX_GB_PER_HOUR, PHOTO_SIZES
from .io import paths, write_jsonl, read_jsonl, _log, write_manifest


def _session(cfg: Config) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = cfg.user_agent
    return s


def _get_json(session: requests.Session, params: dict, retries: int = 5) -> dict:
    """GET a la API con reintentos y espera creciente ante errores o limite (429)."""
    for attempt in range(retries):
        try:
            r = session.get(API_URL, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            wait = 5 * 2 ** attempt
            print(f"  error ({e}); reintento en {wait} s")
            time.sleep(wait)
    raise RuntimeError("La API no respondio despues de varios intentos")


# --------------------------------------------------------------------------
# 1. Descarga de metadatos (copia cruda)
# --------------------------------------------------------------------------

def _saved_pages(p: dict) -> list[Path]:
    return sorted(p["raw"].glob("page_*.json"))


def count_results(cfg: Config) -> int:
    """Cuantos registros devuelve la consulta, sin descargarlos."""
    params = {**cfg.query_params(), "per_page": 0}
    return _get_json(_session(cfg), params)["total_results"]


def download_observations(cfg: Config, max_pages: int | None = None) -> int:
    """
    Descarga todos los registros de la consulta y guarda cada pagina tal
    como llega en raw/api_pages/page_NNNN.json.

    Pagina con id_above (no con page=N) porque la API no permite pasar de
    10.000 resultados con page. Reanudable: parte desde el ultimo ID guardado.
    Devuelve el numero de registros nuevos descargados.
    """
    p = paths(cfg)
    session = _session(cfg)
    pages = _saved_pages(p)

    # Reanudar desde el ultimo ID de la ultima pagina guardada
    last_id, n_page = 0, len(pages)
    if pages:
        last = json.loads(pages[-1].read_text())
        if last["results"]:
            last_id = max(o["id"] for o in last["results"])
        print(f"Reanudando despues del ID {last_id} ({n_page} paginas ya guardadas)")
    else:
        write_manifest(cfg, {"total_results_at_start": count_results(cfg)})

    new = 0
    while max_pages is None or n_page < max_pages:
        params = {**cfg.query_params(), "id_above": last_id}
        data = _get_json(session, params)
        results = data.get("results", [])
        if not results:
            break
        n_page += 1
        out = p["raw"] / f"page_{n_page:04d}.json"
        out.write_text(json.dumps(data, ensure_ascii=False))
        last_id = max(o["id"] for o in results)
        new += len(results)
        _log(p, "download_observations", {"page": n_page, "n": len(results), "last_id": last_id})
        print(f"pagina {n_page}: {len(results)} registros (hasta ID {last_id})")
        time.sleep(API_PAUSE_S)

    print(f"Listo: {new} registros nuevos")
    return new


def download_by_ids(cfg: Config, ids: Iterable[int], tag: str = "ids") -> int:
    """
    Descarga registros especificos por ID (p. ej., los 79 del Excel de prueba).
    Guarda la copia cruda en raw/api_pages/<tag>_NNNN.json.
    No usar junto con download_observations en la misma base_dir.
    """
    p = paths(cfg)
    session = _session(cfg)
    ids = sorted({int(i) for i in ids})
    total = 0
    for k in range(0, len(ids), cfg.per_page):
        chunk = ids[k:k + cfg.per_page]
        out = p["raw"] / f"{tag}_{k // cfg.per_page + 1:04d}.json"
        if out.exists():
            continue  # ya descargado
        data = _get_json(session, {"id": ",".join(map(str, chunk)),
                                   "per_page": cfg.per_page})
        out.write_text(json.dumps(data, ensure_ascii=False))
        found = {o["id"] for o in data.get("results", [])}
        missing = sorted(set(chunk) - found)
        if missing:
            print(f"  no encontrados (borrados o privados): {missing}")
        _log(p, "download_by_ids", {"file": out.name, "n": len(found), "missing": missing})
        total += len(found)
        time.sleep(API_PAUSE_S)
    write_manifest(cfg, {"mode": "by_ids", "n_ids": len(ids)})
    print(f"Listo: {total} registros")
    return total


def load_raw(cfg: Config) -> list[dict]:
    """Lee todas las paginas guardadas y devuelve los registros sin duplicados."""
    p = paths(cfg)
    obs = {}
    for f in sorted(p["raw"].glob("*.json")):
        for o in json.loads(f.read_text()).get("results", []):
            obs[o["id"]] = o
    return [obs[k] for k in sorted(obs)]


# --------------------------------------------------------------------------
# 2. Tablas derivadas (se regeneran desde la copia cruda)
# --------------------------------------------------------------------------

def _coords(o: dict) -> tuple[float | None, float | None]:
    """Latitud y longitud como numeros, desde geojson [lng, lat]."""
    g = o.get("geojson") or {}
    c = g.get("coordinates")
    if c and len(c) == 2:
        return float(c[1]), float(c[0])
    return None, None


def _photo_list(o: dict) -> list[dict]:
    """Fotos del registro en el orden en que las subio el observador."""
    ops = o.get("observation_photos") or []
    if ops:
        ops = sorted(ops, key=lambda x: x.get("position") or 0)
        return [x["photo"] for x in ops if x.get("photo")]
    return o.get("photos") or []


def observations_table(raw: list[dict]) -> pd.DataFrame:
    """Una fila por registro, con los metadatos del diccionario de datos."""
    rows = []
    for o in raw:
        lat, lon = _coords(o)
        rows.append({
            "observation_id": o["id"],
            "uuid": o.get("uuid"),
            "url": f"https://www.inaturalist.org/observations/{o['id']}",
            "observed_on": o.get("observed_on"),
            "time_observed_at": o.get("time_observed_at"),
            "created_at": o.get("created_at"),
            "user_login": (o.get("user") or {}).get("login"),
            "quality_grade": o.get("quality_grade"),
            "captive": o.get("captive"),
            "license": o.get("license_code"),
            "taxon_id": (o.get("taxon") or {}).get("id"),
            "taxon_name": (o.get("taxon") or {}).get("name"),
            "description": o.get("description"),
            "place_guess": o.get("place_guess"),
            "latitude": lat,
            "longitude": lon,
            "positional_accuracy": o.get("positional_accuracy"),
            "public_positional_accuracy": o.get("public_positional_accuracy"),
            "coordinates_obscured": o.get("obscured"),
            "geoprivacy": o.get("geoprivacy"),
            "taxon_geoprivacy": o.get("taxon_geoprivacy"),
            "n_photos": len(_photo_list(o)),
        })
    return pd.DataFrame(rows)


def photo_url(url: str, size: str = "original") -> str:
    """Cambia el tamano en una URL de foto de iNaturalist (square -> original, etc.)."""
    if size not in PHOTO_SIZES:
        raise ValueError(f"size debe ser uno de {PHOTO_SIZES}")
    return re.sub(r"/(square|thumb|small|medium|large|original)\.", f"/{size}.", url)


def photos_table(raw: list[dict], size: str = "original") -> pd.DataFrame:
    """Una fila por foto, con su URL de descarga y el nombre de archivo."""
    rows = []
    for o in raw:
        for pos, ph in enumerate(_photo_list(o), start=1):
            url = ph.get("url")
            if not url:
                continue
            ext = (Path(url.split("?")[0]).suffix or ".jpg").lower()
            ext = ".jpg" if ext == ".jpeg" else ext
            dims = ph.get("original_dimensions") or {}
            rows.append({
                "photo_id": ph["id"],
                "observation_id": o["id"],
                "position": pos,
                "license": ph.get("license_code"),
                "attribution": ph.get("attribution"),
                "width": dims.get("width"),
                "height": dims.get("height"),
                "download_url": photo_url(url, size),
                "file_name": f"{o['id']}_{ph['id']}{ext.lower()}",
            })
    return pd.DataFrame(rows)


def build_tables(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera tables/observations.jsonl y tables/photos.jsonl desde la copia cruda."""
    p = paths(cfg)
    raw = load_raw(cfg)
    obs = observations_table(raw)
    pho = photos_table(raw, cfg.photo_size)
    write_jsonl(obs, p["tables"] / "observations.jsonl")
    write_jsonl(pho, p["tables"] / "photos.jsonl")
    print(f"observations.jsonl: {len(obs)} | photos.jsonl: {len(pho)}")
    return obs, pho


def load_tables(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee observations.jsonl y photos.jsonl."""
    p = paths(cfg)
    return read_jsonl(p["tables"] / "observations.jsonl"), read_jsonl(p["tables"] / "photos.jsonl")


# --------------------------------------------------------------------------
# 3. Descarga de fotos
# --------------------------------------------------------------------------

def download_photos(cfg: Config, photos: pd.DataFrame | None = None,
                    limit: int | None = None) -> dict:
    """
    Descarga las fotos de photos.jsonl a photos/<size>/.
    Reanudable: salta los archivos que ya existen. Escribe primero a un
    archivo temporal y lo renombra al final, para no dejar fotos a medias.
    """
    p = paths(cfg)
    if photos is None:
        photos = read_jsonl(p["tables"] / "photos.jsonl")
    session = _session(cfg)
    stats = {"ok": 0, "skipped": 0, "failed": 0, "bytes": 0}
    window_start, window_bytes = time.time(), 0

    todo = photos if limit is None else photos.head(limit)
    for i, row in enumerate(todo.itertuples(index=False), start=1):
        dest = p["photos"] / row.file_name
        if dest.exists() and dest.stat().st_size > 0:
            stats["skipped"] += 1
            continue
        try:
            r = session.get(row.download_url, timeout=120)
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.rename(dest)
            stats["ok"] += 1
            stats["bytes"] += len(r.content)
            window_bytes += len(r.content)
        except requests.RequestException as e:
            stats["failed"] += 1
            _log(p, "download_photos_failed",
                 {"photo_id": int(row.photo_id), "url": row.download_url, "error": str(e)})

        # Respetar el limite de volumen por hora
        if window_bytes / 1e9 >= MAX_GB_PER_HOUR:
            wait = 3600 - (time.time() - window_start)
            if wait > 0:
                print(f"Limite de {MAX_GB_PER_HOUR} GB/hora alcanzado; espero {wait/60:.0f} min")
                time.sleep(wait)
            window_start, window_bytes = time.time(), 0

        if i % 100 == 0:
            print(f"{i}/{len(todo)} | nuevas {stats['ok']} | saltadas {stats['skipped']} "
                  f"| fallidas {stats['failed']} | {stats['bytes']/1e9:.2f} GB")
        time.sleep(PHOTO_PAUSE_S)

    _log(p, "download_photos", stats)
    print(f"Listo: {stats}")
    return stats


# --------------------------------------------------------------------------
# 4. Versiones reducidas (para GPT y la interfaz)
# --------------------------------------------------------------------------

def make_resized(cfg: Config, max_side: int = 512, quality: int = 85) -> int:
    """
    Crea copias reducidas en photos/<max_side>px/ a partir de los originales.
    Se pueden borrar y regenerar cuando se quiera. Reanudable.
    """
    from PIL import Image, ImageOps

    p = paths(cfg)
    out_dir = p["base"] / "photos" / f"{max_side}px"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(p["photos"].glob("*")):
        if src.suffix == ".part":
            continue
        dest = out_dir / (src.stem + ".jpg")
        if dest.exists():
            continue
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")  # respeta la orientacion
            im.thumbnail((max_side, max_side))
            im.save(dest, "JPEG", quality=quality)
        n += 1
    print(f"{n} fotos reducidas a {max_side} px en {out_dir}")
    return n


# --------------------------------------------------------------------------
# 5. Controles
# --------------------------------------------------------------------------

def check_download(cfg: Config) -> dict:
    """Resumen de control: registros, fotos esperadas vs descargadas, espacio usado."""
    p = paths(cfg)
    obs, pho = load_tables(cfg)
    files = {f.name for f in p["photos"].iterdir() if f.suffix != ".part"}
    missing = pho.loc[~pho["file_name"].isin(files), ["observation_id", "photo_id"]]
    size_gb = sum(f.stat().st_size for f in p["photos"].iterdir()) / 1e9
    report = {
        "observations": len(obs),
        "observations_without_photos": int((obs["n_photos"] == 0).sum()),
        "observations_without_coords": int(obs["latitude"].isna().sum()),
        "coordinates_obscured": int(obs["coordinates_obscured"].fillna(False).astype(bool).sum()),
        "captive": int(obs["captive"].fillna(False).astype(bool).sum()),
        "photos_expected": len(pho),
        "photos_downloaded": len(pho) - len(missing),
        "photos_missing": len(missing),
        "photos_size_gb": round(size_gb, 2),
    }
    for k, v in report.items():
        print(f"{k:32s} {v}")
    if len(missing):
        write_jsonl(missing, p["tables"] / "photos_missing.jsonl")
        print("Detalle en tables/photos_missing.jsonl (correr download_photos de nuevo)")
    return report


def ids_from_excel(path: str, sheet: str = "identification", url_col: str = "url") -> list[int]:
    """Extrae los observation_id desde la columna de URLs del Excel de Martin."""
    df = pd.read_excel(path, sheet_name=sheet)
    ids = df[url_col].dropna().astype(str).str.extract(r"(\d+)\s*$")[0].dropna()
    return sorted(set(ids.astype(int)))


