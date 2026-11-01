"""Carpetas, JSON Lines, logs y manifest."""
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

import pandas as pd
from dataclasses import asdict

from .config import Config, API_URL


def paths(cfg: Config) -> dict[str, Path]:
    """Estructura de carpetas dentro de base_dir (se crea si no existe)."""
    base = Path(cfg.base_dir)
    p = {
        "base": base,
        "raw": base / "raw" / "api_pages",       # copia cruda, una pagina por archivo
        "photos": base / "photos" / cfg.photo_size,
        "tables": base / "tables",               # observations.jsonl, photos.jsonl
        "annotations": base / "annotations",     # una .jsonl por anotador (solo se agrega)
        "exports": base / "exports",             # Excel generados con build_excel()
        "logs": base / "logs",
    }
    for d in p.values():
        d.mkdir(parents=True, exist_ok=True)
    return p


def _log(p: dict, name: str, record: dict) -> None:
    """Agrega una linea a un log JSONL."""
    record = {"time": dt.datetime.now(dt.timezone.utc).isoformat(), **record}
    with open(p["logs"] / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clean(v):
    """Convierte NaN y tipos de numpy/pandas a valores JSON normales."""
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    if hasattr(v, "item"):                # numpy int/float/bool
        return _clean(v.item())
    return v


def _deep_clean(v):
    """_clean recursivo, para dicts y listas anidadas (metricas)."""
    if isinstance(v, dict):
        return {str(k): _deep_clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep_clean(x) for x in v]
    return _clean(v)


def write_jsonl(records, path: Path) -> None:
    """
    Escribe una lista de dicts (o un DataFrame) como JSON Lines.
    Escribe a un archivo temporal y lo renombra: nunca queda un archivo a medias.
    """
    if isinstance(records, pd.DataFrame):
        records = records.to_dict(orient="records")
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: _clean(v) for k, v in r.items()}, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(record: dict, path: Path) -> None:
    """Agrega una linea al final (para anotaciones y salidas de modelos)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: _clean(v) for k, v in record.items()}, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> pd.DataFrame:
    """Lee un .jsonl como DataFrame (vacio si el archivo no existe)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


# --------------------------------------------------------------------------
# Salidas de modelos: un archivo por paso, proveedor, modelo y VERSION DE PROMPT
# --------------------------------------------------------------------------

def model_output_path(cfg: Config, kind: str, provider: str, model: str,
                      prompt_version: str) -> Path:
    """
    outputs/<kind>_<proveedor>_<modelo>_<version>.jsonl
    (p. ej. photo_traits_gemini_gemini-3.5-flash-lite_photo_traits_v5.jsonl).
    Cada version de prompt escribe en su propio archivo.
    """
    out = Path(cfg.base_dir) / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{kind}_{provider}_{model.replace('/', '-')}_{prompt_version}.jsonl"


def legacy_output_path(cfg: Config, kind: str, provider: str, model: str) -> Path:
    """Archivo antiguo (antes del 30 oct 2026): todas las versiones de prompt juntas."""
    return Path(cfg.base_dir) / "outputs" / f"{kind}_{provider}_{model.replace('/', '-')}.jsonl"


def read_model_outputs(cfg: Config, kind: str, provider: str, model: str,
                       prompt_version: str) -> pd.DataFrame:
    """
    Todas las lineas guardadas para esta version de prompt: las del archivo propio
    de la version y, si todavia existe, las del archivo antiguo con esa version.
    (scripts/split_outputs.py separa el archivo antiguo; despues ya no se usa.)
    """
    new = read_jsonl(model_output_path(cfg, kind, provider, model, prompt_version))
    old = read_jsonl(legacy_output_path(cfg, kind, provider, model))
    if not old.empty and "prompt_version" in old:
        old = old[old["prompt_version"] == prompt_version]
    else:
        old = pd.DataFrame()
    parts = [d for d in (old, new) if not d.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def split_legacy_outputs(cfg: Config, kind: str, provider: str, model: str) -> dict[str, int]:
    """
    Separa el archivo antiguo en un archivo por version de prompt.
    Las lineas se AGREGAN al archivo de cada version (no se borra nada) y el archivo
    antiguo se renombra a *.legacy (se conserva, pero ya no se lee: asi no se duplica).
    Devuelve {version: lineas copiadas}.
    """
    src = legacy_output_path(cfg, kind, provider, model)
    if not src.exists():
        return {}
    counts: dict[str, int] = {}
    with open(src, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    for rec in lines:
        v = rec.get("prompt_version") or "unknown_version"
        append_jsonl(rec, model_output_path(cfg, kind, provider, model, v))
        counts[v] = counts.get(v, 0) + 1
    src.rename(src.with_suffix(".jsonl.legacy"))
    return counts


def write_manifest(cfg: Config, extra: dict | None = None) -> None:
    """Guarda la configuracion y la fecha de descarga (para el parrafo de metodos)."""
    p = paths(cfg)
    m = {"config": asdict(cfg), "query_params": cfg.query_params(),
         "api_url": API_URL,
         "written_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if extra:
        m.update(extra)
    (p["base"] / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))




# --------------------------------------------------------------------------
# Metricas de evaluacion: una linea por evaluacion, solo se agregan
# --------------------------------------------------------------------------

def metrics_path(cfg: Config, kind: str) -> Path:
    """outputs/metrics/<kind>_metrics.jsonl (p. ej. photo_traits_metrics.jsonl)."""
    out = Path(cfg.base_dir) / "outputs" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{kind}_metrics.jsonl"


def ids_sha(ids) -> str:
    """Huella corta de un conjunto de IDs (fotos o registros evaluados).
    Dos evaluaciones con la misma huella usaron exactamente el mismo conjunto."""
    import hashlib
    text = ",".join(str(int(i)) for i in sorted(set(ids)))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def git_commit() -> str | None:
    """Commit del codigo que evalua (None si no se puede saber)."""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:
        return None


def save_metrics(cfg: Config, kind: str, record: dict) -> Path:
    """Agrega una linea de metricas (con fecha y commit) a outputs/metrics/<kind>_metrics.jsonl."""
    rec = {"evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "git_commit": git_commit(), "kind": kind, **record}
    rec = _deep_clean(rec)
    path = metrics_path(cfg, kind)
    append_jsonl(rec, path)
    return path


def load_metrics(cfg: Config, kind: str) -> pd.DataFrame:
    """Historial de metricas de un paso (una fila por evaluacion)."""
    return read_jsonl(metrics_path(cfg, kind))
