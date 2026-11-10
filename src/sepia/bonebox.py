"""
Recuadro del hueso por foto con un modelo de vision (paso aparte, como bitemarks).

Solo pide el recuadro: nada de lado, erosion ni calidad (eso lo dan los prompts photo_traits).
Una consulta por foto (1024 px), solo en fotos con exactamente 1 hueso segun una version de
photo_traits (select_from, por defecto photo_traits_v14). Con mas de un hueso el recuadro se
dibuja a mano en AnnotationApp.

El modelo responde box_2d = [ymin, xmin, ymax, xmax] en 0-1000 (formato habitual de Gemini).
El codigo lo pasa a crop_box {x0, y0, x1, y1} en 0-1 (igual que los recuadros de la anotacion,
sobre la foto enderezada), asi sirve para cualquier tamano de foto. La respuesta cruda queda
en raw_response.

Archivo propio: outputs/bonebox_<proveedor>_<modelo>_<version>.jsonl (una linea por foto,
vale la ultima). AnnotationApp lo usa para partir con el recuadro ya dibujado.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from .config import Config
from .io import paths, read_jsonl, append_jsonl, model_output_path, read_model_outputs
from .triage import load_prompt, make_provider, RateLimited, FatalProviderError
from .photo_traits import _call_with_retries

KIND = "bonebox"
MIN_SIDE = 0.01       # recuadros mas chicos que 1 % del lado se descartan (respuesta invalida)


class BoneBoxResult(BaseModel):
    """Respuesta del modelo para una foto."""
    one_cuttlebone: bool
    box_2d: list[int]
    notes: str


SCHEMA_BY_VERSION = {"bonebox_v1": BoneBoxResult}


def box2d_to_crop(box_2d) -> dict | None:
    """
    [ymin, xmin, ymax, xmax] en 0-1000 -> {x0, y0, x1, y1} en 0-1, ordenado y recortado a la foto.
    None si no son 4 numeros o si el recuadro es demasiado chico (p. ej. [0, 0, 0, 0]).
    """
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(v) / 1000 for v in box_2d)
    except (TypeError, ValueError):
        return None
    clip = lambda v: min(1.0, max(0.0, v))
    x0, x1 = sorted((clip(xmin), clip(xmax)))
    y0, y1 = sorted((clip(ymin), clip(ymax)))
    if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
        return None
    return {"x0": round(x0, 4), "y0": round(y0, 4), "x1": round(x1, 4), "y1": round(y1, 4)}


def apply_rules(result: dict) -> tuple[dict, list[str]]:
    """Agrega crop_box (0-1). Sin 1 hueso o con recuadro invalido: crop_box = None."""
    r, applied = dict(result), []
    box = box2d_to_crop(r.get("box_2d")) if r.get("one_cuttlebone") else None
    if r.get("one_cuttlebone") and box is None:
        applied.append("invalid box_2d -> crop_box=None")
    r["crop_box"] = box
    return r, applied


def _settings(cfg: Config, prompt_version: str | None = None) -> dict:
    """Seccion 'bonebox' del YAML; si no existe, proveedor/modelo de 'photo_traits'."""
    base = {k: v for k, v in (cfg.photo_traits or {}).items() if k != "prompt_version"}
    tc = {"provider": "gemini", "model": "gemini-3.5-flash-lite", "temperature": 0.0,
          "image_size": 1024, "requests_per_minute": 8, "thinking_level": "low",
          "timeout_s": 120, **base, "prompt_version": "bonebox_v1",
          "select_from": "photo_traits_v14", **(cfg.bonebox or {})}
    if prompt_version:
        tc["prompt_version"] = prompt_version
    return tc


def output_path(cfg: Config, provider: str, model: str, prompt_version: str) -> Path:
    """outputs/bonebox_<proveedor>_<modelo>_<version>.jsonl."""
    return model_output_path(cfg, KIND, provider, model, prompt_version)


def one_bone_photos(cfg: Config, select_from: str) -> list[int]:
    """photo_id con exactamente 1 hueso visible segun esa version de photo_traits."""
    from .photo_traits import load_photo_traits
    pt = load_photo_traits(cfg, select_from)
    if pt.empty:
        raise RuntimeError(f"No hay resultados de {select_from}: correr photo_traits primero")
    return sorted(pt.loc[pt["n_cuttlebones_visible"] == 1, "photo_id"].astype(int).unique())


def run_bonebox(cfg: Config, provider=None, limit: int | None = None,
                ids: list[int] | None = None, redo: bool = False,
                prompt_version: str | None = None, photo_ids=None) -> dict:
    """
    Una consulta por foto. Reanudable: se saltan las fotos ya hechas con este prompt (hash).
    Por defecto, solo fotos con 1 hueso segun select_from. photo_ids: lista propia de fotos.
    """
    tc = _settings(cfg, prompt_version)
    prompt, prompt_sha = load_prompt(tc["prompt_version"])
    provider = provider or make_provider(tc["provider"], tc["model"], tc["temperature"],
                                         thinking_level=tc["thinking_level"],
                                         timeout_s=tc["timeout_s"])
    schema = SCHEMA_BY_VERSION[tc["prompt_version"]]
    p = paths(cfg)
    img_dir = p["base"] / "photos" / f"{tc['image_size']}px"
    out = output_path(cfg, tc["provider"], tc["model"], tc["prompt_version"])
    saved = read_model_outputs(cfg, KIND, tc["provider"], tc["model"], tc["prompt_version"])
    done = set() if (redo or saved.empty) else set(
        saved.loc[saved["error"].isna() & (saved["prompt_sha"] == prompt_sha), "photo_id"].astype(int))
    pho = read_jsonl(p["tables"] / "photos.jsonl").sort_values(["observation_id", "position"])
    wanted = photo_ids if photo_ids is not None else one_bone_photos(cfg, tc["select_from"])
    pho = pho[pho["photo_id"].astype(int).isin({int(i) for i in wanted})]
    if ids is not None:
        pho = pho[pho["observation_id"].isin({int(i) for i in ids})]
    todo = pho[~pho["photo_id"].astype(int).isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"Recuadros con {tc['provider']}/{tc['model']} · prompt {tc['prompt_version']} "
          f"({prompt_sha}) · fotos con 1 hueso segun {tc['select_from']}: {len(pho)} · "
          f"{len(done)} hechas · {len(todo)} por hacer")
    print(f"salida: {out}")

    pause = 60.0 / tc["requests_per_minute"]
    stats = {"calls": 0, "ok": 0, "no_box": 0, "error": 0, "input_tokens": 0, "output_tokens": 0}
    for n, r in enumerate(todo.itertuples(index=False), start=1):
        base = {"observation_id": int(r.observation_id), "photo_id": int(r.photo_id),
                "position": int(r.position), "provider": tc["provider"], "model": tc["model"],
                "temperature": tc["temperature"], "prompt_version": tc["prompt_version"],
                "prompt_sha": prompt_sha, "image_size": tc["image_size"],
                "thinking_level": tc["thinking_level"], "selected_by": tc["select_from"]}
        try:
            img = (img_dir / (Path(r.file_name).stem + ".jpg")).read_bytes()
            t0 = time.time()
            result, raw, usage = _call_with_retries(provider, [img], prompt, schema)
            res, rules = apply_rules(result)
            line = {**base, "result": res, "rules_applied": rules, "raw_response": raw, **usage,
                    "elapsed_s": round(time.time() - t0, 1), "error": None}
            stats["ok"] += 1
            stats["no_box"] += res["crop_box"] is None
            stats["input_tokens"] += usage.get("input_tokens") or 0
            stats["output_tokens"] += usage.get("output_tokens") or 0
            print(f"{n}/{len(todo)} · {r.observation_id} foto {r.position} · "
                  f"{'recuadro ' + str(res['crop_box']) if res['crop_box'] else 'SIN recuadro'}",
                  flush=True)
        except RateLimited as e:
            print(f"Limite de uso persistente. Se detiene; reanudar mas tarde.\n  {e}")
            break
        except FatalProviderError as e:
            print(f"Error de configuracion: se detiene.\n  {e}")
            break
        except Exception as e:
            line = {**base, "result": None, "raw_response": None,
                    "error": f"{type(e).__name__}: {e}"}
            stats["error"] += 1
            print(f"{n}/{len(todo)} · {r.observation_id} · ERROR {type(e).__name__}: {str(e)[:200]}")
        stats["calls"] += 1
        line["saved_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        append_jsonl(line, out)
        time.sleep(pause)
    print(f"Listo: {stats} · salida en {out}")
    return stats


def load_bonebox(cfg: Config, prompt_version: str | None = None) -> pd.DataFrame:
    """Una fila por foto (vale la ultima sin error): photo_id, crop_box, one_cuttlebone, notes."""
    tc = _settings(cfg, prompt_version)
    df = read_model_outputs(cfg, KIND, tc["provider"], tc["model"], tc["prompt_version"])
    if df.empty:
        return df
    df = df[df["error"].isna()]
    if df.empty:
        return df
    df = df.sort_values("saved_at").drop_duplicates("photo_id", keep="last")
    flat = pd.json_normalize(df["result"].tolist(), max_level=0)
    for c in ("observation_id", "photo_id", "position"):
        flat.insert(0, c, df[c].values)
    flat["prompt_version"] = tc["prompt_version"]
    flat["photo_id"] = flat["photo_id"].astype("int64")
    return flat


def model_boxes(cfg: Config, prompt_version: str | None = None) -> dict[int, dict]:
    """photo_id -> crop_box del modelo (solo fotos con recuadro valido)."""
    df = load_bonebox(cfg, prompt_version)
    if df.empty:
        return {}
    df = df[df["crop_box"].apply(lambda b: isinstance(b, dict))]
    return dict(zip(df["photo_id"].astype(int), df["crop_box"]))
