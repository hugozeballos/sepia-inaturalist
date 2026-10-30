"""
Rasgos por foto con un modelo de vision: huesos visibles, lado (dorsal/ventral), erosion
y (desde photo_traits_v2) calidad de la foto.

Paso aparte del triaje (que cuenta por registro), con prompt propio (prompts/photo_traits_vN.txt).
Salida: outputs/photo_traits_<proveedor>_<modelo>_<version>.jsonl (un archivo por version
de prompt), siempre UNA LINEA POR FOTO.

Dos modos, segun la version del prompt (PROMPT_UNIT):
- "photo"  (v1-v4): una consulta por foto; el modelo solo ve esa foto.
- "record" (v5-):   una consulta por registro con todas sus fotos, para que el lado de una
  foto dudosa se pueda decidir comparando con otra foto del mismo registro. El modelo
  responde por foto y se guarda una linea por foto, con call_id para unir las de una
  misma consulta (tokens y respuesta cruda son de la consulta completa).

El lado y la erosion solo se piden cuando la foto muestra EXACTAMENTE un hueso: con varios,
la letra del modelo no se puede emparejar con la del anotador.

Se evalua contra la anotacion manual (evaluate_photo_traits): solo fotos donde la persona
anoto un unico individuo. La anotacion es a ciegas: la interfaz no muestra estos resultados.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from .config import Config
from .io import paths, read_jsonl, append_jsonl, model_output_path, read_model_outputs
from .triage import (load_prompt, make_provider, RateLimited, FatalProviderError)

Side = Literal["dorsal", "ventral", "unclear", "not_applicable"]
Erosion = Literal["1", "2", "3", "unclear", "not_applicable"]
Quality = Literal["high", "medium", "low", "not_applicable"]   # desde photo_traits_v2


# Como se decidio el lado (desde photo_traits_v5). Permite medir el acierto de cada via.
SideBasis = Literal["direct", "lines_rule", "inferred_same_bone", "inferred_other_bone",
                    "unclear", "not_applicable"]


class PhotoTraits(BaseModel):
    n_cuttlebones_visible: int
    side: Side
    erosion: Erosion
    photo_quality: Quality = "not_applicable"
    notes: str


class PhotoTraitsInRecord(PhotoTraits):
    """Una foto dentro de la respuesta por registro (v5-)."""
    photo_index: int          # el N de "Photo N" (1 = primera foto del registro)
    side_basis: SideBasis


class RecordPhotoTraits(BaseModel):
    photos: list[PhotoTraitsInRecord]


# Unidad de consulta de cada version de prompt (la que no esta, es "photo").
PROMPT_UNIT = {"photo_traits_v5": "record", "photo_traits_v6": "record"}


def apply_rules(result: dict) -> tuple[dict, list[str]]:
    """
    Reglas fijas por foto:
    - lado, erosion y calidad solo con exactamente 1 hueso visible;
    - side_basis (si existe) coherente con el lado: lado 'unclear' -> base 'unclear';
      lado 'not_applicable' -> base 'not_applicable'.
    """
    r, applied = dict(result), []
    if r.get("n_cuttlebones_visible") != 1:
        keys = ("side", "erosion", "photo_quality") + (("side_basis",) if "side_basis" in r else ())
        for k in keys:
            if r.get(k) != "not_applicable":
                r[k] = "not_applicable"
                applied.append(f"n_visible!=1 -> {k}=not_applicable")
    if "side_basis" in r:
        for side in ("unclear", "not_applicable"):
            if r.get("side") == side and r.get("side_basis") != side:
                r["side_basis"] = side
                applied.append(f"side={side} -> side_basis={side}")
    return r, applied


def split_record_result(result: dict, n_photos: int) -> list[dict]:
    """
    Separa la respuesta por registro en una por foto, ordenada por photo_index.
    Error si no hay exactamente una entrada por foto (1..n_photos): la consulta se repite.
    """
    items = result.get("photos") or []
    idx = sorted(int(it["photo_index"]) for it in items)
    if idx != list(range(1, n_photos + 1)):
        raise ValueError(f"la respuesta no trae una entrada por foto: indices {idx}, "
                         f"se esperaban 1..{n_photos}")
    return sorted(items, key=lambda it: int(it["photo_index"]))


def _settings(cfg: Config, prompt_version: str | None = None) -> dict:
    """Parametros de la seccion photo_traits del YAML. prompt_version reemplaza al del YAML
    (para correr o evaluar otra version sin editar la configuracion)."""
    tc = {"provider": "gemini", "model": "gemini-3.5-flash-lite", "temperature": 0.0,
          "prompt_version": "photo_traits_v1", "image_size": 1024, "requests_per_minute": 8,
          "thinking_level": "low", "timeout_s": 120, **(cfg.photo_traits or {})}
    if prompt_version:
        tc["prompt_version"] = prompt_version
    tc["unit"] = PROMPT_UNIT.get(tc["prompt_version"], "photo")
    return tc


def output_path(cfg: Config, provider: str, model: str, prompt_version: str) -> Path:
    """Un archivo por version: outputs/photo_traits_<proveedor>_<modelo>_<version>.jsonl"""
    return model_output_path(cfg, "photo_traits", provider, model, prompt_version)


def done_photo_ids(df, prompt_sha: str) -> set[int]:
    """Fotos ya hechas sin error con el mismo texto de prompt (hash)."""
    if df.empty:
        return set()
    ok = df[df["error"].isna() & (df["prompt_sha"] == prompt_sha)]
    return set(ok["photo_id"].astype(int))


def _call_with_retries(provider, images: list[bytes], prompt: str, schema):
    """Llama al modelo; si hay limite de uso o saturacion, espera y reintenta (4 intentos)."""
    for attempt in range(4):
        try:
            return provider.classify(images, prompt, schema=schema)
        except RateLimited:
            if attempt == 3:
                raise
            wait = 30 * 2 ** attempt
            print(f"  limite de uso o modelo saturado; espero {wait} s", flush=True)
            time.sleep(wait)


def run_photo_traits(cfg: Config, provider=None, limit: int | None = None,
                     ids: list[int] | None = None, redo: bool = False,
                     prompt_version: str | None = None) -> dict:
    """
    Reanudable: se saltan las fotos ya hechas con este prompt (mismo hash).
    ids: IDs de registro (todas sus fotos). redo=True vuelve a hacerlas (agrega lineas).
    limit: N fotos (modo "photo") o N registros (modo "record").
    prompt_version: reemplaza la version del YAML.
    """
    tc = _settings(cfg, prompt_version)
    prompt, prompt_sha = load_prompt(tc["prompt_version"])
    provider = provider or make_provider(tc["provider"], tc["model"], tc["temperature"],
                                         thinking_level=tc["thinking_level"],
                                         timeout_s=tc["timeout_s"])
    p = paths(cfg)
    img_dir = p["base"] / "photos" / f"{tc['image_size']}px"
    out = output_path(cfg, tc["provider"], tc["model"], tc["prompt_version"])
    saved = read_model_outputs(cfg, "photo_traits", tc["provider"], tc["model"], tc["prompt_version"])
    pho = read_jsonl(p["tables"] / "photos.jsonl").sort_values(["observation_id", "position"])
    if ids is not None:
        pho = pho[pho["observation_id"].isin({int(i) for i in ids})]
    skip = set() if redo else done_photo_ids(saved, prompt_sha)

    # Unidades de consulta: una foto, o un registro con TODAS sus fotos.
    # En modo registro se repite el registro completo si le falta alguna foto.
    if tc["unit"] == "record":
        units = [g for _, g in pho.groupby("observation_id", sort=True)
                 if not set(g["photo_id"].astype(int)) <= skip]
    else:
        todo = pho[~pho["photo_id"].astype(int).isin(skip)]
        units = [todo.iloc[[i]] for i in range(len(todo))]
    if limit:
        units = units[:limit]
    print(f"Rasgos por foto con {tc['provider']}/{tc['model']} · prompt {tc['prompt_version']} "
          f"({prompt_sha}) · unidad {tc['unit']} · {tc['image_size']} px")
    print(("modo REHACER · " if redo else f"{len(skip)} fotos ya hechas · ")
          + f"{len(units)} {'registros' if tc['unit'] == 'record' else 'fotos'} por hacer")
    print(f"salida: {out}")

    schema = RecordPhotoTraits if tc["unit"] == "record" else PhotoTraits
    pause = 60.0 / tc["requests_per_minute"]
    stats = {"calls": 0, "ok": 0, "error": 0, "input_tokens": 0, "output_tokens": 0}
    for n, g in enumerate(units, start=1):
        oid = int(g["observation_id"].iloc[0])
        call_id = f"{oid}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        base = {"observation_id": oid, "provider": tc["provider"], "model": tc["model"],
                "temperature": tc["temperature"], "prompt_version": tc["prompt_version"],
                "prompt_sha": prompt_sha, "image_size": tc["image_size"],
                "thinking_level": tc["thinking_level"], "unit": tc["unit"],
                "call_id": call_id, "n_photos_in_call": len(g)}
        lines = []
        try:
            images = [(img_dir / (Path(f).stem + ".jpg")).read_bytes() for f in g["file_name"]]
            t0 = time.time()
            result, raw, usage = _call_with_retries(provider, images, prompt, schema)
            elapsed = round(time.time() - t0, 1)
            per_photo = (split_record_result(result, len(g)) if tc["unit"] == "record"
                         else [result])
            for r, res in zip(g.itertuples(index=False), per_photo):
                res, rules = apply_rules(res)
                lines.append({**base, "photo_id": int(r.photo_id), "position": int(r.position),
                              "result": res, "rules_applied": rules, "raw_response": raw,
                              **usage, "elapsed_s": elapsed, "error": None})
                print(f"{n}/{len(units)} · {oid} foto {r.position} · "
                      f"{res['n_cuttlebones_visible']} huesos · lado {res['side']}"
                      + (f" ({res['side_basis']})" if "side_basis" in res else "")
                      + f" · erosion {res['erosion']} · calidad {res.get('photo_quality')}", flush=True)
            stats["ok"] += len(lines)
            stats["input_tokens"] += usage.get("input_tokens") or 0
            stats["output_tokens"] += usage.get("output_tokens") or 0
        except RateLimited as e:
            print(f"Limite de uso persistente. Se detiene; reanudar mas tarde.\n  {e}")
            break
        except FatalProviderError as e:
            print(f"Error de configuracion: se detiene.\n  {e}")
            break
        except Exception as e:
            # una linea de error por foto: al reanudar se vuelve a intentar
            lines = [{**base, "photo_id": int(r.photo_id), "position": int(r.position),
                      "result": None, "raw_response": None,
                      "error": f"{type(e).__name__}: {e}"} for r in g.itertuples(index=False)]
            stats["error"] += len(lines)
            print(f"{n}/{len(units)} · {oid} · ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
        stats["calls"] += 1
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        for line in lines:
            line["saved_at"] = now
            append_jsonl(line, out)
        time.sleep(pause)
    print(f"Listo: {stats} · salida en {out}")
    return stats


def load_photo_traits(cfg: Config, prompt_version: str | None = None) -> pd.DataFrame:
    """Una fila por foto (vale la ultima), de una sola version de prompt
    (la del YAML, o prompt_version si se indica)."""
    tc = _settings(cfg, prompt_version)
    df = read_model_outputs(cfg, "photo_traits", tc["provider"], tc["model"], tc["prompt_version"])
    if df.empty:
        return df
    df = df[df["error"].isna()]
    if df.empty:
        return df
    df = df.sort_values("saved_at").drop_duplicates("photo_id", keep="last")
    flat = pd.json_normalize(df["result"].tolist())
    for c in ("observation_id", "photo_id", "position"):
        flat.insert(0, c, df[c].values)
    return flat


def evaluate_photo_traits(cfg: Config, prompt_version: str | None = None) -> pd.DataFrame:
    """
    Compara con la anotacion manual, foto por foto:
    - conteo: huesos visibles del modelo vs 'Individuals in photo' del anotador
    - lado y erosion: solo fotos con un unico individuo anotado y 1 hueso para el modelo
    Devuelve la tabla de comparacion e imprime los aciertos.
    """
    from .annotate import load_events
    model = load_photo_traits(cfg, prompt_version)
    print(f"Version: {_settings(cfg, prompt_version)['prompt_version']}")
    ev = load_events(cfg)
    if model.empty:
        print("Sin resultados del modelo todavia")
        return model
    info = ev["photo_info"][["photo_id", "n_in_photo"]].astype({"photo_id": "int64"})
    ind = ev["individual"]
    one = ind.groupby("photo_id").filter(lambda g: len(g) == 1)
    for c in ("side", "erosion", "photo_quality"):
        if c not in one:
            one[c] = None
        if c not in model:
            model[c] = "not_applicable"
    one = one[["photo_id", "side", "erosion", "photo_quality"]].rename(
        columns={c: f"{c}_human" for c in ("side", "erosion", "photo_quality")})
    t = (model.rename(columns={c: f"{c}_model" for c in ("side", "erosion", "photo_quality")})
         .merge(info, on="photo_id", how="left").merge(one, on="photo_id", how="left"))

    cnt = t[t["n_in_photo"].notna()]
    if len(cnt):
        ok = (cnt["n_cuttlebones_visible"] == cnt["n_in_photo"]).mean()
        print(f"Huesos visibles por foto: {ok:.0%} iguales ({len(cnt)} fotos anotadas)")
    for var in ("side", "erosion", "photo_quality"):
        h = t[f"{var}_human"]
        m = t[f"{var}_model"].astype(str)
        sub = t[h.notna() & (m != "not_applicable")].copy()
        if not len(sub):
            print(f"{var}: aun no hay fotos comparables")
            continue
        sub["h"] = sub[f"{var}_human"].map(          # erosion humana viene como numero
            lambda v: str(int(v)) if isinstance(v, (int, float)) else str(v))
        sub["m"] = sub[f"{var}_model"].astype(str)
        print(f"{var}: {(sub['h'] == sub['m']).mean():.0%} de acuerdo ({len(sub)} fotos con 1 hueso)")
        print(pd.crosstab(sub["h"], sub["m"], rownames=["human"], colnames=["model"]))
        if var == "side" and "side_basis" in sub:
            # acierto por via de decision (v5-): direct, lines_rule, inferred_*
            sub["ok"] = sub["h"] == sub["m"]
            by = sub.groupby("side_basis")["ok"].agg(["mean", "size"])
            by.columns = ["agreement", "n_photos"]
            print("lado segun como se decidio:")
            print(by.round(2))
    return t
