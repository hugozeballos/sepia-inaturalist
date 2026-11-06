"""
Marcas de picoteo por foto con un modelo de vision (zero-shot, sin ejemplos).

Paso aparte de photo_traits (prompt propio, prompts/bitemarks_vN.txt; archivo propio
outputs/bitemarks_<proveedor>_<modelo>_<version>.jsonl), una consulta por foto, solo con
exactamente 1 hueso visible.

Para bajar los falsos positivos, el modelo no responde solo si/no: lista cada marca candidata
y contesta para cada una si podria ser arena/suciedad, un borde roto, una grieta, un hoyo de
erosion o una sombra, y que forma y bordes tiene. Con esas respuestas un script fijo decide
(bitemarks_rule, ver rule_from_candidates). Se guarda tambien la respuesta del modelo
(bitemarks) para comparar las dos.

Se evalua contra la anotacion manual (evaluate_bitemarks): fotos con un unico individuo
anotado; 'yes' y '(yes)' del anotador cuentan como positivo, 'no' como negativo.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from .config import Config
from .io import (paths, read_jsonl, append_jsonl, model_output_path, read_model_outputs,
                 save_metrics, ids_sha)
from .triage import load_prompt, make_provider, RateLimited, FatalProviderError
from .photo_traits import _call_with_retries

KIND = "bitemarks"

Shape = Literal["triangular_notch", "beak_side_arc", "round_hole", "linear", "irregular"]
BITE_SHAPES = {"triangular_notch", "beak_side_arc"}
# Las cinco preguntas que descartan falsos positivos
CHECKS = ("is_sand_or_debris", "is_broken_edge", "is_crack", "is_erosion_pit",
          "is_shadow_or_reflection")


class Candidate(BaseModel):
    """Una posible marca de picoteo descrita por el modelo, con sus preguntas de descarte."""
    location: Literal["edge", "surface"]
    shape: Shape
    edges: Literal["sharp", "worn"]
    is_sand_or_debris: bool
    is_broken_edge: bool
    is_crack: bool
    is_erosion_pit: bool
    is_shadow_or_reflection: bool
    description: str


class BitemarksResult(BaseModel):
    """Respuesta del modelo para una foto: hueso evaluable y candidatas."""
    n_cuttlebones_visible: int
    assessable: Literal["yes", "no", "not_applicable"]
    candidates: list[Candidate]
    bitemarks: Literal["yes", "probable", "no", "not_assessable", "not_applicable"]
    notes: str


SCHEMA_BY_VERSION = {"bitemarks_v1": BitemarksResult}


def rule_from_candidates(candidates, n_visible, assessable) -> str:
    """
    Decision fija con las respuestas del modelo:
    - no_applicable si no hay exactamente 1 hueso; not_assessable si assessable = 'no';
    - yes: alguna candidata con forma de picoteo, bordes 'sharp' y las 5 preguntas en false;
    - probable: ninguna 'yes', pero alguna con forma de picoteo y las 5 preguntas en false
      (bordes 'worn');
    - no: en cualquier otro caso.
    """
    if n_visible != 1:
        return "not_applicable"
    if assessable == "no":
        return "not_assessable"
    clean = [c for c in (candidates or [])
             if c.get("shape") in BITE_SHAPES and not any(c.get(k) for k in CHECKS)]
    if any(c.get("edges") == "sharp" for c in clean):
        return "yes"
    if clean:
        return "probable"
    return "no"


def apply_rules(result: dict) -> tuple[dict, list[str]]:
    """Con != 1 hueso todo queda en not_applicable; agrega bitemarks_rule y n_candidates."""
    r, applied = dict(result), []
    if r.get("n_cuttlebones_visible") != 1:
        for k in ("assessable", "bitemarks"):
            if r.get(k) != "not_applicable":
                r[k] = "not_applicable"
                applied.append(f"n_visible!=1 -> {k}=not_applicable")
        if r.get("candidates"):
            r["candidates"] = []
            applied.append("n_visible!=1 -> candidates=[]")
    r["bitemarks_rule"] = rule_from_candidates(r.get("candidates"), r.get("n_cuttlebones_visible"),
                                               r.get("assessable"))
    r["n_candidates"] = len(r.get("candidates") or [])
    return r, applied


def _settings(cfg: Config, prompt_version: str | None = None) -> dict:
    """Seccion 'bitemarks' del YAML; si no existe, proveedor/modelo de 'photo_traits'."""
    base = {k: v for k, v in (cfg.photo_traits or {}).items() if k != "prompt_version"}
    tc = {"provider": "gemini", "model": "gemini-3.5-flash-lite", "temperature": 0.0,
          "image_size": 1024, "requests_per_minute": 8, "thinking_level": "low",
          "timeout_s": 120, **base, "prompt_version": "bitemarks_v1", **(cfg.bitemarks or {})}
    if prompt_version:
        tc["prompt_version"] = prompt_version
    return tc


def output_path(cfg: Config, provider: str, model: str, prompt_version: str) -> Path:
    """Archivo de salida de esta version: outputs/bitemarks_<proveedor>_<modelo>_<version>.jsonl."""
    return model_output_path(cfg, KIND, provider, model, prompt_version)


def run_bitemarks(cfg: Config, provider=None, limit: int | None = None,
                  ids: list[int] | None = None, redo: bool = False,
                  prompt_version: str | None = None, photo_ids=None) -> dict:
    """
    Una consulta por foto. Reanudable: se saltan las fotos ya hechas con este prompt (hash).
    ids: solo fotos de estos registros. photo_ids: solo estas fotos (p. ej. las de 1 hueso).
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
    if ids is not None:
        pho = pho[pho["observation_id"].isin({int(i) for i in ids})]
    if photo_ids is not None:
        pho = pho[pho["photo_id"].astype(int).isin({int(i) for i in photo_ids})]
    todo = pho[~pho["photo_id"].astype(int).isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"Marcas de picoteo con {tc['provider']}/{tc['model']} · prompt {tc['prompt_version']} "
          f"({prompt_sha}) · {tc['image_size']} px · {len(done)} hechas · {len(todo)} por hacer")
    print(f"salida: {out}")

    pause = 60.0 / tc["requests_per_minute"]
    stats = {"calls": 0, "ok": 0, "error": 0, "input_tokens": 0, "output_tokens": 0}
    for n, r in enumerate(todo.itertuples(index=False), start=1):
        base = {"observation_id": int(r.observation_id), "photo_id": int(r.photo_id),
                "position": int(r.position), "provider": tc["provider"], "model": tc["model"],
                "temperature": tc["temperature"], "prompt_version": tc["prompt_version"],
                "prompt_sha": prompt_sha, "image_size": tc["image_size"],
                "thinking_level": tc["thinking_level"]}
        try:
            img = (img_dir / (Path(r.file_name).stem + ".jpg")).read_bytes()
            t0 = time.time()
            result, raw, usage = _call_with_retries(provider, [img], prompt, schema)
            res, rules = apply_rules(result)
            line = {**base, "result": res, "rules_applied": rules, "raw_response": raw, **usage,
                    "elapsed_s": round(time.time() - t0, 1), "error": None}
            stats["ok"] += 1
            stats["input_tokens"] += usage.get("input_tokens") or 0
            stats["output_tokens"] += usage.get("output_tokens") or 0
            print(f"{n}/{len(todo)} · {r.observation_id} foto {r.position} · "
                  f"{res['n_cuttlebones_visible']} huesos · modelo {res['bitemarks']} · "
                  f"regla {res['bitemarks_rule']} · {res['n_candidates']} candidatas", flush=True)
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


def load_bitemarks(cfg: Config, prompt_version: str | None = None) -> pd.DataFrame:
    """Una fila por foto (vale la ultima sin error), de una version de prompt."""
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
    flat["prompt_sha"] = df["prompt_sha"].values
    flat["photo_id"] = flat["photo_id"].astype("int64")
    return flat


def _binary(v):
    """Positivo / negativo / None (sin decision) para comparar."""
    if v in ("yes", "(yes)", "probable"):
        return True
    if v == "no":
        return False
    return None


def evaluate_bitemarks(cfg: Config, prompt_version: str | None = None, save: bool = True,
                       quiet: bool = False) -> pd.DataFrame:
    """
    Compara con la anotacion manual en fotos con un unico individuo anotado y 1 hueso para el
    modelo. Positivo = yes o (yes) del anotador; yes o probable del modelo/regla.
    Metricas para la respuesta del modelo (bitemarks) y la regla fija (bitemarks_rule), en
    todas las fotos y separadas por el lado anotado (el picoteo se ve sobre todo en el ventral).
    """
    from .annotate import load_events
    tc = _settings(cfg, prompt_version)
    say = (lambda *a: None) if quiet else print
    model = load_bitemarks(cfg, prompt_version)
    say(f"Version: {tc['prompt_version']}")
    if model.empty:
        say("Sin resultados del modelo todavia")
        return model
    ind = load_events(cfg)["individual"]
    one = ind.groupby("photo_id").filter(lambda g: len(g) == 1)
    for c in ("bitemarks", "side"):
        if c not in one:
            one[c] = None
    one = one[["photo_id", "bitemarks", "side"]].rename(
        columns={"bitemarks": "bitemarks_human", "side": "side_human"})
    one["photo_id"] = one["photo_id"].astype("int64")
    t = model.merge(one, on="photo_id", how="inner")
    t = t[t["n_cuttlebones_visible"] == 1]

    m = {"n_photos_compared": len(t)}
    for col in ("bitemarks", "bitemarks_rule"):
        for scope, sub in (("all", t), ("ventral", t[t["side_human"] == "ventral"]),
                           ("dorsal", t[t["side_human"] == "dorsal"])):
            pairs = [(_binary(a), _binary(b)) for a, b in zip(sub["bitemarks_human"], sub[col])]
            both = [(a, b) for a, b in pairs if a is not None and b is not None]
            tp = sum(a and b for a, b in both)
            fp = sum((not a) and b for a, b in both)
            fn = sum(a and (not b) for a, b in both)
            tn = sum((not a) and (not b) for a, b in both)
            n_ok = len(both)
            key = f"{col}_{scope}"
            m[key] = {"n": n_ok, "abstained": sum(a is not None and b is None for a, b in pairs),
                      "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "precision": tp / (tp + fp) if tp + fp else None,
                      "recall": tp / (tp + fn) if tp + fn else None,
                      "agreement": (tp + tn) / n_ok if n_ok else None}
            say(f"{key}: {m[key]}")
        ct = pd.crosstab(t["bitemarks_human"].fillna("—"), t[col].fillna("—"),
                         rownames=["human"], colnames=[col])
        m[f"{col}_confusion"] = {r: {c: int(ct.loc[r, c]) for c in ct.columns} for r in ct.index}
        say(ct)
    t.attrs["metrics"] = m
    if save:
        path = save_metrics(cfg, KIND, {
            "provider": tc["provider"], "model": tc["model"],
            "prompt_version": tc["prompt_version"],
            "prompt_sha": sorted(set(t["prompt_sha"].dropna())),
            "image_size": tc["image_size"], "eval_set": "all",
            "eval_set_sha": ids_sha(t["photo_id"]), "n_photos_compared": len(t), "metrics": m})
        say(f"Metricas guardadas en {path}")
    return t
