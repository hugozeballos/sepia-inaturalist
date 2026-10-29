"""
Rasgos por foto con un modelo de vision: huesos visibles, lado (dorsal/ventral) y erosion.

Paso aparte del triaje (que cuenta por registro): una consulta por foto, prompt propio
(prompts/photo_traits_v1.txt), salida en outputs/photo_traits_<proveedor>_<modelo>.jsonl.
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
from .io import paths, read_jsonl, append_jsonl
from .triage import (load_prompt, make_provider, RateLimited, FatalProviderError)

Side = Literal["dorsal", "ventral", "unclear", "not_applicable"]
Erosion = Literal["1", "2", "3", "unclear", "not_applicable"]


class PhotoTraits(BaseModel):
    n_cuttlebones_visible: int
    side: Side
    erosion: Erosion
    notes: str


def apply_rules(result: dict) -> tuple[dict, list[str]]:
    """Regla fija: lado y erosion solo con exactamente 1 hueso visible."""
    r, applied = dict(result), []
    if r.get("n_cuttlebones_visible") != 1:
        for k in ("side", "erosion"):
            if r.get(k) != "not_applicable":
                r[k] = "not_applicable"
                applied.append(f"n_visible!=1 -> {k}=not_applicable")
    return r, applied


def _settings(cfg: Config) -> dict:
    return {"provider": "gemini", "model": "gemini-3.5-flash-lite", "temperature": 0.0,
            "prompt_version": "photo_traits_v1", "image_size": 1024, "requests_per_minute": 8,
            "thinking_level": "low", "timeout_s": 120, **(cfg.photo_traits or {})}


def output_path(cfg: Config, provider: str, model: str) -> Path:
    out = paths(cfg)["base"] / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"photo_traits_{provider}_{model.replace('/', '-')}.jsonl"


def done_photo_ids(path: Path, prompt_sha: str) -> set[int]:
    df = read_jsonl(path)
    if df.empty:
        return set()
    ok = df[df["error"].isna() & (df["prompt_sha"] == prompt_sha)]
    return set(ok["photo_id"].astype(int))


def run_photo_traits(cfg: Config, provider=None, limit: int | None = None,
                     ids: list[int] | None = None, redo: bool = False) -> dict:
    """
    Una consulta por foto. Reanudable: se saltan las fotos ya hechas con este prompt.
    ids: IDs de registro (todas sus fotos). redo=True vuelve a hacerlas (agrega lineas).
    """
    tc = _settings(cfg)
    prompt, prompt_sha = load_prompt(tc["prompt_version"])
    provider = provider or make_provider(tc["provider"], tc["model"], tc["temperature"],
                                         thinking_level=tc["thinking_level"],
                                         timeout_s=tc["timeout_s"])
    p = paths(cfg)
    img_dir = p["base"] / "photos" / f"{tc['image_size']}px"
    out = output_path(cfg, tc["provider"], tc["model"])
    pho = read_jsonl(p["tables"] / "photos.jsonl").sort_values(["observation_id", "position"])
    if ids is not None:
        pho = pho[pho["observation_id"].isin({int(i) for i in ids})]
    skip = set() if redo else done_photo_ids(out, prompt_sha)
    todo = pho[~pho["photo_id"].astype(int).isin(skip)]
    if limit:
        todo = todo.head(limit)
    print(f"Rasgos por foto con {tc['provider']}/{tc['model']} · prompt {tc['prompt_version']} "
          f"({prompt_sha}) · {tc['image_size']} px")
    print(("modo REHACER · " if redo else f"{len(skip)} fotos ya hechas · ") + f"{len(todo)} por hacer")

    pause = 60.0 / tc["requests_per_minute"]
    stats = {"ok": 0, "error": 0, "input_tokens": 0, "output_tokens": 0}
    for n, r in enumerate(todo.itertuples(index=False), start=1):
        record = {"observation_id": int(r.observation_id), "photo_id": int(r.photo_id),
                  "position": int(r.position), "provider": tc["provider"], "model": tc["model"],
                  "temperature": tc["temperature"], "prompt_version": tc["prompt_version"],
                  "prompt_sha": prompt_sha, "image_size": tc["image_size"],
                  "thinking_level": tc["thinking_level"]}
        try:
            img = (img_dir / (Path(r.file_name).stem + ".jpg")).read_bytes()
            t0 = time.time()
            for attempt in range(4):
                try:
                    result, raw, usage = provider.classify([img], prompt, schema=PhotoTraits)
                    break
                except RateLimited:
                    if attempt == 3:
                        raise
                    wait = 30 * 2 ** attempt
                    print(f"  limite de uso o modelo saturado; espero {wait} s", flush=True)
                    time.sleep(wait)
            result, rules = apply_rules(result)
            record.update(result=result, rules_applied=rules, raw_response=raw, **usage,
                          elapsed_s=round(time.time() - t0, 1), error=None)
            stats["ok"] += 1
            stats["input_tokens"] += usage.get("input_tokens") or 0
            stats["output_tokens"] += usage.get("output_tokens") or 0
            print(f"{n}/{len(todo)} · {r.observation_id} foto {r.position} · "
                  f"{result['n_cuttlebones_visible']} huesos · lado {result['side']} · "
                  f"erosion {result['erosion']}", flush=True)
        except RateLimited as e:
            print(f"Limite de uso persistente. Se detiene; reanudar mas tarde.\n  {e}")
            break
        except FatalProviderError as e:
            print(f"Error de configuracion: se detiene.\n  {e}")
            break
        except Exception as e:
            record.update(result=None, raw_response=None, error=f"{type(e).__name__}: {e}")
            stats["error"] += 1
            print(f"{n}/{len(todo)} · {r.observation_id} foto {r.position} · ERROR {e}", flush=True)
        record["saved_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        append_jsonl(record, out)
        time.sleep(pause)
    print(f"Listo: {stats} · salida en {out}")
    return stats


def load_photo_traits(cfg: Config) -> pd.DataFrame:
    """Una fila por foto (vale la ultima), solo la version de prompt de la config."""
    tc = _settings(cfg)
    df = read_jsonl(output_path(cfg, tc["provider"], tc["model"]))
    if df.empty:
        return df
    df = df[df["error"].isna() & (df["prompt_version"] == tc["prompt_version"])]
    if df.empty:
        return df
    df = df.sort_values("saved_at").drop_duplicates("photo_id", keep="last")
    flat = pd.json_normalize(df["result"].tolist())
    for c in ("observation_id", "photo_id", "position"):
        flat.insert(0, c, df[c].values)
    return flat


def evaluate_photo_traits(cfg: Config) -> pd.DataFrame:
    """
    Compara con la anotacion manual, foto por foto:
    - conteo: huesos visibles del modelo vs 'Individuals in photo' del anotador
    - lado y erosion: solo fotos con un unico individuo anotado y 1 hueso para el modelo
    Devuelve la tabla de comparacion e imprime los aciertos.
    """
    from .annotate import load_events
    model = load_photo_traits(cfg)
    ev = load_events(cfg)
    if model.empty:
        print("Sin resultados del modelo todavia")
        return model
    info = ev["photo_info"][["photo_id", "n_in_photo"]].astype({"photo_id": "int64"})
    ind = ev["individual"]
    one = ind.groupby("photo_id").filter(lambda g: len(g) == 1)
    one = one[["photo_id", "side", "erosion"]].rename(columns={"side": "side_human",
                                                                 "erosion": "erosion_human"})
    t = (model.rename(columns={"side": "side_model", "erosion": "erosion_model"})
         .merge(info, on="photo_id", how="left").merge(one, on="photo_id", how="left"))

    cnt = t[t["n_in_photo"].notna()]
    if len(cnt):
        ok = (cnt["n_cuttlebones_visible"] == cnt["n_in_photo"]).mean()
        print(f"Huesos visibles por foto: {ok:.0%} iguales ({len(cnt)} fotos anotadas)")
    for var in ("side", "erosion"):
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
    return t
