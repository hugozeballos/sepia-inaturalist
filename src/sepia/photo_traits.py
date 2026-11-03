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
from .io import (paths, read_jsonl, append_jsonl, model_output_path, read_model_outputs,
                 save_metrics, ids_sha)
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


class PhotoTraitsInRecordV7(PhotoTraitsInRecord):
    """Desde v7: + side_evidence, el rasgo visible que decidio el lado (para auditar el lado
    y verificar la regla de las lineas). Campo obligatorio para que el modelo siempre lo llene."""
    side_evidence: str


class RecordPhotoTraitsV7(BaseModel):
    photos: list[PhotoTraitsInRecordV7]


SideBasisV9 = Literal["direct", "lines_rule", "bands_rule", "inferred_same_bone",
                      "inferred_other_bone", "unclear", "not_applicable"]


class PhotoTraitsInRecordV9(PhotoTraitsInRecordV7):
    """Desde v9: + 'bands_rule' (ventral por las bandas curvas transversales). EXPERIMENTO:
    la senal ventral esta por verificar (Hugo / Martin)."""
    side_basis: SideBasisV9


class RecordPhotoTraitsV9(BaseModel):
    photos: list[PhotoTraitsInRecordV9]


# --- v10-v11: el modelo marca rasgos en una lista cerrada (side_signs) -------------------
# Rasgos de la literatura (v10) y de la guia de Martin (v11, se agregan). El modelo decide
# 'side' y ademas el codigo calcula 'side_rule' con una tabla fija (side_from_signs), para
# poder medir las dos cosas con una sola corrida.
SignV10 = Literal["inner_cone_rim", "striated_zone", "granular_shield", "median_groove",
                  "chitinous_posterior_cap", "transverse_lines", "lengthwise_lines",
                  "white_chalky_or_porous"]
SignV11 = Literal["inner_cone_rim", "striated_zone", "granular_shield", "median_groove",
                  "chitinous_posterior_cap", "transverse_lines", "lengthwise_lines",
                  "white_chalky_or_porous", "laminated_surface", "bite_marks",
                  "smooth_hard_uniform_face"]
SideBasisV10 = Literal["direct", "inferred_same_bone", "unclear", "not_applicable"]

# Tabla fija rasgo -> lado. 'strong' decide; 'weak' solo cuenta como contradiccion.
VENTRAL_STRONG = {"inner_cone_rim", "striated_zone", "laminated_surface"}
DORSAL_STRONG = {"granular_shield"}
VENTRAL_WEAK = {"median_groove", "bite_marks"}
DORSAL_WEAK = {"chitinous_posterior_cap"}
DORSAL_IF_SHARP = {"smooth_hard_uniform_face"}   # dorsal fuerte solo con calidad 'high'
NEUTRAL = {"transverse_lines", "lengthwise_lines", "white_chalky_or_porous"}


def side_from_signs(signs, photo_quality=None) -> str:
    """
    Lado decidido por regla fija a partir de los rasgos (solo de ESTA foto, sin otras fotos):
    - ventral: algun rasgo ventral fuerte y ningun rasgo dorsal (fuerte ni debil);
    - dorsal: algun rasgo dorsal fuerte y ningun rasgo ventral (fuerte ni debil);
      'smooth_hard_uniform_face' cuenta como dorsal fuerte solo si la calidad es 'high';
    - si no: unclear (contradiccion, solo rasgos debiles o ninguno).
    """
    s = set(signs or [])
    dorsal_strong = s & DORSAL_STRONG
    if photo_quality == "high":
        dorsal_strong |= s & DORSAL_IF_SHARP
    dorsal_any = s & (DORSAL_STRONG | DORSAL_WEAK | DORSAL_IF_SHARP)
    ventral_strong = s & VENTRAL_STRONG
    ventral_any = s & (VENTRAL_STRONG | VENTRAL_WEAK)
    if ventral_strong and not dorsal_any:
        return "ventral"
    if dorsal_strong and not ventral_any:
        return "dorsal"
    return "unclear"


class PhotoTraitsInRecordV10(PhotoTraitsInRecordV7):
    side_basis: SideBasisV10
    side_signs: list[SignV10]


class RecordPhotoTraitsV10(BaseModel):
    photos: list[PhotoTraitsInRecordV10]


class PhotoTraitsInRecordV11(PhotoTraitsInRecordV10):
    side_signs: list[SignV11]


class RecordPhotoTraitsV11(BaseModel):
    photos: list[PhotoTraitsInRecordV11]


# --- v14: calidad descrita por el modelo y decidida por una regla fija --------------------
# El modelo no da la calidad: describe que tapa el hueso (cover) y el foco (focus), y el codigo
# calcula photo_quality con quality_from_cover. Criterio de Hugo (3 nov), compatible con la guia
# de Martin (high = sin arena ni algas encima); umbrales pendientes de confirmar con Martin.
Cover = Literal["none", "few_grains", "scattered_grains_or_small_objects", "large_part_covered",
                "not_applicable"]
Focus = Literal["sharp", "blurry", "not_applicable"]


class PhotoTraitsInRecordV14(BaseModel):
    """Como v7, pero sin photo_quality: + cover, focus y quality_evidence."""
    photo_index: int
    n_cuttlebones_visible: int
    side: Side
    side_basis: SideBasis
    side_evidence: str
    erosion: Erosion
    cover: Cover
    focus: Focus
    quality_evidence: str
    notes: str


class RecordPhotoTraitsV14(BaseModel):
    photos: list[PhotoTraitsInRecordV14]


def quality_from_cover(cover, focus) -> str:
    """
    Regla fija de calidad (v14-):
    - low:    borrosa, o gran parte tapada (> ~60 %);
    - medium: varios o muchos granos, mancha de arena u objetos que tapan parte (~5-60 %);
    - high:   nada encima o solo un par de granos sueltos (>= ~95 % visible);
    - not_applicable: sin descripcion (p. ej. no hay exactamente 1 hueso).
    """
    if cover == "not_applicable" or focus == "not_applicable" or cover is None or focus is None:
        return "not_applicable"
    if focus == "blurry" or cover == "large_part_covered":
        return "low"
    if cover == "scattered_grains_or_small_objects":
        return "medium"
    return "high"


# Versiones donde la calidad la decide quality_from_cover
QUALITY_RULE = {"photo_traits_v14"}


# Esquema de respuesta por version (las que no estan usan el de su unidad).
SCHEMA_BY_VERSION = {"photo_traits_v7": RecordPhotoTraitsV7,
                     "photo_traits_v8": RecordPhotoTraitsV7,
                     "photo_traits_v9": RecordPhotoTraitsV9,
                     "photo_traits_v10": RecordPhotoTraitsV10,
                     "photo_traits_v11": RecordPhotoTraitsV11,
                     "photo_traits_v12": RecordPhotoTraitsV7,
                     "photo_traits_v13": RecordPhotoTraitsV7,
                     "photo_traits_v14": RecordPhotoTraitsV14}


# Unidad de consulta de cada version de prompt (la que no esta, es "photo").
PROMPT_UNIT = {"photo_traits_v5": "record", "photo_traits_v6": "record",
               "photo_traits_v7": "record", "photo_traits_v8": "record",
               "photo_traits_v9": "record", "photo_traits_v10": "record",
               "photo_traits_v11": "record", "photo_traits_v12": "record",
               "photo_traits_v13": "record", "photo_traits_v14": "record"}

# Regla fija de v12: con erosion 3 (segun el modelo) el lado queda 'unclear'.
# Simulada sobre v8 (2 nov): el acierto del lado cae con la erosion (88 % con 1, 76 % con 2,
# 60 % con 3). El lado original del modelo se guarda en side_raw para medir el efecto.
EROSION3_SIDE_UNCLEAR = {"photo_traits_v12", "photo_traits_v13", "photo_traits_v14"}


def apply_rules(result: dict, prompt_version: str | None = None) -> tuple[dict, list[str]]:
    """
    Reglas fijas por foto:
    - v12: con erosion '3' el lado queda 'unclear' (el lado del modelo queda en side_raw);
    - lado, erosion y calidad solo con exactamente 1 hueso visible;
    - side_basis (si existe) coherente con el lado: lado 'unclear' -> base 'unclear';
      lado 'not_applicable' -> base 'not_applicable';
    - con side_signs (v10-): side_rule = lado por la tabla fija (side_from_signs);
    - v14: photo_quality = quality_from_cover(cover, focus).
    """
    r, applied = dict(result), []
    if r.get("n_cuttlebones_visible") != 1:
        keys = ("side", "erosion", "photo_quality") + tuple(
            k for k in ("side_basis", "side_evidence", "cover", "focus", "quality_evidence")
            if k in r)
        for k in keys:
            if r.get(k) != "not_applicable":
                r[k] = "not_applicable"
                applied.append(f"n_visible!=1 -> {k}=not_applicable")
        if r.get("side_signs"):
            r["side_signs"] = []
            applied.append("n_visible!=1 -> side_signs=[]")
    if prompt_version in QUALITY_RULE:
        r["photo_quality"] = quality_from_cover(r.get("cover"), r.get("focus"))
        applied.append(f"quality_rule -> photo_quality={r['photo_quality']}")
    if prompt_version in EROSION3_SIDE_UNCLEAR:
        r["side_raw"] = r.get("side")
        if r.get("erosion") == "3" and r.get("side") in ("dorsal", "ventral"):
            r["side"] = "unclear"
            applied.append("erosion=3 -> side=unclear")
    if "side_basis" in r:
        for side in ("unclear", "not_applicable"):
            if r.get("side") == side and r.get("side_basis") != side:
                r["side_basis"] = side
                applied.append(f"side={side} -> side_basis={side}")
    if "side_signs" in r:
        r["side_rule"] = ("not_applicable" if r.get("n_cuttlebones_visible") != 1
                          else side_from_signs(r["side_signs"], r.get("photo_quality")))
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

    schema = SCHEMA_BY_VERSION.get(
        tc["prompt_version"], RecordPhotoTraits if tc["unit"] == "record" else PhotoTraits)
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
                res, rules = apply_rules(res, tc["prompt_version"])
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
    flat["prompt_sha"] = df["prompt_sha"].values
    return flat


def evaluate_photo_traits(cfg: Config, prompt_version: str | None = None,
                          photo_ids=None, save: bool = True, eval_set: str = "all",
                          quiet: bool = False) -> pd.DataFrame:
    """
    Compara con la anotacion manual, foto por foto:
    - conteo: huesos visibles del modelo vs 'Individuals in photo' del anotador
    - lado, erosion y calidad: solo fotos con un unico individuo anotado y 1 hueso para el modelo
    photo_ids: evaluar solo estas fotos (p. ej. las comunes a varias versiones).
    save=True agrega las metricas a outputs/metrics/photo_traits_metrics.jsonl.
    Devuelve la tabla foto a foto; las metricas quedan en tabla.attrs["metrics"].
    """
    from .annotate import load_events
    tc = _settings(cfg, prompt_version)
    say = (lambda *a: None) if quiet else print
    model = load_photo_traits(cfg, prompt_version)
    say(f"Version: {tc['prompt_version']}")
    ev = load_events(cfg)
    if model.empty:
        say("Sin resultados del modelo todavia")
        return model
    if photo_ids is not None:
        model = model[model["photo_id"].astype(int).isin({int(i) for i in photo_ids})]
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
    one["photo_id"] = one["photo_id"].astype("int64")
    t = (model.rename(columns={c: f"{c}_model" for c in ("side", "erosion", "photo_quality")})
         .merge(info, on="photo_id", how="left").merge(one, on="photo_id", how="left"))

    m = {"n_photos_model": len(t)}
    cnt = t[t["n_in_photo"].notna()]
    m["count_n"] = len(cnt)
    m["count_agreement"] = (float((cnt["n_cuttlebones_visible"] == cnt["n_in_photo"]).mean())
                            if len(cnt) else None)
    if len(cnt):
        say(f"Huesos visibles por foto: {m['count_agreement']:.0%} iguales ({len(cnt)} fotos anotadas)")
    compared = set(cnt["photo_id"].astype(int))
    # v10-: lado por la tabla fija de rasgos, comparado con el mismo lado humano
    if "side_rule" in t:
        t = t.rename(columns={"side_rule": "side_rule_model"})
        t["side_rule_human"] = t["side_human"]
    # v12: lado del modelo antes de la regla de erosion
    if "side_raw" in t:
        t = t.rename(columns={"side_raw": "side_raw_model"})
        t["side_raw_human"] = t["side_human"]
    for var in ("side", "side_raw", "side_rule", "erosion", "photo_quality"):
        if f"{var}_model" not in t:
            continue
        h = t[f"{var}_human"]
        mv = t[f"{var}_model"].astype(str)
        sub = t[h.notna() & (mv != "not_applicable")].copy()
        m[f"{var}_n"] = len(sub)
        if not len(sub):
            m[f"{var}_agreement"] = None
            say(f"{var}: aun no hay fotos comparables")
            continue
        sub["h"] = sub[f"{var}_human"].map(          # erosion humana viene como numero
            lambda v: str(int(v)) if isinstance(v, (int, float)) else str(v))
        sub["m"] = sub[f"{var}_model"].astype(str)
        sub["ok"] = sub["h"] == sub["m"]
        compared |= set(sub["photo_id"].astype(int))
        m[f"{var}_agreement"] = float(sub["ok"].mean())
        ct = pd.crosstab(sub["h"], sub["m"], rownames=["human"], colnames=["model"])
        m[f"{var}_confusion"] = {r: {c: int(ct.loc[r, c]) for c in ct.columns} for r in ct.index}
        say(f"{var}: {m[f'{var}_agreement']:.0%} de acuerdo ({len(sub)} fotos con 1 hueso)")
        say(ct)
        if var in ("side", "side_raw", "side_rule"):
            # el modelo dijo 'unclear' pero la persona si decidio: no es error, es abstencion
            m[f"{var}_unclear_model"] = int((sub["m"] == "unclear").sum())
            decided = sub[sub["m"] != "unclear"]
            m[f"{var}_agreement_when_decided"] = (float(decided["ok"].mean())
                                                  if len(decided) else None)
            # acierto segun la erosion anotada (para decidir si con erosion 3 conviene 'unclear')
            ero = sub["erosion_human"].map(
                lambda v: str(int(v)) if isinstance(v, (int, float)) and v == v else "none")
            dec = sub["m"] != "unclear"
            m[f"{var}_by_erosion"] = {
                e: {"n": int((ero == e).sum()),
                    "unclear": int(((ero == e) & ~dec).sum()),
                    "agreement_when_decided": (float(sub.loc[(ero == e) & dec, "ok"].mean())
                                               if ((ero == e) & dec).any() else None)}
                for e in sorted(ero.unique())}
            say(f"{var} segun erosion anotada: {m[f'{var}_by_erosion']}")
        if var == "side" and "side_signs" in sub:
            # fiabilidad de cada rasgo: en cuantas fotos aparece y que lado anoto la persona
            rel = {}
            for _, row in sub.iterrows():
                for sg in (row["side_signs"] if isinstance(row["side_signs"], list) else []):
                    d = rel.setdefault(sg, {"n": 0, "human_dorsal": 0, "human_ventral": 0})
                    d["n"] += 1
                    if row["h"] in ("dorsal", "ventral"):
                        d[f"human_{row['h']}"] += 1
            m["side_sign_reliability"] = rel
            say("rasgos y lado anotado:")
            say(pd.DataFrame(rel).T)
        if var == "side":
            if "side_basis" in sub:
                # acierto por via de decision (v5-): direct, lines_rule, inferred_*
                by = sub.groupby("side_basis")["ok"].agg(["mean", "size"])
                by.columns = ["agreement", "n_photos"]
                m["side_by_basis"] = {b: {"agreement": float(r.agreement), "n": int(r.n_photos)}
                                      for b, r in by.iterrows()}
                say("lado segun como se decidio:")
                say(by.round(2))
    t.attrs["metrics"] = m
    if save:
        shas = sorted(set(t["prompt_sha"].dropna())) if "prompt_sha" in t else []
        path = save_metrics(cfg, "photo_traits", {
            "provider": tc["provider"], "model": tc["model"],
            "prompt_version": tc["prompt_version"], "prompt_sha": shas,
            "image_size": tc["image_size"], "eval_set": eval_set,
            "eval_set_sha": ids_sha(compared), "n_photos_compared": len(compared),
            "metrics": m})
        say(f"Metricas guardadas en {path}")
    return t


def compare_photo_traits(cfg: Config, versions: list[str], save: bool = True) -> pd.DataFrame:
    """
    Compara varias versiones de prompt sobre las MISMAS fotos: las que tienen resultado
    en todas las versiones y anotacion manual. Asi la diferencia no viene de evaluar
    conjuntos distintos. Guarda una linea de metricas por version (eval_set='common').
    """
    from .annotate import load_events
    ev = load_events(cfg)
    annotated = set(ev["photo_info"]["photo_id"].dropna().astype(int))
    common = annotated
    for v in versions:
        d = load_photo_traits(cfg, v)
        common &= set(d["photo_id"].astype(int)) if len(d) else set()
    print(f"Fotos comunes (anotadas y con resultado en {len(versions)} versiones): {len(common)}")
    rows = []
    for v in versions:
        t = evaluate_photo_traits(cfg, v, photo_ids=common, save=save,
                                  eval_set="common:" + ",".join(versions), quiet=True)
        m = t.attrs.get("metrics", {})
        rows.append({"prompt_version": v,
                     **{k: m.get(k) for k in ("count_agreement", "count_n",
                                              "side_agreement", "side_agreement_when_decided",
                                              "side_unclear_model", "side_n",
                                              "side_raw_agreement_when_decided",
                                              "side_rule_agreement_when_decided",
                                              "side_rule_unclear_model",
                                              "erosion_agreement", "erosion_n",
                                              "photo_quality_agreement", "photo_quality_n")}})
    out = pd.DataFrame(rows).set_index("prompt_version")
    print(out.round(2).to_string())
    return out
