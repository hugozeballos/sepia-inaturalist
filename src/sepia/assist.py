"""
v15: triaje + rasgos por foto + recuadro en UNA consulta, con varios registros por consulta.

Prompt unico: prompts/assist_v15.txt. Para cada registro y cada foto, el modelo sigue una
secuencia fija: contenido -> huesos visibles -> si hay exactamente 1: recuadro, lado, erosion y
calidad (cubierta/foco, como v14), mirando dentro del recuadro. Al final de cada registro, el
resumen del registro (estado y huesos distintos, como el triaje).

Varios registros por consulta (pack_size, por defecto 5) para gastar menos consultas:
- cada registro empieza con "=== NEW RECORD Rk ===" y la instruccion de evaluarlo desde cero;
  sus fotos se rotulan "Rk-P1", "Rk-P2"...;
- el orden de los registros dentro de la consulta se baraja con una semilla fija (los IDs del
  paquete), asi es reproducible y un posible arrastre entre registros no va siempre igual;
- control: cada registro tiene que volver con exactamente sus fotos. Si no, ese registro se
  repite solo (paquete de 1). Lo demas del paquete se guarda.

Salidas:
- outputs/assist_<proveedor>_<modelo>_assist_v15<sufijo>.jsonl: una linea por CONSULTA, con la
  respuesta cruda, los tokens y los registros que incluyo.
- Y, para que las interfaces y evaluaciones existentes la lean sin cambios, lineas derivadas en
  los archivos de siempre, con versiones triage_v15, photo_traits_v15 y bonebox_v15 (+ sufijo):
  triage (una por registro), photo_traits (una por foto) y bonebox (una por foto con 1 hueso).
  Llevan source_prompt = assist_v15, el hash del prompt y el call_id de la consulta.
El sufijo (p. ej. "_p5") sirve para probar tamanos de paquete sin mezclar resultados.

Las versiones derivadas no tienen prompt propio: no correr triage.py ni photo_traits.py con
triage_v15 / photo_traits_v15 (fallan al no encontrar el prompt, y es lo correcto).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import random
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .config import Config
from .io import paths, read_jsonl, append_jsonl, model_output_path, read_model_outputs
from .triage import (load_prompt, make_provider, RateLimited, FatalProviderError,
                     PhotoContent, RecordState, apply_rules as triage_rules)
from .photo_traits import Side, SideBasis, Erosion, Cover, Focus, apply_rules as traits_rules
from .bonebox import apply_rules as box_rules

KIND = "assist"
PROMPT = "assist_v15"


class PhotoV15(BaseModel):
    """Una foto, en el orden de la secuencia (la evidencia antes del valor)."""
    photo_label: str
    content: PhotoContent
    n_cuttlebones_visible: int
    box_2d: list[int]
    side_evidence: str
    side: Side
    side_basis: SideBasis
    erosion: Erosion
    quality_evidence: str
    cover: Cover
    focus: Focus
    notes: str


class RecordV15(BaseModel):
    """Un registro: sus fotos y el resumen del registro (como el triaje)."""
    record_label: str
    photos: list[PhotoV15]
    record_state: RecordState
    n_cuttlebones: int
    count_possible: bool
    same_bone_in_several_photos: bool
    fragments_present: bool
    confidence: Literal["high", "medium", "low"]
    record_notes: str


class PackV15(BaseModel):
    """Respuesta completa de una consulta: una entrada por registro."""
    records: list[RecordV15]


def _settings(cfg: Config, suffix: str = "", pack_size: int | None = None) -> dict:
    """Seccion 'assist' del YAML; proveedor y modelo por defecto los de 'photo_traits'
    (asi las lineas derivadas quedan donde las buscan las interfaces)."""
    base = {k: v for k, v in (cfg.photo_traits or {}).items() if k != "prompt_version"}
    tc = {"provider": "gemini", "model": "gemini-3.5-flash-lite", "temperature": 0.0,
          "image_size": 1024, "requests_per_minute": 8, "thinking_level": "low",
          "timeout_s": 300, **base, "pack_size": 5, **(cfg.assist or {}),
          "prompt_version": PROMPT}
    if pack_size:
        tc["pack_size"] = int(pack_size)
    tc["suffix"] = suffix
    tc["versions"] = {k: f"{k}_v15{suffix}" for k in ("triage", "photo_traits", "bonebox")}
    return tc


def make_packs(record_ids: list[int], pack_size: int) -> list[list[int]]:
    """Paquetes de pack_size registros en orden de ID; dentro de cada uno, orden barajado con
    una semilla fija (los IDs del paquete): reproducible."""
    packs = []
    for k in range(0, len(record_ids), pack_size):
        pack = sorted(record_ids[k:k + pack_size])
        seed = int(hashlib.sha256(",".join(map(str, pack)).encode()).hexdigest()[:8], 16)
        random.Random(seed).shuffle(pack)
        packs.append(pack)
    return packs


def build_request(pack: list[int], photos_by_record: dict) -> tuple[list[str], list[str], dict]:
    """
    Rotulos de cada foto (texto antes de la imagen), archivos en orden y mapa de etiquetas:
    {'R2': (observation_id, ['R2-P1', ...]), ...}.
    """
    labels, files, expected = [], [], {}
    for k, oid in enumerate(pack, start=1):
        rec = f"R{k}"
        g = photos_by_record[oid]
        plabels = [f"{rec}-P{j}" for j in range(1, len(g) + 1)]
        expected[rec] = (oid, plabels)
        for j, (fname, pl) in enumerate(zip(g["file_name"], plabels)):
            head = (f"=== NEW RECORD {rec} === Evaluate this record from scratch; do not use "
                    f"anything from other records. It has {len(g)} photo(s).\n") if j == 0 else ""
            labels.append(f"{head}{pl}:")
            files.append(fname)
    return labels, files, expected


def check_response(result: dict, expected: dict) -> tuple[dict, list[str]]:
    """
    Registros validos de la respuesta ({etiqueta: registro}) y etiquetas que fallaron
    (faltan, sobran o no traen exactamente sus fotos).
    """
    got = {}
    for r in result.get("records") or []:
        got.setdefault(r.get("record_label"), []).append(r)
    ok, bad = {}, []
    for rec, (_, plabels) in expected.items():
        items = got.get(rec) or []
        if len(items) != 1:
            bad.append(rec)
            continue
        labs = sorted(p.get("photo_label") for p in items[0].get("photos") or [])
        if labs != sorted(plabels):
            bad.append(rec)
            continue
        ok[rec] = items[0]
    return ok, bad


def derived_lines(rec: dict, oid: int, g, base: dict, tc: dict) -> dict[str, list[dict]]:
    """Lineas para los archivos de triage, photo_traits y bonebox desde un registro valido."""
    by_label = {p["photo_label"]: p for p in rec["photos"]}
    rec_label = rec["record_label"]
    out = {"triage": [], "photo_traits": [], "bonebox": []}
    tri_photos = []
    for j, r in enumerate(g.itertuples(index=False), start=1):
        ph = dict(by_label[f"{rec_label}-P{j}"])
        tri_photos.append({"photo_index": j, "content": ph["content"],
                           "n_cuttlebones_visible": ph["n_cuttlebones_visible"]})
        common = {**base, "observation_id": oid, "photo_id": int(r.photo_id),
                  "position": int(r.position), "photo_label": ph["photo_label"],
                  "raw_response": None, "error": None}
        traits = {k: v for k, v in ph.items() if k not in ("box_2d", "photo_label", "content")}
        traits["photo_index"] = j
        traits, rules = traits_rules(traits, "photo_traits_v15")
        traits["content"] = ph["content"]
        out["photo_traits"].append({**common, "prompt_version": tc["versions"]["photo_traits"],
                                    "result": traits, "rules_applied": rules})
        if ph["n_cuttlebones_visible"] == 1:
            box, brules = box_rules({"one_cuttlebone": True, "box_2d": ph["box_2d"],
                                     "notes": ""})
            out["bonebox"].append({**common, "prompt_version": tc["versions"]["bonebox"],
                                   "result": box, "rules_applied": brules})
    tri = {"photos": tri_photos, **{k: rec[k] for k in (
        "record_state", "n_cuttlebones", "count_possible", "same_bone_in_several_photos",
        "fragments_present", "confidence")}, "notes": rec.get("record_notes", "")}
    tri, trules = triage_rules(tri)
    out["triage"].append({**base, "observation_id": oid,
                          "photo_ids": [int(x) for x in g["photo_id"]],
                          "prompt_version": tc["versions"]["triage"], "result": tri,
                          "rules_applied": trules, "raw_response": None, "error": None})
    return out


def done_records(cfg: Config, tc: dict, prompt_sha: str) -> set[int]:
    """Registros ya hechos sin error con este prompt (segun las lineas de triaje derivadas)."""
    df = read_model_outputs(cfg, "triage", tc["provider"], tc["model"], tc["versions"]["triage"])
    if df.empty:
        return set()
    ok = df[df["error"].isna() & (df["prompt_sha"] == prompt_sha)]
    return set(ok["observation_id"].astype(int))


def _call(provider, images, prompt, labels):
    """Llama al modelo con reintentos ante limite de uso o saturacion (4 intentos)."""
    for attempt in range(4):
        try:
            return provider.classify(images, prompt, schema=PackV15, labels=labels)
        except RateLimited as e:
            if attempt == 3:
                raise
            wait = 30 * 2 ** attempt
            print(f"  limite de uso o modelo saturado; espero {wait} s · {str(e)[:160]}", flush=True)
            time.sleep(wait)


def run_assist(cfg: Config, provider=None, pack_size: int | None = None, suffix: str = "",
               ids: list[int] | None = None, limit: int | None = None,
               redo: bool = False) -> dict:
    """
    Corre v15 por paquetes. limit = numero de CONSULTAS (paquetes). Reanudable: salta los
    registros ya hechos con este prompt. suffix: etiqueta de prueba (p. ej. '_p5').
    """
    tc = _settings(cfg, suffix, pack_size)
    prompt, prompt_sha = load_prompt(PROMPT)
    provider = provider or make_provider(tc["provider"], tc["model"], tc["temperature"],
                                         thinking_level=tc["thinking_level"],
                                         timeout_s=tc["timeout_s"])
    p = paths(cfg)
    img_dir = p["base"] / "photos" / f"{tc['image_size']}px"
    pho = read_jsonl(p["tables"] / "photos.jsonl").sort_values(["observation_id", "position"])
    if ids is not None:
        pho = pho[pho["observation_id"].isin({int(i) for i in ids})]
    photos_by_record = {int(o): g for o, g in pho.groupby("observation_id", sort=True)}
    skip = set() if redo else done_records(cfg, tc, prompt_sha)
    todo = [o for o in photos_by_record if o not in skip]
    packs = make_packs(todo, tc["pack_size"])
    if limit:
        packs = packs[:limit]
    outs = {k: model_output_path(cfg, k, tc["provider"], tc["model"], v)
            for k, v in tc["versions"].items()}
    raw_out = model_output_path(cfg, KIND, tc["provider"], tc["model"], PROMPT + suffix)
    print(f"v15 con {tc['provider']}/{tc['model']} · prompt {PROMPT} ({prompt_sha}) · "
          f"{tc['pack_size']} registros por consulta · {len(skip)} registros hechos · "
          f"{len(todo)} por hacer en {len(packs)} consultas")
    print(f"salidas: {', '.join(tc['versions'].values())} (+ {raw_out.name})")

    pause = 60.0 / tc["requests_per_minute"]
    stats = {"calls": 0, "records_ok": 0, "records_retried": 0, "records_error": 0,
             "input_tokens": 0, "output_tokens": 0}
    queue = [(pk, False) for pk in packs]        # (paquete, es_reintento)
    n = 0
    while queue:
        pack, is_retry = queue.pop(0)
        n += 1
        call_id = f"v15-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        labels, files, expected = build_request(pack, photos_by_record)
        base = {"provider": tc["provider"], "model": tc["model"],
                "temperature": tc["temperature"], "source_prompt": PROMPT,
                "prompt_sha": prompt_sha, "image_size": tc["image_size"],
                "thinking_level": tc["thinking_level"], "unit": "pack", "call_id": call_id,
                "pack_size": len(pack), "is_retry": is_retry}
        raw_line = {**base, "observation_ids": pack, "labels": {k: v[0] for k, v in expected.items()}}
        lines = {"triage": [], "photo_traits": [], "bonebox": []}
        try:
            images = [(img_dir / (Path(f).stem + ".jpg")).read_bytes() for f in files]
            t0 = time.time()
            result, raw, usage = _call(provider, images, prompt, labels)
            elapsed = round(time.time() - t0, 1)
            ok, bad = check_response(result, expected)
            for rec_label, rec in ok.items():
                oid = expected[rec_label][0]
                for k, ls in derived_lines(rec, oid, photos_by_record[oid],
                                           {**base, "elapsed_s": elapsed}, tc).items():
                    lines[k] += ls
            stats["records_ok"] += len(ok)
            stats["input_tokens"] += usage.get("input_tokens") or 0
            stats["output_tokens"] += usage.get("output_tokens") or 0
            raw_line.update(raw_response=raw, **usage, elapsed_s=elapsed, error=None,
                            records_ok=sorted(expected[r][0] for r in ok),
                            records_bad=sorted(expected[r][0] for r in bad))
            msg = f"{n} · {len(pack)} registros · ok {len(ok)}"
            if bad:
                bad_ids = [expected[r][0] for r in bad]
                if len(pack) > 1:                 # repetir solos los que fallaron
                    queue[:0] = [([o], True) for o in bad_ids]
                    stats["records_retried"] += len(bad_ids)
                    msg += f" · se repiten solos: {bad_ids}"
                else:
                    stats["records_error"] += 1
                    msg += f" · ERROR en {bad_ids} (respuesta incompleta)"
                    lines["triage"].append({**base, "observation_id": bad_ids[0], "result": None,
                                            "prompt_version": tc["versions"]["triage"],
                                            "raw_response": None,
                                            "error": "ValueError: incomplete record in response"})
            print(f"{msg} · {elapsed} s · {usage.get('input_tokens')}+"
                  f"{usage.get('output_tokens')} tokens", flush=True)
        except RateLimited as e:
            print(f"Limite de uso persistente. Se detiene; reanudar mas tarde.\n  {e}")
            break
        except FatalProviderError as e:
            print(f"Error de configuracion: se detiene.\n  {e}")
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            raw_line.update(raw_response=None, error=err)
            print(f"{n} · {pack} · ERROR {err[:200]}", flush=True)
            if len(pack) > 1:
                queue[:0] = [([o], True) for o in pack]
                stats["records_retried"] += len(pack)
            else:
                stats["records_error"] += 1
                lines["triage"].append({**base, "observation_id": pack[0], "result": None,
                                        "prompt_version": tc["versions"]["triage"],
                                        "raw_response": None, "error": err})
        stats["calls"] += 1
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        raw_line["saved_at"] = now
        append_jsonl(raw_line, raw_out)
        for k, ls in lines.items():
            for line in ls:
                line["saved_at"] = now
                append_jsonl(line, outs[k])
        time.sleep(pause)
    print(f"Listo: {stats}")
    return stats
