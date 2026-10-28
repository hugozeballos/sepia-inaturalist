"""
Fase de triaje: clasificacion por registro con un modelo de vision.

Para cada registro se envian TODAS sus fotos juntas (version reducida, 512 px),
para que el modelo pueda reconocer el mismo hueso en varias fotos y no
contarlo dos veces. Devuelve estado del registro, numero de huesos distintos
y que muestra cada foto.

Reproducibilidad:
- Proveedor, modelo, temperatura y version del prompt quedan en configs/*.yaml.
- Cada respuesta se guarda completa (JSON) en outputs/triage_<proveedor>_<modelo>.jsonl,
  con el hash del prompt, los tokens usados y la fecha.
- Reanudable: los registros ya clasificados (sin error) se saltan.
- El proveedor es intercambiable (gemini ahora; qwen despues).

La clave de API se lee de una variable de entorno (GEMINI_API_KEY), nunca del codigo.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import Config
from .io import paths, read_jsonl, append_jsonl

# --------------------------------------------------------------------------
# Esquema de la respuesta (el modelo debe devolver exactamente esto)
# --------------------------------------------------------------------------

PhotoContent = Literal["cuttlebone", "live_animal", "egg_case", "other", "unclear"]
RecordState = Literal["cuttlebone", "live_animal", "egg_case", "mixed", "other", "unclear"]


class PhotoResult(BaseModel):
    photo_index: int = Field(description="1-based index, as labelled 'Photo N'")
    content: PhotoContent
    n_cuttlebones_visible: int = Field(description="cuttlebones visible in this photo")


class TriageResult(BaseModel):
    photos: list[PhotoResult]
    record_state: RecordState
    n_cuttlebones: int
    count_possible: bool
    same_bone_in_several_photos: bool
    confidence: Literal["high", "medium", "low"]
    notes: str


# --------------------------------------------------------------------------
# Prompt versionado
# --------------------------------------------------------------------------

def load_prompt(version: str) -> tuple[str, str]:
    """Devuelve (texto, sha256 corto) del prompt src/sepia/prompts/<version>.txt."""
    text = resources.files("sepia").joinpath(f"prompts/{version}.txt").read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# Proveedores (intercambiables)
# --------------------------------------------------------------------------

class RateLimited(Exception):
    """Rechazo temporal: limite de uso (429) o modelo saturado (503). Se reintenta."""


class FatalProviderError(Exception):
    """Error de configuracion (modelo inexistente, clave invalida, sin permiso).
    No se arregla reintentando: la corrida se detiene al primer caso."""


FATAL_MARKERS = ("404", "NOT_FOUND", "401", "UNAUTHENTICATED", "403",
                 "PERMISSION_DENIED", "API_KEY_INVALID", "INVALID_ARGUMENT")


class GeminiProvider:
    """Gemini via google-genai. Requiere GEMINI_API_KEY en el entorno."""
    name = "gemini"

    def __init__(self, model: str, temperature: float = 0.0,
                 thinking_level: str | None = "low", timeout_s: int = 120):
        from google import genai
        from google.genai import types
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY")
        # timeout: ninguna llamada puede quedar colgada mas de timeout_s segundos
        self._client = genai.Client(api_key=key,
                                    http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self._types = types
        self.model = model
        self.temperature = temperature
        # razonamiento del modelo: MINIMAL, LOW, MEDIUM, HIGH (None = lo que decida el modelo)
        self.thinking_level = thinking_level.upper() if thinking_level else None

    def classify(self, images: list[bytes], prompt: str) -> tuple[dict, str, dict]:
        t = self._types
        contents = []
        for i, img in enumerate(images, start=1):
            contents.append(f"Photo {i}:")
            contents.append(t.Part.from_bytes(data=img, mime_type="image/jpeg"))
        contents.append(prompt)
        try:
            r = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=t.GenerateContentConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                    response_schema=TriageResult,
                    thinking_config=(t.ThinkingConfig(thinking_level=self.thinking_level)
                                     if self.thinking_level else None),
                ),
            )
        except Exception as e:  # el SDK usa varias clases de error
            msg = str(e)
            # 429 = limite de uso; 503 = modelo saturado. Ambos son temporales: esperar y reintentar
            if any(m in msg for m in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                raise RateLimited(msg) from e
            if any(m in msg for m in FATAL_MARKERS):
                raise FatalProviderError(msg) from e
            raise
        u = r.usage_metadata
        usage = {"input_tokens": getattr(u, "prompt_token_count", None),
                 "output_tokens": getattr(u, "candidates_token_count", None),
                 "thinking_tokens": getattr(u, "thoughts_token_count", None)}
        parsed = TriageResult.model_validate_json(r.text).model_dump()
        return parsed, r.text, usage


def make_provider(provider: str, model: str, temperature: float = 0.0, **opts):
    if provider == "gemini":
        return GeminiProvider(model, temperature,
                              thinking_level=opts.get("thinking_level", "low"),
                              timeout_s=opts.get("timeout_s", 120))
    raise ValueError(f"Proveedor no soportado todavia: {provider}")


# --------------------------------------------------------------------------
# Corrida del triaje
# --------------------------------------------------------------------------

def output_path(cfg: Config, provider: str, model: str) -> Path:
    p = paths(cfg)
    out = p["base"] / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"triage_{provider}_{model.replace('/', '-')}.jsonl"


def done_ids(path: Path) -> set[int]:
    """IDs ya clasificados sin error (para reanudar)."""
    df = read_jsonl(path)
    if df.empty:
        return set()
    ok = df[df["error"].isna()] if "error" in df else df
    return set(ok["observation_id"].astype(int))


def run_triage(cfg: Config, provider=None, limit: int | None = None,
               ids: list[int] | None = None) -> dict:
    """
    Clasifica cada registro con todas sus fotos (version de cfg.triage['image_size'] px).
    Guarda una linea por registro en outputs/triage_<proveedor>_<modelo>.jsonl.
    """
    tc = {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0.0,
          "prompt_version": "triage_v1", "image_size": 512, "requests_per_minute": 8,
          **(cfg.triage or {})}
    prompt, prompt_sha = load_prompt(tc["prompt_version"])
    provider = provider or make_provider(tc["provider"], tc["model"], tc["temperature"],
                                         thinking_level=tc.get("thinking_level", "low"),
                                         timeout_s=tc.get("timeout_s", 120))
    p = paths(cfg)
    img_dir = p["base"] / "photos" / f"{tc['image_size']}px"
    out = output_path(cfg, tc["provider"], tc["model"])

    obs = read_jsonl(p["tables"] / "observations.jsonl")
    pho = read_jsonl(p["tables"] / "photos.jsonl").sort_values(["observation_id", "position"])
    skip = done_ids(out)
    todo = [int(i) for i in obs["observation_id"] if int(i) not in skip]
    if ids is not None:                       # solo una lista de registros (casos de prueba)
        wanted = {int(i) for i in ids}
        missing = wanted - set(obs["observation_id"].astype(int))
        if missing:
            print(f"No estan en esta descarga: {sorted(missing)}")
        todo = [i for i in todo if i in wanted]
    if limit:
        todo = todo[:limit]
    print(f"Triaje con {tc['provider']}/{tc['model']} · prompt {tc['prompt_version']} ({prompt_sha}) "
          f"· razonamiento {tc.get('thinking_level', 'low')} · timeout {tc.get('timeout_s', 120)} s")
    print(f"{len(skip)} ya clasificados · {len(todo)} por clasificar")

    pause = 60.0 / tc["requests_per_minute"]
    stats = {"ok": 0, "error": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    for n, oid in enumerate(todo, start=1):
        files = pho.loc[pho["observation_id"] == oid, ["photo_id", "file_name"]]
        photo_ids = [int(x) for x in files["photo_id"]]
        record = {"observation_id": oid, "photo_ids": photo_ids,
                  "provider": tc["provider"], "model": tc["model"],
                  "temperature": tc["temperature"], "prompt_version": tc["prompt_version"],
                  "prompt_sha": prompt_sha, "image_size": tc["image_size"],
                  "thinking_level": tc.get("thinking_level", "low")}
        try:
            images = [(img_dir / (Path(f).stem + ".jpg")).read_bytes() for f in files["file_name"]]
            if not images:
                raise FileNotFoundError("registro sin fotos reducidas")
            t0 = time.time()
            for attempt in range(4):
                try:
                    result, raw, usage = provider.classify(images, prompt)
                    break
                except RateLimited as e:
                    if attempt == 3:
                        raise
                    wait = 30 * 2 ** attempt
                    print(f"  limite de uso o modelo saturado; espero {wait} s", flush=True)
                    time.sleep(wait)
            elapsed = round(time.time() - t0, 1)
            record.update(result=result, raw_response=raw, **usage, elapsed_s=elapsed, error=None)
            stats["ok"] += 1
            print(f"{n}/{len(todo)} · {oid} · {len(images)} fotos · {result['record_state']}, "
                  f"{result['n_cuttlebones']} huesos ({result['confidence']}) · {elapsed} s · "
                  f"{usage.get('input_tokens')}+{usage.get('output_tokens')} tokens "
                  f"(+{usage.get('thinking_tokens') or 0} razonamiento)", flush=True)
            stats["input_tokens"] += usage.get("input_tokens") or 0
            stats["output_tokens"] += usage.get("output_tokens") or 0
            stats["thinking_tokens"] += usage.get("thinking_tokens") or 0
        except RateLimited as e:
            print(f"Limite de uso o saturacion persistente. Se detiene; reanudar mas tarde.\n  {e}")
            break
        except FatalProviderError as e:
            print(f"Error de configuracion: se detiene (revisar modelo o clave en el YAML / secretos).\n  {e}")
            break
        except Exception as e:
            record.update(result=None, raw_response=None, error=f"{type(e).__name__}: {e}")
            stats["error"] += 1
            print(f"{n}/{len(todo)} · {oid} · ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
        record["saved_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        append_jsonl(record, out)
        time.sleep(pause)

    print(f"Listo: {stats} · salida en {out}")
    return stats


def load_triage(cfg: Config, provider: str, model: str):
    """Resultados del triaje como tabla plana (una fila por registro, vale la ultima)."""
    import pandas as pd
    df = read_jsonl(output_path(cfg, provider, model))
    if df.empty:
        return df
    df = df[df["error"].isna()].sort_values("saved_at").drop_duplicates("observation_id", keep="last")
    flat = pd.json_normalize(df["result"].tolist())
    flat.insert(0, "observation_id", df["observation_id"].values)
    return flat
