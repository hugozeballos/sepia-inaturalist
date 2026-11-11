"""
Fase de triaje: clasificacion por registro con un modelo de vision.

Para cada registro se envian TODAS sus fotos juntas (version reducida, 512 px),
para que el modelo pueda reconocer el mismo hueso en varias fotos y no
contarlo dos veces. Devuelve estado del registro, numero de huesos distintos
y que muestra cada foto.

Reproducibilidad:
- Proveedor, modelo, temperatura y version del prompt quedan en configs/*.yaml.
- Cada respuesta se guarda completa (JSON) en
  outputs/triage_<proveedor>_<modelo>_<version_prompt>.jsonl (un archivo por version
  de prompt), con el hash del prompt, los tokens usados y la fecha.
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
from .io import paths, read_jsonl, append_jsonl, model_output_path, read_model_outputs

# --------------------------------------------------------------------------
# Esquema de la respuesta (el modelo debe devolver exactamente esto)
# --------------------------------------------------------------------------

PhotoContent = Literal["cuttlebone", "live_animal", "egg_case", "other", "unclear"]
RecordState = Literal["cuttlebone", "live_animal", "egg_case", "mixed", "other", "unclear"]


class PhotoResult(BaseModel):
    """Resultado del triaje para una foto: contenido y huesos visibles."""
    photo_index: int = Field(description="1-based index, as labelled 'Photo N'")
    content: PhotoContent
    n_cuttlebones_visible: int = Field(description="cuttlebones visible in this photo")


class TriageResult(BaseModel):
    """Resultado del triaje para un registro (esquema que el modelo debe devolver)."""
    photos: list[PhotoResult]
    record_state: RecordState
    n_cuttlebones: int
    count_possible: bool
    same_bone_in_several_photos: bool
    fragments_present: bool = False   # desde triage_v2
    confidence: Literal["high", "medium", "low"]
    notes: str


# --------------------------------------------------------------------------
# Reglas fijas sobre la respuesta (deterministas, documentadas en los metodos)
# --------------------------------------------------------------------------

MAX_EXACT_COUNT = 5   # sobre este numero de huesos, el conteo lo hace una persona


def apply_rules(result: dict) -> tuple[dict, list[str]]:
    """
    Corrige respuestas incoherentes del modelo con reglas fijas:
    - mas de MAX_EXACT_COUNT huesos -> count_possible = False
    - conteo no posible -> la confianza no puede ser 'high'
    Devuelve (resultado corregido, lista de reglas aplicadas).
    """
    r = dict(result)
    applied = []
    if r.get("n_cuttlebones", 0) > MAX_EXACT_COUNT and r.get("count_possible"):
        r["count_possible"] = False
        applied.append(f"n_cuttlebones>{MAX_EXACT_COUNT} -> count_possible=false")
    if not r.get("count_possible") and r.get("confidence") == "high":
        r["confidence"] = "medium"
        applied.append("count_possible=false -> confidence=medium")
    return r, applied


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
        """Lee la clave GEMINI_API_KEY del entorno y prepara el cliente de google-genai."""
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

    def classify(self, images: list[bytes], prompt: str, schema=None,
                 labels: list[str] | None = None) -> tuple[dict, str, dict]:
        """
        schema: modelo Pydantic de la respuesta (por defecto TriageResult).
        labels: texto antes de cada foto (por defecto "Photo N:"), p. ej. para varios registros.
        """
        schema = schema or TriageResult
        t = self._types
        contents = []
        for i, img in enumerate(images, start=1):
            contents.append(labels[i - 1] if labels else f"Photo {i}:")
            contents.append(t.Part.from_bytes(data=img, mime_type="image/jpeg"))
        contents.append(prompt)
        try:
            r = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=t.GenerateContentConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
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
        parsed = schema.model_validate_json(r.text).model_dump()
        return parsed, r.text, usage


def extract_json(text: str) -> str:
    """
    El JSON de una respuesta de texto libre: quita el razonamiento (<think>...</think>), los
    cercos ```json y lo que haya antes de la primera { o despues de la ultima }.
    """
    import re
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"<\|?begin_of_box\|?>|<\|?end_of_box\|?>", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError(f"la respuesta no trae JSON: {text[:200]!r}")
    return text[a:b + 1]


class ZaiProvider:
    """
    GLM (Z.ai) por su API compatible con OpenAI, p. ej. glm-4.6v-flash (gratis).
    Requiere ZAI_API_KEY en el entorno. A diferencia de Gemini, la API no garantiza la estructura
    del JSON: el esquema va escrito al final del prompt y la respuesta se valida con Pydantic
    (si no cumple, la foto queda con error y se puede reintentar).
    """
    name = "zai"
    BASE_URL = "https://api.z.ai/api/paas/v4"

    def __init__(self, model: str, temperature: float = 0.0, thinking: bool = False,
                 timeout_s: int = 120):
        import requests
        key = os.environ.get("ZAI_API_KEY")
        if not key:
            raise RuntimeError("Falta la variable de entorno ZAI_API_KEY")
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"})
        self.model, self.temperature, self.timeout_s = model, temperature, timeout_s
        self.thinking = thinking

    def classify(self, images: list[bytes], prompt: str, schema=None,
                 labels: list[str] | None = None) -> tuple[dict, str, dict]:
        """Misma interfaz que GeminiProvider: (resultado validado, texto crudo, tokens)."""
        import base64
        schema = schema or TriageResult
        content = []
        for i, img in enumerate(images, start=1):
            content.append({"type": "text", "text": labels[i - 1] if labels else f"Photo {i}:"})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}})
        content.append({"type": "text", "text": prompt + (
            "\n\nAnswer with ONLY one JSON object (no other text) that follows this JSON schema:\n"
            + json.dumps(schema.model_json_schema(), separators=(",", ":")))})
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "user", "content": content}],
                "thinking": {"type": "enabled" if self.thinking else "disabled"}}
        r = self._session.post(f"{self.BASE_URL}/chat/completions", json=body,
                               timeout=self.timeout_s)
        # Z.ai responde 429 tambien por problemas de cuenta (1113: sin saldo o sin paquete de
        # recursos): eso no se arregla esperando, asi que detiene la corrida con el mensaje.
        if r.status_code == 429 and any(c in r.text for c in ('"1113"', "1113", "1112", "1301")):
            raise FatalProviderError(f"HTTP 429 (cuenta Z.ai): {r.text[:300]}")
        if r.status_code == 429 or r.status_code >= 500:
            raise RateLimited(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code in (400, 401, 403, 404):
            raise FatalProviderError(f"HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                 "thinking_tokens": None}
        parsed = schema.model_validate_json(extract_json(text)).model_dump()
        return parsed, text, usage


def make_provider(provider: str, model: str, temperature: float = 0.0, **opts):
    """Crea el proveedor de modelo indicado en la configuracion: 'gemini' o 'zai' (GLM)."""
    if provider == "gemini":
        return GeminiProvider(model, temperature,
                              thinking_level=opts.get("thinking_level", "low"),
                              timeout_s=opts.get("timeout_s", 120))
    if provider == "zai":
        return ZaiProvider(model, temperature, thinking=bool(opts.get("thinking_level")),
                           timeout_s=opts.get("timeout_s", 120))
    raise ValueError(f"Proveedor no soportado todavia: {provider}")


# --------------------------------------------------------------------------
# Corrida del triaje
# --------------------------------------------------------------------------

def output_path(cfg: Config, provider: str, model: str, prompt_version: str) -> Path:
    """Un archivo por version de prompt: outputs/triage_<proveedor>_<modelo>_<version>.jsonl"""
    return model_output_path(cfg, "triage", provider, model, prompt_version)


def done_ids(df, prompt_sha: str | None = None) -> set[int]:
    """IDs ya clasificados sin error (para reanudar), con el MISMO texto de prompt (hash).
    df: lineas guardadas de esta version (read_model_outputs)."""
    if df.empty:
        return set()
    ok = df[df["error"].isna()] if "error" in df else df
    if prompt_sha is not None and "prompt_sha" in ok:
        ok = ok[ok["prompt_sha"] == prompt_sha]
    return set(ok["observation_id"].astype(int))


def run_triage(cfg: Config, provider=None, limit: int | None = None,
               ids: list[int] | None = None, redo: bool = False) -> dict:
    """
    Clasifica cada registro con todas sus fotos (version de cfg.triage['image_size'] px).
    Guarda una linea por registro en outputs/triage_<proveedor>_<modelo>_<version>.jsonl.
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
    out = output_path(cfg, tc["provider"], tc["model"], tc["prompt_version"])
    saved = read_model_outputs(cfg, "triage", tc["provider"], tc["model"], tc["prompt_version"])

    obs = read_jsonl(p["tables"] / "observations.jsonl")
    pho = read_jsonl(p["tables"] / "photos.jsonl").sort_values(["observation_id", "position"])
    # Por defecto se saltan los ya clasificados con ESTA version del prompt.
    # redo=True los vuelve a clasificar: se agregan lineas nuevas (no se borra nada)
    # y al leer vale la ultima.
    skip = set() if redo else done_ids(saved, prompt_sha)
    # Registros excluidos por sector (excluded_reason) no tienen fotos en photos.jsonl: se saltan
    with_photos = set(pho["observation_id"].astype(int)) if len(pho) else set()
    todo = [int(i) for i in obs["observation_id"] if int(i) not in skip and int(i) in with_photos]
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
    print(("modo REHACER: se vuelven a clasificar · " if redo else f"{len(skip)} ya clasificados · ")
          + f"{len(todo)} por clasificar")

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
                    print(f"  limite de uso o modelo saturado; espero {wait} s · {str(e)[:160]}",
                          flush=True)
                    time.sleep(wait)
            elapsed = round(time.time() - t0, 1)
            result, rules = apply_rules(result)
            record.update(result=result, rules_applied=rules, raw_response=raw, **usage,
                          elapsed_s=elapsed, error=None)
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


def load_triage(cfg: Config, provider: str, model: str, prompt_version: str | None = None):
    """
    Resultados del triaje como tabla plana (una fila por registro, vale la ultima).
    prompt_version: si se indica, solo esa version (p. ej. 'triage_v2'); por defecto
    la de cfg.triage, si existe.
    """
    import pandas as pd
    if prompt_version is None:
        prompt_version = (cfg.triage or {}).get("prompt_version", "triage_v1")
    df = read_model_outputs(cfg, "triage", provider, model, prompt_version)
    if df.empty:
        return df
    df = df[df["error"].isna()]
    if df.empty:
        return df
    df = df.sort_values("saved_at").drop_duplicates("observation_id", keep="last")
    flat = pd.json_normalize(df["result"].tolist())
    flat.insert(0, "observation_id", df["observation_id"].values)
    return flat


# --------------------------------------------------------------------------
# Evaluacion contra casos de prueba con respuesta conocida
# --------------------------------------------------------------------------

def load_cases(path: str):
    """Casos de prueba (JSONL): observation_id + respuesta esperada."""
    from .io import read_jsonl
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de casos: {path.resolve()} "
                                "(¿falta un git pull, o no estas en la carpeta del repositorio?)")
    cases = read_jsonl(path)
    if cases.empty or "observation_id" not in cases:
        raise ValueError(f"El archivo de casos esta vacio o no tiene 'observation_id': {path}")
    return cases


def evaluate_triage(cfg: Config, provider: str, model: str, cases_path: str,
                    save: bool = True):
    """
    Compara el triaje con los casos de prueba. Criterio:
    - estado igual al esperado;
    - si se esperan <= MAX_EXACT_COUNT huesos: numero exacto y count_possible igual;
    - si se esperan mas: el modelo debe marcar > MAX_EXACT_COUNT y count_possible = False
      (el numero exacto lo cuenta una persona).
    """
    import pandas as pd
    cases = load_cases(cases_path)
    tri = load_triage(cfg, provider, model)
    rows = []
    for c in cases.to_dict(orient="records"):
        got = tri.loc[tri["observation_id"] == c["observation_id"]] if len(tri) else tri
        if got.empty:
            rows.append({"observation_id": c["observation_id"], "status": "sin resultado"})
            continue
        g = got.iloc[0]
        state_ok = g["record_state"] == c["record_state"]
        if c["n_cuttlebones"] > MAX_EXACT_COUNT:
            count_ok = (g["n_cuttlebones"] > MAX_EXACT_COUNT) and not g["count_possible"]
        else:
            count_ok = (g["n_cuttlebones"] == c["n_cuttlebones"]
                        and bool(g["count_possible"]) == bool(c["count_possible"]))
        rows.append({"observation_id": c["observation_id"],
                     "expected": f"{c['record_state']}, {c['n_cuttlebones']} ({'exacto' if c['count_possible'] else 'no exacto'})",
                     "got": f"{g['record_state']}, {g['n_cuttlebones']} ({'exacto' if g['count_possible'] else 'no exacto'}), {g['confidence']}",
                     "state_ok": state_ok, "count_ok": count_ok,
                     "status": "OK" if (state_ok and count_ok) else "FALLA"})
    out = pd.DataFrame(rows, columns=["observation_id", "expected", "got",
                                      "state_ok", "count_ok", "status"])
    if tri.empty:
        print(f"No hay resultados de triaje para {provider}/{model}: correr scripts/triage.py primero")
    n_ok = (out["status"] == "OK").sum()
    print(f"{n_ok}/{len(out)} casos correctos")
    if save and not tri.empty:
        # una linea por evaluacion en outputs/metrics/triage_metrics.jsonl (solo se agrega)
        from .io import save_metrics, ids_sha
        done = out[out["status"] != "sin resultado"]
        version = (cfg.triage or {}).get("prompt_version", "triage_v1")   # igual que load_triage
        _, sha = load_prompt(version)
        path = save_metrics(cfg, "triage", {
            "provider": provider, "model": model, "prompt_version": version, "prompt_sha": sha,
            "eval_set": Path(cases_path).name, "eval_set_sha": ids_sha(cases["observation_id"]),
            "metrics": {"n_cases": len(out), "n_with_result": len(done), "n_ok": int(n_ok),
                        "state_ok": int(done["state_ok"].sum()),
                        "count_ok": int(done["count_ok"].sum()),
                        "failed_ids": [int(i) for i in out.loc[out["status"] == "FALLA",
                                                                 "observation_id"]]}})
        print(f"Metricas guardadas en {path}")
    return out
