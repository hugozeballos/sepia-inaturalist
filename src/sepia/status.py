"""
Estado general de una corrida: descarga y modelos (que version de prompt paso por cuantos
registros y fotos, errores pendientes y repetidos). Solo lee archivos; no cambia nada.

Uso (Colab):
    from sepia.status import show_status
    show_status(cfg)                       # una corrida
    show_status([cfg_b, cfg_c])            # varias
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .io import paths, read_jsonl

# Prefijo del archivo -> paso. 'assist' (v15) guarda una linea por CONSULTA.
KINDS = ("photo_traits", "triage", "bonebox", "bitemarks", "assist")
UNIT = {"triage": "observation_id", "assist": None}     # el resto: photo_id


def download_status(cfg: Config) -> dict:
    """Registros y fotos en las tablas, y fotos en disco (original, 512 y 1024 px)."""
    p = paths(cfg)
    obs = read_jsonl(p["tables"] / "observations.jsonl")
    pho = read_jsonl(p["tables"] / "photos.jsonl")
    out = {"run": Path(cfg.base_dir).name, "records": len(obs), "photos": len(pho)}
    names = set(pho["file_name"]) if len(pho) else set()
    stems = {Path(n).stem for n in names}
    for label, d in [("original", p["photos"]),
                     ("512px", p["base"] / "photos" / "512px"),
                     ("1024px", p["base"] / "photos" / "1024px")]:
        files = {f.name for f in d.glob("*") if f.suffix != ".part"} if d.exists() else set()
        have = names & files if label == "original" else stems & {Path(f).stem for f in files}
        out[f"photos_{label}"] = len(have)
    out["photos_missing"] = len(pho) - out["photos_original"]
    raw = list(p["raw"].glob("*.json"))
    out["raw_pages"] = len(raw)
    skipped = read_jsonl(p["logs"] / "skipped_existing.jsonl")
    out["skipped_in_other_runs"] = len(skipped)
    return out


def _kind(name: str) -> str | None:
    for k in sorted(KINDS, key=len, reverse=True):
        if name.startswith(k + "_"):
            return k
    return None


def model_status(cfg: Config) -> pd.DataFrame:
    """
    Una fila por archivo de outputs/ (paso + modelo + version de prompt):
    - done: registros (triaje) o fotos (resto) con al menos una respuesta sin error;
    - of: total de registros o fotos de la corrida (en bonebox, solo se piden fotos con 1 hueso);
    - records_done: registros con TODAS sus fotos hechas (pasos por foto);
    - errors_pending: con error y sin ninguna respuesta buena (se reintentan al relanzar);
    - repeated: con mas de una respuesta buena (redo, o corrida relanzada); vale la ultima.
    En assist (v15), done = consultas sin error y records = registros respondidos.
    """
    p = paths(cfg)
    pho = read_jsonl(p["tables"] / "photos.jsonl")
    obs = read_jsonl(p["tables"] / "observations.jsonl")
    n_photos_by_rec = pho.groupby("observation_id")["photo_id"].nunique() if len(pho) else pd.Series(dtype=int)
    rows = []
    out_dir = Path(cfg.base_dir) / "outputs"
    for f in sorted(out_dir.glob("*.jsonl")) if out_dir.exists() else []:
        kind = _kind(f.name)
        if kind is None:
            continue
        df = read_jsonl(f)
        if df.empty:
            continue
        version = (df["prompt_version"].dropna().mode().iloc[0] if "prompt_version" in df
                   and df["prompt_version"].notna().any() else f.stem.rsplit("_", 2)[-1])
        if kind == "assist":
            version = f.stem[f.stem.find("assist_v"):] if "assist_v" in f.stem else version
        model = df["model"].dropna().iloc[0] if "model" in df and df["model"].notna().any() else ""
        ok = df[df["error"].isna()] if "error" in df else df
        row = {"run": Path(cfg.base_dir).name, "step": kind, "version": version, "model": model,
               "lines": len(df), "last": str(df["saved_at"].max())[:16] if "saved_at" in df else ""}
        if kind == "assist":
            rec_ok = set()
            for x in ok.get("records_ok", pd.Series(dtype=object)).dropna():
                rec_ok |= {int(i) for i in x}
            # misma unidad que el triaje (registros); las consultas van aparte
            row.update(done=len(rec_ok), of=len(obs), unit="records", records_done=len(rec_ok),
                       calls=len(df),
                       errors_pending=int(df["error"].notna().sum()) if "error" in df else 0,
                       repeated=None,
                       tokens_in=int(df.get("input_tokens", pd.Series(dtype=float)).fillna(0).sum()),
                       tokens_out=int(df.get("output_tokens", pd.Series(dtype=float)).fillna(0).sum()))
            rows.append(row)
            continue
        unit = UNIT.get(kind, "photo_id")
        if unit not in df:
            continue
        good = ok[unit].dropna().astype("int64")
        all_units = set(df[unit].dropna().astype("int64"))
        done = set(good)
        counts = good.value_counts()
        row.update(unit="records" if unit == "observation_id" else "photos",
                   done=len(done),
                   of=len(obs) if unit == "observation_id" else (None if kind == "bonebox" else len(pho)),
                   errors_pending=len(all_units - done),
                   repeated=int((counts > 1).sum()))
        if unit == "photo_id" and len(pho) and kind != "bonebox":
            d = pho[pho["photo_id"].astype("int64").isin(done)].groupby("observation_id")["photo_id"].nunique()
            row["records_done"] = int((d == n_photos_by_rec.reindex(d.index)).sum())
        elif unit == "observation_id":
            row["records_done"] = len(done)
        tok = ok.drop_duplicates("call_id") if "call_id" in ok else ok
        if kind != "bonebox" or "call_id" not in ok:
            row["calls"] = (int(df["call_id"].nunique()) if "call_id" in df else len(df))
        if "source_prompt" in df and df["source_prompt"].notna().any():
            row["calls"] = None                 # lineas derivadas de v15: consultas en 'assist'
            tok = tok.iloc[0:0]
        row["tokens_in"] = int(tok.get("input_tokens", pd.Series(dtype=float)).fillna(0).sum())
        row["tokens_out"] = int(tok.get("output_tokens", pd.Series(dtype=float)).fillna(0).sum())
        rows.append(row)
    cols = ["run", "step", "version", "model", "unit", "done", "of", "pct", "records_done",
            "errors_pending", "repeated", "calls", "tokens_in", "tokens_out", "last"]
    t = pd.DataFrame(rows)
    if t.empty:
        return pd.DataFrame(columns=cols)
    t["pct"] = [f"{d / o:.0%}" if o == o and o else "" for d, o in zip(t["done"], t["of"])]
    if "calls" not in t:
        t["calls"] = None
    # consultas: en pasos por registro o foto, una linea buena = una respuesta (por registro en
    # photo_traits v5+: se cuentan los call_id distintos)
    t = t.reindex(columns=cols)
    t = t.sort_values(["run", "step", "version"]).reset_index(drop=True)
    for c in ("of", "records_done", "repeated", "calls"):
        t[c] = t[c].astype("Int64")             # enteros; vacio en vez de NaN
    return t


def annotation_status(cfg: Config) -> pd.DataFrame:
    """
    Avance de la anotacion, por anotador:
    - manual (AnnotationApp, annotations/): fotos revisadas (con individuos, solo conteo, sin
      individuos o repetidas), registros con todas sus fotos revisadas, individuos, con recuadro,
      prellenados desde la asistida, registros excluidos;
    - asistida (annotations_assisted/): fotos guardadas y cuantas quedan para AnnotationApp
      (conteo distinto de 1).
    """
    from .annotate import load_events, done_photos
    p = paths(cfg)
    pho = read_jsonl(p["tables"] / "photos.jsonl")
    obs = read_jsonl(p["tables"] / "observations.jsonl")
    n_by_rec = pho.groupby("observation_id")["photo_id"].nunique() if len(pho) else pd.Series(dtype=int)
    rows = []
    ev = load_events(cfg)
    done = done_photos(ev)
    if done:
        d = pd.DataFrame(list(done), columns=["observation_id", "photo_id"])
        per = d.groupby("observation_id")["photo_id"].nunique()
        ind = ev["individual"]
        rows.append({
            "run": Path(cfg.base_dir).name, "annotation": "manual (AnnotationApp)",
            "photos": len(done), "of": len(pho),
            "records_done": int((per == n_by_rec.reindex(per.index)).sum()),
            "records_of": len(obs), "individuals": len(ind),
            "with_box": int(ind["crop_box"].apply(lambda b: isinstance(b, dict)).sum())
            if "crop_box" in ind else 0,
            "from_assisted": int((ind.get("prefilled_from") == "assisted").sum())
            if "prefilled_from" in ind else 0,
            "excluded_records": len(ev["exclude_record"])})
    adir = p["base"] / "annotations_assisted"
    for f in sorted(adir.glob("*.jsonl")) if adir.exists() else []:
        a = read_jsonl(f)
        if a.empty:
            continue
        a = a.sort_values("saved_at").drop_duplicates("photo_id", keep="last")
        rows.append({"run": Path(cfg.base_dir).name, "annotation": f"assisted ({f.stem})",
                     "photos": len(a), "of": len(pho),
                     "records_done": None, "records_of": len(obs),
                     "individuals": int((a["n_cuttlebones_visible"] == 1).sum())
                     if "n_cuttlebones_visible" in a else None,
                     "with_box": int(a["crop_box"].apply(lambda b: isinstance(b, dict)).sum())
                     if "crop_box" in a else 0,
                     "from_assisted": None,
                     "excluded_records": None,
                     "needs_annotationapp": int(a.get("needs_manual_annotation",
                                                      pd.Series(dtype=bool)).fillna(False).sum())})
    t = pd.DataFrame(rows)
    if len(t):
        t.insert(3, "pct", [f"{x / o:.0%}" if o else "" for x, o in zip(t["photos"], t["of"])])
        for c in t.columns:
            if c not in ("run", "annotation", "pct"):
                t[c] = t[c].astype("Int64")
    return t


def repeated_units(cfg: Config, step: str, version: str) -> pd.DataFrame:
    """Registros o fotos con mas de una respuesta buena en esa version, con fechas."""
    out_dir = Path(cfg.base_dir) / "outputs"
    unit = UNIT.get(step, "photo_id")
    parts = []
    for f in out_dir.glob(f"{step}_*.jsonl"):
        df = read_jsonl(f)
        if df.empty or "prompt_version" not in df:
            continue
        parts.append(df[(df["prompt_version"] == version) & df["error"].isna()])
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    rep = df.groupby(unit).filter(lambda g: len(g) > 1)
    keep = [c for c in ("observation_id", "photo_id", "position", "saved_at", "call_id", "is_retry")
            if c in rep]
    return rep[keep].sort_values([unit, "saved_at"]).reset_index(drop=True)


def show_status(cfgs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Muestra (y devuelve) el estado de la descarga y de los modelos de una o varias corridas."""
    from IPython.display import display
    cfgs = cfgs if isinstance(cfgs, (list, tuple)) else [cfgs]
    dl = pd.DataFrame([download_status(c) for c in cfgs])
    ms = pd.concat([model_status(c) for c in cfgs], ignore_index=True)
    an = pd.concat([annotation_status(c) for c in cfgs], ignore_index=True)
    print("DESCARGA")
    display(dl)
    print("\nMODELOS · unit = en que se cuenta done/of: registros (triaje, v15) o fotos (v12-v15, "
          "recuadros) · records_done = registros con todas sus fotos · errors_pending = se "
          "reintentan al relanzar · repeated = respondidos mas de una vez (vale la ultima) · "
          "calls = consultas a la API (v15: en la fila assist)")
    display(ms.astype(object).fillna("") if len(ms) else ms)
    print("\nANOTACION · photos = fotos revisadas · records_done = registros con todas sus fotos "
          "revisadas a mano · with_box = con recuadro · from_assisted = prellenados desde la "
          "asistida · needs_annotationapp = guardadas en la asistida con conteo distinto de 1")
    display(an.astype(object).fillna("") if len(an) else an)
    ms.attrs["annotation"] = an
    rep = ms[ms["repeated"].fillna(0) > 0] if len(ms) else ms
    if len(rep):
        print("\nHay repetidos. Para verlos: sepia.status.repeated_units(cfg, '<step>', '<version>')")
    return dl, ms
