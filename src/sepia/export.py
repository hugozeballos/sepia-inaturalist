"""Excel de entrega con el formato de Martin (se puede generar en cualquier momento)."""
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

import pandas as pd

from .config import Config
from .io import paths, read_jsonl, _clean
from .inat import load_tables


# --------------------------------------------------------------------------
# 6. Excel de entrega (se puede generar en cualquier momento)
# --------------------------------------------------------------------------

# Columnas del Excel de Martin, en su orden: (encabezado en Excel, nombre interno)
EXCEL_COLUMNS = [
    ("n° observation", "obs_n"),
    ("ID", "observation_id"),
    ("site", "site"),
    ("latitude", "latitude"),
    ("longitude", "longitude"),
    ("place", "place_guess"),
    ("observed on", "observed_on"),
    ("url", "url"),
    ("individual (a,b,c,d,e)", "individual"),
    ("photo quality", "photo_quality"),
    ("state", "state"),
    ("number of pictures in record", "n_photos"),
    ("number individuals (cuttlebone)", "n_cuttlebones"),
    ("dorsal side (cuttlebone)", "dorsal_side"),
    ("ventral side (cuttlebone)", "ventral_side"),
    ("level of erosion", "erosion"),
    ("¿bitemarks?", "bitemarks"),
    ("¿epibionts?", "epibionts"),
    ("certainty (1 high, 0 low)", "certainty"),
    ("taxonomic groups", "taxonomic_groups"),
    ("number of barnacles (0; 1-10; 10-100; >100)", "barnacles_n"),
    ("aproximate size of barnacle (small, medium, large)", "barnacle_size"),
    ("possible genus/specie of balano", "barnacle_taxon"),
]

# Columnas de trazabilidad: van al final de la tabla entregada
TRACE_COLUMNS = ["photo_ids", "crop_box", "label_source", "model_version", "model_prob",
                 "reviewed", "annotator", "round", "validation_set",
                 "possible_duplicate_of", "excluded_reason", "download_date"]

# Leyenda (definiciones de la hoja 'legend' de Martin + columnas nuevas)
LEGEND = {
    "n° observation": "observation number in ascending order with respect to the site",
    "ID": "iNaturalist observation ID",
    "site": "sector defined by the authors (A-E, see sector map)",
    "latitude": "coordinates from iNaturalist (decimal degrees, unmodified)",
    "longitude": "coordinates from iNaturalist (decimal degrees, unmodified)",
    "place": "description of the location recorded by the iNaturalist observer",
    "observed on": "observation date",
    "url": "iNaturalist observation URL",
    "individual (a,b,c,d,e)": "individual within the record, labelled clockwise from the upper left",
    "photo quality": "high (not covered by sand or seaweeds), medium (most details visible, moderate cover), low (blurry or covered, details difficult to distinguish)",
    "state": "cuttlebone, egg case or live animal",
    "number of pictures in record": "number of photos in the iNaturalist record",
    "number individuals (cuttlebone)": "number of cuttlebones in the record",
    "dorsal side (cuttlebone)": "1 if this row describes the dorsal side",
    "ventral side (cuttlebone)": "1 if this row describes the ventral side",
    "level of erosion": "1 (almost intact), 2 (moderate erosion, some pieces missing), 3 (high erosion, at least 10% missing)",
    "¿bitemarks?": "presence of bird peck marks: yes, no, (yes) = probable",
    "¿epibionts?": "presence of organisms attached to the bone",
    "certainty (1 high, 0 low)": "certainty of the row's classification",
    "taxonomic groups": "taxonomic group of attached organisms, mainly barnacles",
    "number of barnacles (0; 1-10; 10-100; >100)": "barnacle abundance class",
    "aproximate size of barnacle (small, medium, large)": "approximate barnacle size",
    "possible genus/specie of balano": "genus or species of Balanidae, if the photo allows",
    "photo_ids": "iNaturalist photo IDs that support the row",
    "crop_box": "position of the individual in the photo",
    "label_source": "who assigned the labels: gpt, model, annotator",
    "model_version": "model or prompt version used",
    "model_prob": "model probability per variable",
    "reviewed": "whether a person reviewed the row",
    "annotator": "initials of the annotator or reviewer",
    "round": "annotation round (1, 2, 3, full)",
    "validation_set": "whether the row belongs to the fixed validation set",
    "possible_duplicate_of": "observation ID of a possible duplicate by another user",
    "excluded_reason": "reason for excluding the record",
    "download_date": "date the record was downloaded from iNaturalist",
}

ANNOTATION_KEY = ["observation_id", "individual", "side"]


def load_annotations(cfg: Config) -> pd.DataFrame:
    """
    Junta todas las anotaciones (annotations/*.jsonl). Cada linea es un
    individuo x lado. Si una misma clave aparece varias veces, vale la ultima
    guardada (columna 'saved_at').
    """
    p = paths(cfg)
    frames = [read_jsonl(f) for f in sorted(p["annotations"].glob("*.jsonl"))]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=ANNOTATION_KEY)
    ann = pd.concat(frames, ignore_index=True)
    for k in ANNOTATION_KEY:
        if k not in ann:
            ann[k] = None
    if "saved_at" in ann:
        ann = ann.sort_values("saved_at")
    return ann.drop_duplicates(subset=ANNOTATION_KEY, keep="last")


# Estado del triaje -> palabras de la hoja 'legend' de Martin
STATE_WORDS = {"cuttlebone": "cuttlebone", "live_animal": "live animal",
               "egg_case": "egg case", "mixed": "mixed", "other": "other",
               "unclear": "unclear"}


def triage_for_excel(cfg: Config) -> pd.DataFrame:
    """
    Resultados del triaje (modelo y version de prompt de cfg.triage) listos para
    unir al Excel: una fila por registro con state, n_cuttlebones y trazabilidad.
    Devuelve una tabla vacia si no hay seccion 'triage' o aun no hay resultados.
    """
    cols = ["observation_id", "state_gpt", "n_cuttlebones_gpt", "model_version_gpt",
            "model_prob_gpt"]
    if not cfg.triage:
        return pd.DataFrame(columns=cols)
    from .triage import load_triage
    t = cfg.triage
    tri = load_triage(cfg, t["provider"], t["model"], t.get("prompt_version"))
    if tri.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({"observation_id": tri["observation_id"]})
    out["state_gpt"] = tri["record_state"].map(STATE_WORDS)
    out["n_cuttlebones_gpt"] = tri["n_cuttlebones"]
    out["model_version_gpt"] = f'{t["model"]}/{t.get("prompt_version", "")}'
    # El modelo no da probabilidades: se guarda su confianza y si el conteo es exacto
    out["model_prob_gpt"] = [
        json.dumps({"confidence": c, "count_possible": bool(cp)})
        for c, cp in zip(tri.get("confidence"), tri.get("count_possible"))]
    return out


def build_excel(cfg: Config, out_name: str | None = None) -> Path:
    """
    Genera el Excel de entrega con el formato de Martin, en cualquier momento.
    - Una fila por individuo x lado (o una por registro si aun no hay anotaciones).
    - Solo valores: sin formulas, coordenadas numericas.
    - Hojas: identification, legend, excluded.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    p = paths(cfg)
    obs, _ = load_tables(cfg)
    if obs.empty:
        raise RuntimeError("No hay observations.jsonl: correr build_tables primero")

    manifest = p["base"] / "manifest.json"
    if manifest.exists():
        obs["download_date"] = json.loads(manifest.read_text())["written_at"][:10]

    # Registros excluidos van a su propia hoja
    if "excluded_reason" in obs:
        excluded = obs[obs["excluded_reason"].notna() & (obs["excluded_reason"] != "")]
        obs = obs.drop(excluded.index)
    else:
        excluded = obs.iloc[0:0]

    # Unir las columnas del registro con las anotaciones (join, nunca copiando celdas)
    ann = load_annotations(cfg)
    rec_cols = [c for c in obs.columns if c not in ann.columns or c == "observation_id"]
    table = obs[rec_cols].merge(ann, on="observation_id", how="left")

    # Triaje (GPT): completa state y n_cuttlebones SOLO donde no hay anotacion manual.
    # La anotacion de una persona siempre tiene prioridad sobre el modelo.
    tri = triage_for_excel(cfg)
    table = table.merge(tri, on="observation_id", how="left")
    for c in ["state", "n_cuttlebones", "label_source", "model_version", "model_prob",
              "reviewed"]:
        if c not in table:
            table[c] = None
        table[c] = table[c].astype(object)
    used_gpt = (table["state"].isna() & table["state_gpt"].notna()) | \
               (table["n_cuttlebones"].isna() & table["n_cuttlebones_gpt"].notna())
    for c in ["state", "n_cuttlebones"]:
        fill = table[c].isna() & table[f"{c}_gpt"].notna()
        table.loc[fill, c] = table.loc[fill, f"{c}_gpt"]
    no_source = table["label_source"].isna()
    table.loc[used_gpt & no_source, "label_source"] = "gpt"
    table.loc[used_gpt & no_source, "reviewed"] = False
    for c in ["model_version", "model_prob"]:
        fill = used_gpt & table[c].isna()
        table.loc[fill, c] = table.loc[fill, f"{c}_gpt"]
    table = table.drop(columns=[c for c in table.columns if c.endswith("_gpt")])

    # side -> las dos columnas del Excel de Martin
    side = table.get("side", pd.Series(index=table.index, dtype=object))
    table["dorsal_side"] = (side == "dorsal").map({True: 1, False: None})
    table["ventral_side"] = (side == "ventral").map({True: 1, False: None})

    # Orden: sector, registro, individuo, lado. n° observation correlativo por sector
    if "site" not in table:
        table["site"] = None
    table = table.sort_values(["site", "observation_id", "individual", "side"],
                              na_position="last").reset_index(drop=True)
    order = (table.drop_duplicates("observation_id")
             .assign(obs_n=lambda d: d.groupby("site", dropna=False).cumcount() + 1)
             .set_index("observation_id")["obs_n"])
    table["obs_n"] = table["observation_id"].map(order)

    # Fecha como fecha real de Excel (no texto)
    table["observed_on"] = pd.to_datetime(table["observed_on"], errors="coerce").dt.date

    headers = [h for h, _ in EXCEL_COLUMNS] + TRACE_COLUMNS
    internal = [c for _, c in EXCEL_COLUMNS] + TRACE_COLUMNS

    wb = Workbook()
    ws = wb.active
    ws.title = "identification"
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for rec in table.to_dict(orient="records"):
        values = []
        for c in internal:
            v = _clean(rec.get(c))
            if isinstance(v, (list, dict)):  # p. ej. photo_ids o crop_box
                v = json.dumps(v, ensure_ascii=False)
            values.append(v)
        ws.append(values)
    # Coordenadas: numeros con formato fijo de 10 decimales (solo visual, el valor no cambia)
    for col_name in ("latitude", "longitude"):
        col = get_column_letter(headers.index(col_name) + 1)
        for cell in ws[col][1:]:
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.0000000000"
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    lg = wb.create_sheet("legend")
    lg.append(["column", "description"])
    for h in headers:
        lg.append([h, LEGEND.get(h, "")])

    ex = wb.create_sheet("excluded")
    ex_cols = ["observation_id", "url", "observed_on", "latitude", "longitude", "excluded_reason"]
    ex.append(ex_cols)
    for r in excluded.itertuples(index=False):
        ex.append([_clean(getattr(r, c, None)) for c in ex_cols])

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = p["exports"] / (out_name or f"sepia_inat_{stamp}.xlsx")
    wb.save(out)
    print(f"Excel: {out} | {len(table)} filas | {len(excluded)} excluidos")
    return out
