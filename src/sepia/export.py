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
TRACE_COLUMNS = ["photo_id", "photo_position", "n_in_photo", "crop_box", "label_source", "model_version", "model_prob",
                 "reviewed", "annotator", "round", "validation_set",
                 "possible_duplicate_of", "excluded_reason", "download_date", "notes"]

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
    "individual (a,b,c,d,e)": "individual within the record, labelled clockwise from the upper left in the first photo where it appears; the same letter is kept in all photos of the record",
    "photo quality": "high (not covered by sand or seaweeds), medium (most details visible, moderate cover), low (blurry or covered, details difficult to distinguish). Operational rule: high = whole bone visible, in focus, nothing on top; medium = small areas hidden; low = more than half hidden, or blurry, or too small to see details",
    "state": "cuttlebone, egg case or live animal",
    "number of pictures in record": "number of photos in the iNaturalist record",
    "number individuals (cuttlebone)": "number of distinct cuttlebones in the record (distinct letters, not rows)",
    "dorsal side (cuttlebone)": "1 if this row describes the dorsal side",
    "ventral side (cuttlebone)": "1 if this row describes the ventral side",
    "level of erosion": "1 (almost intact), 2 (moderate erosion, some pieces missing), 3 (high erosion, at least 10% missing). Operational rule: 1 = outline complete; 2 = pieces missing, <10% in total; 3 = 10% or more missing, incl. fragments",
    "¿bitemarks?": "presence of bird peck marks: yes, no, (yes) = probable",
    "¿epibionts?": "presence of organisms attached to the bone",
    "certainty (1 high, 0 low)": "certainty of the row's classification",
    "taxonomic groups": "taxonomic group of attached organisms, mainly barnacles",
    "number of barnacles (0; 1-10; 10-100; >100)": "barnacle abundance class",
    "aproximate size of barnacle (small, medium, large)": "approximate barnacle size",
    "possible genus/specie of balano": "genus or species of Balanidae, if the photo allows",
    "photo_id": "iNaturalist photo ID of the row (one row per photo and individual)",
    "photo_position": "position of the photo in the iNaturalist record (1 = first)",
    "n_in_photo": "number of cuttlebones visible in the photo, annotated in detail or not (annotator)",
    "crop_box": "position of the individual in the photo",
    "label_source": "who assigned the labels: gpt, model, annotator",
    "model_version": "model or prompt version used",
    "model_prob": "model probability per variable",
    "reviewed": "whether a person reviewed the row",
    "annotator": "initials of the annotator or reviewer",
    "round": "annotation round (1, 2, 3, full)",
    "validation_set": "whether the row belongs to the fixed validation set",
    "possible_duplicate_of": "same bone in another iNaturalist record (same or other user): observation ID, optionally :letter there",
    "excluded_reason": "reason for excluding the record",
    "download_date": "date the record was downloaded from iNaturalist",
    "notes": "free-text remarks of the annotator (doubts, unusual cases)",
}

# Una anotacion = un individuo en una foto. El lado es un atributo de esa fila
# (un hueso muestra un solo lado por foto), no parte de la clave.
ANNOTATION_KEY = ["observation_id", "photo_id", "individual"]


def load_annotations(cfg: Config) -> pd.DataFrame:
    """
    Individuos anotados (una fila por observation_id + photo_id + individual), con
    'vale la ultima' ya aplicado y sin los borrados. Ver annotate.load_events.
    """
    from .annotate import load_events
    return load_events(cfg)["individual"]

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
    - Una fila por foto x individuo (una por foto si la foto aun no esta anotada).
    - Solo valores: sin formulas, coordenadas numericas.
    - Hojas: identification, legend, excluded.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    p = paths(cfg)
    obs, pho = load_tables(cfg)
    if obs.empty:
        raise RuntimeError("No hay observations.jsonl: correr build_tables primero")

    manifest = p["base"] / "manifest.json"
    if manifest.exists():
        obs["download_date"] = json.loads(manifest.read_text())["written_at"][:10]

    # Eventos de anotacion: individuos, fotos sin individuos y registros excluidos
    from .annotate import load_events
    events = load_events(cfg)
    ex_ann = events["exclude_record"]
    if len(ex_ann):
        reasons = ex_ann.set_index("observation_id")["excluded_reason"]
        if "excluded_reason" not in obs:
            obs["excluded_reason"] = None
        obs["excluded_reason"] = obs["observation_id"].map(reasons).fillna(obs["excluded_reason"])

    # Registros excluidos van a su propia hoja
    if "excluded_reason" in obs:
        excluded = obs[obs["excluded_reason"].notna() & (obs["excluded_reason"] != "")]
        obs = obs.drop(excluded.index)
    else:
        excluded = obs.iloc[0:0]

    # Esqueleto: una fila por foto (registro x foto). Join, nunca copiando celdas.
    photos = (pho[["observation_id", "photo_id", "position"]]
              .rename(columns={"position": "photo_position"}) if len(pho) else
              pd.DataFrame(columns=["observation_id", "photo_id", "photo_position"]))
    obs = obs.drop(columns=[c for c in ("photo_ids",) if c in obs])
    skeleton = obs.merge(photos, on="observation_id", how="left")
    # Conteo por foto (anotador): huesos que se ven, anotados en detalle o no
    info = events["photo_info"]
    if len(info):
        info = info[["observation_id", "photo_id", "n_in_photo"]].astype({"photo_id": "int64"})
        skeleton = skeleton.merge(info, on=["observation_id", "photo_id"], how="left")

    # Anotaciones: una fila por individuo en cada foto anotada
    ann = events["individual"]
    ann = ann.drop(columns=[c for c in ("kind", "deleted", "schema_version", "empty", "excluded_reason", "n_in_photo")
                          if c in ann])
    keep = ["observation_id", "photo_id"]
    rec_cols = [c for c in skeleton.columns if c not in ann.columns or c in keep]
    table = skeleton[rec_cols].merge(ann, on=keep, how="left")

    # Fotos marcadas "sin individuos" o con "solo conteo": fila revisada, sin individuo
    emp = (pd.concat([events["empty_photo"], events["photo_info"]], ignore_index=True)
           .drop_duplicates(subset=["observation_id", "photo_id"], keep="last"))
    if len(emp):
        is_empty = pd.MultiIndex.from_frame(table[["observation_id", "photo_id"]]).isin(
            pd.MultiIndex.from_frame(emp[["observation_id", "photo_id"]].astype("int64")))
        is_empty = is_empty & table["individual"].isna()
        for c, v in (("label_source", "annotator"), ("reviewed", True)):
            if c not in table:
                table[c] = None
            table[c] = table[c].astype(object)
            table.loc[is_empty, c] = v
        if "annotator" in emp:
            if "annotator" not in table:
                table["annotator"] = None
            amap = emp.set_index(["observation_id", "photo_id"])["annotator"]
            keys = list(zip(table["observation_id"], table["photo_id"]))
            table["annotator"] = table["annotator"].astype(object)
            table.loc[is_empty, "annotator"] = [amap.get(k) for k, e in zip(keys, is_empty) if e]

    # Filas anotadas por una persona: revisadas
    if "label_source" in table and "reviewed" in table:
        table.loc[(table["label_source"] == "annotator") & table["reviewed"].isna(), "reviewed"] = True

    # n_cuttlebones anotado = letras distintas del registro (no filas), salvo que
    # el anotador lo haya escrito explicitamente
    if "n_cuttlebones" not in table:
        table["n_cuttlebones"] = None
    letters = table.groupby("observation_id")["individual"].nunique()
    derived = table["observation_id"].map(letters).where(lambda x: x > 0)
    table["n_cuttlebones"] = table["n_cuttlebones"].astype(object)
    fill = table["n_cuttlebones"].isna() & derived.notna()
    table.loc[fill, "n_cuttlebones"] = derived[fill].astype(int)

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
    table = table.sort_values(["site", "observation_id", "photo_position", "individual"],
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
            if isinstance(v, (list, dict)):  # p. ej. crop_box
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
