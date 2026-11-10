"""Excel de entrega con el formato de Martin (se puede generar en cualquier momento)."""
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

import pandas as pd

from .config import Config
from .io import paths, _clean
from .inat import load_tables, load_raw, _coords


# --------------------------------------------------------------------------
# 6. Excel de entrega (se puede generar en cualquier momento)
# --------------------------------------------------------------------------

# Columnas del Excel de Martin, en su orden: (encabezado en Excel, nombre interno)
EXCEL_COLUMNS = [
    ("n° observation", "obs_n"),
    ("ID", "observation_id"),
    ("photo ID", "photo_id"),
    ("site", "site"),
    ("latitude", "latitude"),
    ("longitude", "longitude"),
    ("place", "place_guess"),
    ("observed on", "observed_on"),
    ("season", "season"),
    ("url", "url"),
    ("individual (a,b,c,d,e)", "individual"),
    ("photo quality", "photo_quality"),
    ("state", "state"),
    ("number of pictures in record", "n_photos"),
    ("number individuals (cuttlebone)", "n_cuttlebones"),          # en el REGISTRO
    ("number individuals in photo (cuttlebone)", "n_in_photo"),    # en la FOTO (6 nov)
    ("side (cuttlebone)", "side"),
    ("level of erosion", "erosion"),
    ("¿bitemarks?", "bitemarks"),
    ("¿epibionts?", "epibionts"),
    ("certainty (1 high, 0 low)", "certainty"),
    ("taxonomic groups", "taxonomic_groups"),
    ("number of barnacles (0; 1-10; 10-100; >100)", "barnacles_n"),
    ("aproximate size of barnacle (small, medium, large)", "barnacle_size"),
    ("possible genus/specie of balano", "barnacle_taxon"),
    # duplicados marcados por la persona (Hugo, 6 nov): al final, despues de las de Martin
    ("same photo as (photo ID)", "duplicate_of_photo"),
    ("same bone as other record (ID:letter)", "possible_duplicate_of"),
]

# Columnas de trazabilidad: van en la hoja 'traceability', con las mismas filas y en el mismo
# orden que 'identification' (que queda con las columnas de Martin, sin agregados).
# Las primeras tres repiten la clave de la fila para poder unir las dos hojas.
TRACE_KEY = ["n° observation", "ID", "individual (a,b,c,d,e)"]
TRACE_COLUMNS = ["run", "prefilled_from", "photo_id", "photo_position", "crop_box", "box_source",
                 "label_source", "reviewed", "annotator", "round",
                 "excluded_reason", "download_date", "notes"]
# (sin model_version, model_prob ni validation_set: el Excel lleva solo anotacion manual; 6 nov)

# Leyenda (definiciones de la hoja 'legend' de Martin + columnas nuevas)
LEGEND = {
    "n° observation": "observation number in ascending order with respect to the site",
    "ID": "iNaturalist observation ID",
    "site": "location defined by the authors for extracting data from iNaturalist (see details on the \u201cMap\u201d sheet)",
    "latitude": "coordinates extracted from iNaturalist based on the location of the observation (decimal degrees, unmodified)",
    "longitude": "coordinates extracted from iNaturalist based on the location of the observation (decimal degrees, unmodified)",
    "place": "description of the location recorded by the iNaturalist observer",
    "observed on": "observation date",
    "season": "season of the year, such as summer, spring, autumn, and winter",
    "url": "iNaturalist observation URL",
    "individual (a,b,c,d,e)": "individual within the record, labelled clockwise from the upper left in the first photo where it appears; the same letter is kept in all photos of the record",
    "photo quality": "high quality (good quality, not covered by sand or seaweeds), medium quality (most details can be seen, only moderate cover by sand or seaweeds), and low quality (blurry photo or covered by sand or seaweeds, details difficult to distinguish)",
    "state": "organism in the form of cuttlebone, eggs, or squid",
    "number of pictures in record": "number of photos in the iNaturalist record",
    "number individuals (cuttlebone)": "number of distinct cuttlebones in the whole RECORD (distinct letters with state cuttlebone, not rows); same value in all rows of the record",
    "number individuals in photo (cuttlebone)": "number of cuttlebones visible in THIS PHOTO (annotator's 'Individuals in photo'), annotated in detail or not; a bone counts if at least about half of it is visible",
    "photo ID": "iNaturalist photo ID: the row describes what is visible in this photo",
    "side (cuttlebone)": "side of the cuttlebone visible in this photo: dorsal, ventral or unclear (one column instead of the two columns 'dorsal side' / 'ventral side')",
    "level of erosion": "1 (almost perfect), 2 (somewhat eroded, with some pieces missing), and 3 (very eroded, at least 10% missing)",
    "¿bitemarks?": "presence of bird peck marks",
    "¿epibionts?": "presence of organisms that adhere to bone",
    "certainty (1 high, 0 low)": "certainty of the row's classification",
    "taxonomic groups": "taxonomic group of organisms that attach themselves to bone, mainly barnacles (crustaceans)",
    "number of barnacles (0; 1-10; 10-100; >100)": "barnacle abundance class",
    "aproximate size of barnacle (small, medium, large)": "approximate barnacle size",
    "possible genus/specie of balano": "if the photograph is good, the genus or species of the Balanidae family can be identified here",
    "same photo as (photo ID)": "this photo repeats another photo of the SAME record (photo ID of that photo): not annotated again",
    "same bone as other record (ID:letter)": "the same cuttlebone appears in ANOTHER iNaturalist record (same or other user): observation ID, optionally :letter there. Marked by a person, not by the duplicate search",
    "run": "download run (data folder) the row comes from, e.g. pilot_a1, a2_test",
    "prefilled_from": "empty = blind manual annotation; 'assisted' = the annotation started from an assisted review where the model outputs were visible",
    "photo_id": "iNaturalist photo ID of the row (one row per photo and individual)",
    "photo_position": "position of the photo in the iNaturalist record (1 = first)",
    "duplicate_of_photo": "this photo repeats another photo of the same record (its photo ID): not annotated",
    "n_in_photo": "number of cuttlebones visible in the photo, annotated in detail or not (annotator)",
    "crop_box": "position of the individual in the photo",
    "box_source": "origin of crop_box: model (suggested by bonebox_v1 and accepted after looking at the photo), model_edited (adjusted by the annotator), manual (drawn by the annotator)",
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

# Hojas de registros (A1, A2...): una fila por registro. Columnas de la hoja A1 de Martin, en
# su orden, + user_id, user_login, updated_at (de su hoja A2) + resumen de la anotacion manual.
RECORD_COLUMNS = ["id", "uuid", "observed_on_string", "observed_on", "time_observed_at",
                  "time_zone", "user_name", "created_at", "quality_grade", "license", "url",
                  "description", "place_guess", "latitude", "longitude", "positional_accuracy",
                  "private_latitude", "private_longitude", "public_positional_accuracy",
                  "geoprivacy", "taxon_geoprivacy", "coordinates_obscured", "positioning_method",
                  "positioning_device", "scientific_name", "common_name", "taxon_id",
                  "user_id", "user_login", "updated_at"]
RECORD_ANNOTATION = ["number of pictures in record", "photo IDs", "photos annotated",
                     "state (annotated)", "number individuals (cuttlebone)", "excluded_reason"]
RECORD_LEGEND = {
    "private_latitude": "true coordinates of obscured records: NOT available from the public iNaturalist API (only for the observer or curators), left empty",
    "private_longitude": "see private_latitude",
    "photo IDs": "iNaturalist photo IDs of the record, in their order in the record",
    "photos annotated": "photos of the record already annotated by a person / total photos",
    "state (annotated)": "states annotated in the record: cuttlebone, live animal, egg case, or mixed",
}
# Texto de la hoja 'map' de Martin, por nombre de hoja
SECTOR_TEXT = {"A1": "from Thurso to North Shields",
               "A2": "from South Shields to the mouth of River Yare",
               "A": "Western North Sea", "B": "North-Eastern British Channel",
               "C": "South-Eastern British Channel", "D": "Southern North Sea (Western Wadden Sea)",
               "E": "Eastern North Sea (North-Eastern Wadden Sea)"}


def _record_row(o: dict) -> dict:
    """Una fila de la hoja de registros desde la copia cruda de la API (iNaturalist v1)."""
    lat, lon = _coords(o)
    user, taxon = o.get("user") or {}, o.get("taxon") or {}
    return {"id": o.get("id"), "uuid": o.get("uuid"),
            "observed_on_string": o.get("observed_on_string"), "observed_on": o.get("observed_on"),
            "time_observed_at": o.get("time_observed_at"),
            "time_zone": o.get("observed_time_zone") or o.get("time_zone"),
            "user_name": user.get("name"), "created_at": o.get("created_at"),
            "quality_grade": o.get("quality_grade"), "license": o.get("license_code"),
            "url": f"https://www.inaturalist.org/observations/{o.get('id')}",
            "description": o.get("description"), "place_guess": o.get("place_guess"),
            "latitude": lat, "longitude": lon,
            "positional_accuracy": o.get("positional_accuracy"),
            "private_latitude": None, "private_longitude": None,
            "public_positional_accuracy": o.get("public_positional_accuracy"),
            "geoprivacy": o.get("geoprivacy"), "taxon_geoprivacy": o.get("taxon_geoprivacy"),
            "coordinates_obscured": o.get("obscured"),
            "positioning_method": o.get("positioning_method"),
            "positioning_device": o.get("positioning_device"),
            "scientific_name": taxon.get("name"), "common_name": taxon.get("preferred_common_name"),
            "taxon_id": taxon.get("id"), "user_id": user.get("id"), "user_login": user.get("login"),
            "updated_at": o.get("updated_at")}


def records_sheet(cfg: Config) -> pd.DataFrame:
    """
    Hoja de registros de una corrida: una fila por registro de sus tablas (tambien los
    excluidos, con el motivo), metadatos de iNaturalist + resumen de la anotacion manual.
    """
    from .annotate import load_events, done_photos
    obs, pho = load_tables(cfg)
    keep = set(obs["observation_id"].astype(int))
    raw = {int(o["id"]): o for o in load_raw(cfg)}
    rows = [_record_row(raw[i]) for i in sorted(keep) if i in raw]
    rec = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    ev = load_events(cfg)
    ind = ev["individual"]
    done = done_photos(ev)
    by_rec = pho.sort_values(["observation_id", "position"]).groupby("observation_id")["photo_id"]
    photo_ids = by_rec.apply(lambda s: ", ".join(str(int(x)) for x in s))
    n_photos = by_rec.size()
    n_done = pd.Series({oid: sum((int(oid), int(p)) in done for p in ps)
                        for oid, ps in by_rec}, dtype="int64")
    rec["number of pictures in record"] = rec["id"].map(n_photos)
    rec["photo IDs"] = rec["id"].map(photo_ids)
    rec["photos annotated"] = [f"{int(n_done.get(i, 0))}/{int(n_photos.get(i, 0))}"
                               for i in rec["id"]]
    if len(ind) and "state" in ind:
        states = ind.dropna(subset=["state"]).groupby("observation_id")["state"].agg(
            lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
        bone = ind["state"].isna() | (ind["state"] == "cuttlebone")   # igual que identification
        bones = ind[bone].groupby("observation_id")["individual"].nunique()
    else:
        states, bones = pd.Series(dtype=object), pd.Series(dtype="int64")
    rec["state (annotated)"] = rec["id"].map(states)
    rec["number individuals (cuttlebone)"] = rec["id"].map(bones)
    reasons = {}                               # motivo de exclusion: tabla y anotacion
    if "excluded_reason" in obs:
        reasons.update(obs.dropna(subset=["excluded_reason"])
                       .set_index("observation_id")["excluded_reason"].to_dict())
    exc = ev["exclude_record"]
    if len(exc):
        reasons.update(exc.set_index("observation_id")["excluded_reason"].to_dict())
    rec["excluded_reason"] = rec["id"].map(reasons)
    rec["observed_on"] = pd.to_datetime(rec["observed_on"], errors="coerce").dt.date
    return rec


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


def _excel_table(cfg: Config, use_triage: bool = False):
    """Tabla de una corrida para el Excel (una fila por foto x individuo) y sus excluidos."""
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
    dup = events["duplicate_photo"]
    if len(dup):
        dup = dup[["observation_id", "photo_id", "duplicate_of_photo"]].astype(
            {"photo_id": "int64", "duplicate_of_photo": "int64"})
        skeleton = skeleton.merge(dup, on=["observation_id", "photo_id"], how="left")

    # Anotaciones: una fila por individuo en cada foto anotada
    ann = events["individual"]
    ann = ann.drop(columns=[c for c in ("kind", "deleted", "schema_version", "empty", "excluded_reason", "n_in_photo",
                          "duplicate_of_photo")
                          if c in ann])
    keep = ["observation_id", "photo_id"]
    rec_cols = [c for c in skeleton.columns if c not in ann.columns or c in keep]
    table = skeleton[rec_cols].merge(ann, on=keep, how="left")

    # Fotos marcadas "sin individuos" o con "solo conteo": fila revisada, sin individuo
    emp = (pd.concat([events["empty_photo"], events["photo_info"], events["duplicate_photo"]],
                     ignore_index=True)
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

    # Fotos sin individuo (solo conteo, vacias, repetidas) que vienen del prellenado de la
    # asistida: la marca prefilled_from esta en su linea de conteo (photo_info)
    pinfo = events["photo_info"]
    if len(pinfo) and "prefilled_from" in pinfo:
        pre = pinfo.dropna(subset=["prefilled_from"]).astype({"photo_id": "int64"})
        pre = pre.set_index(["observation_id", "photo_id"])["prefilled_from"]
        if "prefilled_from" not in table:
            table["prefilled_from"] = None
        table["prefilled_from"] = table["prefilled_from"].astype(object)
        no_ind = table["individual"].isna() & table["prefilled_from"].isna()
        keys = list(zip(table["observation_id"], table["photo_id"]))
        table.loc[no_ind, "prefilled_from"] = [pre.get(k) for k, e in zip(keys, no_ind) if e]

    # Filas anotadas por una persona: revisadas
    if "label_source" in table and "reviewed" in table:
        table.loc[(table["label_source"] == "annotator") & table["reviewed"].isna(), "reviewed"] = True

    # n_cuttlebones anotado = letras distintas del registro con estado 'cuttlebone' (no filas;
    # vivos y puestas no cuentan), salvo que el anotador lo haya escrito explicitamente.
    # Lineas antiguas sin estado se cuentan como hueso (estado por defecto de la interfaz).
    if "n_cuttlebones" not in table:
        table["n_cuttlebones"] = None
    st = table["state"] if "state" in table else pd.Series(None, index=table.index, dtype=object)
    is_bone = st.isna() | (st == "cuttlebone")
    letters = table[is_bone].groupby("observation_id")["individual"].nunique()
    derived = table["observation_id"].map(letters).where(lambda x: x > 0)
    table["n_cuttlebones"] = table["n_cuttlebones"].astype(object)
    fill = table["n_cuttlebones"].isna() & derived.notna()
    table.loc[fill, "n_cuttlebones"] = derived[fill].astype(int)

    # Triaje (GPT): completa state y n_cuttlebones SOLO donde no hay anotacion manual.
    # La anotacion de una persona siempre tiene prioridad sobre el modelo.
    tri = (triage_for_excel(cfg) if use_triage else
           pd.DataFrame(columns=["observation_id", "state_gpt", "n_cuttlebones_gpt",
                                 "model_version_gpt", "model_prob_gpt"]))
    table = table.merge(tri.astype({"observation_id": table["observation_id"].dtype}),
                        on="observation_id", how="left")
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

    # Lado: UNA columna (dorsal / ventral / unclear), no las dos de Martin (Hugo, 6 nov)
    if "side" not in table:
        table["side"] = None

    # Pendientes: sector y estacion quedan en blanco (columna presente, como en la tabla de Martin)
    table["season"] = None
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

    return table, excluded


def build_excel(cfg, out_name: str | None = None, use_triage: bool = False) -> Path:
    """
    Genera el Excel de entrega con el formato de Martin, en cualquier momento.
    - cfg: una corrida o una lista de corridas (p. ej. [cfg_a1, cfg_a2]). Con varias, las
      filas de cada corrida van una debajo de la otra (en el orden de la lista) en UN solo
      Excel; las corridas no se modifican. Error si un registro esta en dos corridas.
    - Una fila por foto x individuo (una por foto si la foto aun no esta anotada).
    - Solo valores: sin formulas, coordenadas numericas.
    - Hojas: legend, identification (columnas de Martin + individual), traceability
      (mismas filas: run, prefilled_from, photo_id, ...), excluded.
    - Por defecto solo anotaciones manuales. use_triage=True completa state y n_cuttlebones
      con el triaje donde no hay anotacion (label_source = gpt).
    - season y site quedan en blanco por ahora (pendientes). n° observation es correlativo
      por sector (con site en blanco: sobre toda la tabla).
    - Una corrida: se guarda en exports/ de la corrida. Varias: en <carpeta del proyecto>/exports/
      (junto a data/).
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    cfgs = list(cfg) if isinstance(cfg, (list, tuple)) else [cfg]
    tables, excl = [], []
    for k, c in enumerate(cfgs):
        name = Path(c.base_dir).name
        t, ex = _excel_table(c, use_triage)
        tables.append(t.assign(run=name, _run_order=k))
        excl.append(ex.assign(run=name))
    seen = {}
    for t in tables:
        for oid in t["observation_id"].dropna().astype(int).unique():
            if oid in seen and seen[oid] != t["run"].iloc[0]:
                raise ValueError(f"el registro {oid} esta en dos corridas ({seen[oid]} y "
                                 f"{t['run'].iloc[0]}): revisar skip_ids_from")
            seen[oid] = t["run"].iloc[0]
    table = pd.concat(tables, ignore_index=True)
    excluded = pd.concat(excl, ignore_index=True) if excl else pd.DataFrame()
    # Orden y n° observation sobre la tabla completa (correlativo por sector)
    table = table.sort_values(["site", "_run_order", "observation_id", "photo_position",
                               "individual"], na_position="last").reset_index(drop=True)
    order = (table.drop_duplicates("observation_id")
             .assign(obs_n=lambda d: d.groupby("site", dropna=False).cumcount() + 1)
             .set_index("observation_id")["obs_n"])
    table["obs_n"] = table["observation_id"].map(order)
    if "prefilled_from" not in table:
        table["prefilled_from"] = None

    headers = [h for h, _ in EXCEL_COLUMNS]
    internal = [c for _, c in EXCEL_COLUMNS]
    to_internal = dict(EXCEL_COLUMNS)
    trace_headers = TRACE_KEY + TRACE_COLUMNS
    trace_internal = [to_internal[h] for h in TRACE_KEY] + TRACE_COLUMNS

    def _value(rec, c):
        v = _clean(rec.get(c))
        if isinstance(v, (list, dict)):  # p. ej. crop_box
            v = json.dumps(v, ensure_ascii=False)
        return v

    wb = Workbook()
    lg = wb.active
    lg.title = "legend"                       # primera hoja, como en la tabla de Martin
    mp = wb.create_sheet("map")
    ws = wb.create_sheet("identification")
    # hojas de registros (A1, A2...), una por corrida, despues de identification (como Martin)
    rec_sheets = []
    for c in cfgs:
        name = c.sheet_name or Path(c.base_dir).name
        rs = wb.create_sheet(name[:31])
        rec = records_sheet(c)
        cols = RECORD_COLUMNS + RECORD_ANNOTATION
        rs.append(cols)
        for row in rec[cols].to_dict(orient="records"):
            rs.append([_clean(row[k]) for k in cols])
        for cell in rs[1]:
            cell.font = Font(bold=True)
        for col_name in ("latitude", "longitude"):
            col = get_column_letter(cols.index(col_name) + 1)
            for cell in rs[col][1:]:
                if isinstance(cell.value, (int, float)):
                    cell.number_format = "0.0000000000"
        rs.freeze_panes = "A2"
        rs.auto_filter.ref = rs.dimensions
        rec_sheets.append((name, c, len(rec)))
    mp.append(["sheet", "run", "sector (Martin)", "download rectangle [south, west, north, east]",
               "records"])
    for cell in mp[1]:
        cell.font = Font(bold=True)
    for name, c, n in rec_sheets:
        mp.append([name, Path(c.base_dir).name, SECTOR_TEXT.get(name, ""),
                   json.dumps(list(c.bbox)) if c.bbox else "", n])
    mp.append([])
    mp.append(["Note: column 'site' = sector A-E from Martin's slides (SepiaFORhugo21oct2026.pptx, "
               "slides 5-12): rectangle minus the areas marked 'do not include' (configs/sectors.yaml). "
               "North-east corners are exact (iNaturalist URL); south-west corners were estimated from "
               "the screenshots (pending: exact URLs). Runs pilot_a1 and a2_test used approximate "
               "download rectangles; their records are assigned to sectors with the same rule."])
    mp.append([])
    mp.append(["sector", "label", "rectangle [south, west, north, east]", "excluded areas"])
    from .sectors import load_sectors
    for s in load_sectors(cfgs[0].sectors_file):
        mp.append([s["name"], s.get("label", ""), json.dumps(list(s["bbox"])),
                   ", ".join(s["exclude"]) or "-"])
    tr = wb.create_sheet("traceability")
    ws.append(headers)
    tr.append(trace_headers)
    for sheet in (ws, tr):
        for cell in sheet[1]:
            cell.font = Font(bold=True)
    for rec in table.to_dict(orient="records"):
        ws.append([_value(rec, c) for c in internal])
        tr.append([_value(rec, c) for c in trace_internal])
    # Coordenadas: numeros con formato fijo de 10 decimales (solo visual, el valor no cambia)
    for col_name in ("latitude", "longitude"):
        col = get_column_letter(headers.index(col_name) + 1)
        for cell in ws[col][1:]:
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.0000000000"
    for sheet in (ws, tr):
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

    lg.append(["column", "description"])
    for h in headers + TRACE_COLUMNS:
        lg.append([h, LEGEND.get(h, "")])
    lg.append([])
    lg.append(["Record sheets (" + ", ".join(n for n, _, _ in rec_sheets) + "): one row per "
               "iNaturalist record, columns as exported by iNaturalist + annotation summary", ""])
    for h in RECORD_COLUMNS + RECORD_ANNOTATION:
        if h in RECORD_LEGEND:
            lg.append([h, RECORD_LEGEND[h]])

    ex = wb.create_sheet("excluded")
    ex_cols = ["run", "observation_id", "url", "observed_on", "latitude", "longitude",
               "excluded_reason"]
    ex.append(ex_cols)
    for r in excluded.itertuples(index=False):
        ex.append([_clean(getattr(r, c, None)) for c in ex_cols])

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    if len(cfgs) == 1:
        out_dir = paths(cfgs[0])["exports"]
    else:
        out_dir = Path(cfgs[0].base_dir).parent.parent / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (out_name or f"sepia_inat_{stamp}.xlsx")
    wb.save(out)
    print(f"Excel: {out} | {len(table)} filas | {len(excluded)} excluidos")
    return out
