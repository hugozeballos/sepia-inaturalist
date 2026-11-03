"""
Revision persona vs modelos: fotos donde la anotacion manual no coincide con las versiones
de prompt de photo_traits. El objetivo es encontrar errores de anotacion (o confirmar errores
del modelo) y dejar constancia de cada decision.

Pasos (Colab):
    from sepia.disagree import rank_versions, DisagreementApp, verdict_summary
    table, top = rank_versions(cfg)                  # 1. metricas de todas las versiones, top 5
    DisagreementApp(cfg, top, annotator="HZ").show() # 2. revisar todas las fotos anotadas
    verdict_summary(cfg, "HZ")                       # 3. resumen de veredictos

Los veredictos se guardan en reviews/human_vs_models_<anotador>.jsonl (solo se agrega; vale la
ultima linea por foto y campo; field='_photo' marca la foto como revisada). NO se guardan en annotations/: load_events() lee todo lo que hay
ahi y los trataria como anotaciones. Esta herramienta no cambia ninguna anotacion; las
correcciones se hacen despues en AnnotationApp.

Cuidado (sesgo): por defecto se revisan todas las fotos anotadas, pero mirando las respuestas
de los modelos. Si se corrigen anotaciones con esto, el acuerdo modelo-persona sube, sobre todo
con las versiones elegidas por parecerse a la persona. Por eso se guarda cada veredicto y la
anotacion original sigue en el archivo de anotaciones.
"""
from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import pandas as pd

from .config import Config
from .io import paths, read_jsonl, append_jsonl
from .photo_traits import load_photo_traits, compare_photo_traits
from .review import review_table, _norm

# Campos que se comparan con la anotacion manual (lado, erosion y calidad: solo fotos con
# 1 individuo anotado, como en review_table y evaluate_photo_traits)
COMPARE_FIELDS = ["n_cuttlebones_visible", "side", "erosion", "photo_quality"]
# Campos que solo se muestran (ayudan a entender por que decidio el modelo)
INFO_FIELDS = ["side_basis", "side_signs", "side_evidence", "notes"]
# Valores del modelo que no cuentan como desacuerdo: no aplica o se abstuvo
NOT_COMPARED = {"", "not_applicable", "not_assessable", "unclear"}
# Metricas que entran al puntaje para elegir las mejores versiones (promedio simple)
SCORE_METRICS = ["count_agreement", "side_agreement", "erosion_agreement",
                 "photo_quality_agreement"]
PHOTO_CHECKED = "_photo"      # linea de verdicts que marca la foto como revisada
ORDERS = ["record", "most_disagreement"]
VERDICTS = [("—", "none"), ("My error", "human_error"), ("Model error", "model_error"),
            ("Unsure / ask Martin", "ask_martin")]


# --------------------------------------------------------------------------
# 1. Metricas de todas las versiones y eleccion de las mejores
# --------------------------------------------------------------------------

def photo_traits_versions(cfg: Config, max_version: int = 30) -> list[str]:
    """Versiones de photo_traits que tienen resultados guardados (v1, v2, ...)."""
    out = []
    for i in range(1, max_version + 1):
        v = f"photo_traits_v{i}"
        if not load_photo_traits(cfg, v).empty:
            out.append(v)
    return out


def rank_versions(cfg: Config, versions: list[str] | None = None, n_top: int = 5,
                  save: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Evalua todas las versiones sobre las MISMAS fotos (compare_photo_traits) y las ordena por
    el promedio de acuerdo en conteo, lado, erosion y calidad.
    - side_agreement cuenta 'unclear' del modelo como no acierto: castiga a las versiones que
      se abstienen (v12). Por eso tambien se muestra side_agreement_when_decided.
    - save=True agrega una linea por version a outputs/metrics/photo_traits_metrics.jsonl.
    Devuelve (tabla ordenada, lista de las n_top mejores).
    """
    versions = versions or photo_traits_versions(cfg)
    print(f"Versiones con resultados: {', '.join(v.replace('photo_traits_', '') for v in versions)}")
    table = compare_photo_traits(cfg, versions, save=save)
    cols = [c for c in SCORE_METRICS if c in table]
    table["score"] = table[cols].astype(float).mean(axis=1)
    table = table.sort_values("score", ascending=False)
    top = list(table.index[:n_top])
    print("\nOrdenadas por score (promedio de " + ", ".join(cols) + "):")
    show = ["score"] + cols + [c for c in ("side_agreement_when_decided", "side_unclear_model",
                                           "side_n", "count_n") if c in table]
    print(table[show].round(2).to_string())
    print(f"\nMejores {n_top}: {', '.join(top)}")
    return table, top


# --------------------------------------------------------------------------
# 2. Tabla de desacuerdos
# --------------------------------------------------------------------------

def disagreement_table(cfg: Config, versions: list[str]) -> pd.DataFrame:
    """
    Una fila por foto anotada: valores de cada version y de la persona (review_table) y,
    por campo comparado:
      n_models__<campo>    versiones con un valor comparable (no vacio, no unclear/not_applicable)
      n_disagree__<campo>  de esas, cuantas difieren de la persona
    n_disagree_max = maximo entre campos (para ordenar).
    """
    t = review_table(cfg, versions)
    t = t[t["n_cuttlebones_visible__human"].notna()].copy()    # solo fotos anotadas
    for f in COMPARE_FIELDS:
        h = t[f"{f}__human"].map(_norm) if f"{f}__human" in t else pd.Series("", index=t.index)
        n_mod = pd.Series(0, index=t.index)
        n_dis = pd.Series(0, index=t.index)
        for v in versions:
            col = f"{f}__{v}"
            if col not in t:
                continue
            m = t[col].map(_norm)
            ok = (h != "") & ~m.isin(NOT_COMPARED)
            n_mod += ok.astype(int)
            n_dis += (ok & (m != h)).astype(int)
        t[f"n_models__{f}"] = n_mod
        t[f"n_disagree__{f}"] = n_dis
    t["n_disagree_max"] = t[[f"n_disagree__{f}" for f in COMPARE_FIELDS]].max(axis=1)
    return t.sort_values(["observation_id", "position"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Veredictos (archivo propio, solo se agrega)
# --------------------------------------------------------------------------

def verdict_file(cfg: Config, annotator: str) -> Path:
    d = paths(cfg)["base"] / "reviews"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"human_vs_models_{annotator}.jsonl"


def load_verdicts(cfg: Config, annotator: str) -> pd.DataFrame:
    """Ultimo veredicto por foto y campo ('none' = se borro el veredicto)."""
    v = read_jsonl(verdict_file(cfg, annotator))
    if v.empty:
        return v
    v = v.sort_values("saved_at", kind="stable").drop_duplicates(["photo_id", "field"], keep="last")
    return v[v["verdict"] != "none"].reset_index(drop=True)


def verdict_summary(cfg: Config, annotator: str) -> pd.DataFrame:
    """Conteo de veredictos por campo y lista de registros a corregir en AnnotationApp."""
    v = load_verdicts(cfg, annotator)
    if v.empty:
        print("Sin veredictos todavia")
        return v
    checked = v[v["field"] == PHOTO_CHECKED]
    v = v[v["field"] != PHOTO_CHECKED]
    print(f"Fotos revisadas: {checked['photo_id'].nunique()}")
    if v.empty:
        print("Sin veredictos de error todavia")
        return v
    print(pd.crosstab(v["field"], v["verdict"], margins=True))
    fix = v[v["verdict"] == "human_error"]
    if len(fix):
        ids = sorted(set(fix["observation_id"].astype(int)))
        print(f"\nRegistros con errores de anotacion ({len(ids)}): {ids}")
        print(f"Corregir con: AnnotationApp(cfg, '{annotator}', ids={ids}).show()")
    ask = v[v["verdict"] == "ask_martin"]
    if len(ask):
        print(f"\nPara preguntar a Martin ({len(ask)}): "
              f"{sorted(set(ask['observation_id'].astype(int)))}")
    return v


# --------------------------------------------------------------------------
# 4. Interfaz
# --------------------------------------------------------------------------

class DisagreementApp:
    """
    Todas las fotos anotadas, en orden de registro (o de mas a menos desacuerdo).
    min_disagree > 0 deja solo las fotos donde al menos esa cantidad de versiones difieren.
    Foto grande + miniaturas del registro (clic = ver esa foto en grande) + tabla de valores
    por version y persona + resumen del registro (lado anotado y del modelo en cada foto)
    + veredicto por campo.
    """

    def __init__(self, cfg: Config, versions: list[str], annotator: str = "HZ",
                 min_disagree: int = 0, field: str = "any", hide_reviewed: bool = False,
                 order: str = "record",
                 image_width: str = "560px", thumb_px: str = "130px"):
        import ipywidgets as w
        self.w, self.cfg, self.versions, self.annotator = w, cfg, versions, annotator
        base = paths(cfg)["base"] / "photos"
        self.big_dirs = [base / "1024px", base / "original"]
        self.thumb_dirs = [base / "512px"] + self.big_dirs
        self.thumb_px = thumb_px
        self.full = disagreement_table(cfg, versions)
        # todas las fotos (tambien sin anotar) para la tira de miniaturas
        self.all_photos = review_table(cfg, versions)
        self.individuals = self._load_individuals()
        self._cache = {}

        n = len(versions)
        self.field = w.Dropdown(options=["any"] + COMPARE_FIELDS, value=field,
                                description="Field", layout=w.Layout(width="260px"))
        self.order = w.Dropdown(options=ORDERS, value=order, description="Order",
                                layout=w.Layout(width="260px"))
        self.min_dis = w.IntSlider(value=min(min_disagree, n), min=0, max=n,
                                   description=f"Models ≠ me (of {n})",
                                   style={"description_width": "initial"})
        self.hide = w.Checkbox(value=hide_reviewed, description="Hide reviewed", indent=False)
        for c in (self.field, self.order, self.min_dis, self.hide):
            c.observe(lambda _: self._apply_filter(), names="value")
        prev_b, next_b = w.Button(description="◀ Prev"), w.Button(description="Next ▶")
        prev_b.on_click(lambda _: self._go(self.i - 1))
        next_b.on_click(lambda _: self._go(self.i + 1))
        self.jump = w.Text(placeholder="iNaturalist ID", layout=w.Layout(width="150px"))
        go_b = w.Button(description="Go to ID")
        go_b.on_click(lambda _: self._jump())

        self.header = w.HTML()
        self.image = w.Image(format="jpeg", layout=w.Layout(max_width=image_width,
                                                            max_height="560px",
                                                            object_fit="contain"))
        self.image_label = w.HTML()
        self.strip = w.HBox(layout=w.Layout(flex_flow="row wrap"))
        self.table = w.HTML()
        self.record = w.HTML()
        self.verdict_box = w.VBox()
        self.msg = w.HTML()
        save_b = w.Button(description="Save (photo reviewed)", button_style="success",
                          layout=w.Layout(width="auto"))
        save_b.on_click(lambda _: self._save())
        save_next_b = w.Button(description="Save + next ▶", button_style="success")
        save_next_b.on_click(lambda _: (self._save(), self._go(self.i + 1)))

        left = w.VBox([self.image_label, self.image, self.strip],
                      layout=w.Layout(width=image_width))
        right = w.VBox([self.table, self.record, self.verdict_box,
                        w.HBox([save_b, save_next_b]), self.msg])
        self.box = w.VBox([w.HBox([self.field, self.order, self.min_dis, self.hide]),
                           w.HBox([prev_b, next_b, self.jump, go_b]),
                           self.header,
                           w.HBox([left, right], layout=w.Layout(align_items="flex-start"))])
        self._apply_filter()

    # ---- datos ----
    def _load_individuals(self) -> pd.DataFrame:
        from .annotate import load_events
        ind = load_events(self.cfg)["individual"]
        return ind if len(ind) else pd.DataFrame(columns=["photo_id", "individual"])

    def _apply_filter(self):
        f = self.field.value
        cols = COMPARE_FIELDS if f == "any" else [f]
        rows = self.full
        if self.min_dis.value > 0:          # 0 = todas las fotos anotadas
            keep = pd.Series(False, index=rows.index)
            for c in cols:
                keep |= rows[f"n_disagree__{c}"] >= self.min_dis.value
            rows = rows[keep]
        if self.order.value == "most_disagreement":
            key = "n_disagree_max" if f == "any" else f"n_disagree__{f}"
            rows = rows.sort_values([key, "observation_id", "position"],
                                    ascending=[False, True, True])
        if self.hide.value:
            v = load_verdicts(self.cfg, self.annotator)
            if len(v):
                done = v[v["field"] == PHOTO_CHECKED]["photo_id"].astype(int)
                rows = rows[~rows["photo_id"].isin(set(done))]
        self.rows = rows.reset_index(drop=True)
        self._go(0)

    def _go(self, i):
        if self.rows.empty:
            self.header.value = "<b>No photos for this filter</b>"
            self.image.value, self.table.value, self.record.value = b"", "", ""
            self.strip.children, self.verdict_box.children = (), ()
            return
        self.i = max(0, min(i, len(self.rows) - 1))
        self.msg.value = ""
        self._render()

    def _jump(self):
        try:
            oid = int(self.jump.value.strip())
        except ValueError:
            return
        idx = self.rows.index[self.rows["observation_id"] == oid]
        if not len(idx):
            self.header.value = f"<b style='color:#b00'>ID {oid} not in this filter</b>"
            return
        self._go(int(idx[0]))

    def _read(self, file_name, dirs) -> bytes:
        name = Path(file_name).stem + ".jpg"
        key = (name, str(dirs[0]))
        if key not in self._cache:
            data = b""
            for d in dirs:
                for p in (d / name, d / file_name):
                    if p.exists():
                        data = p.read_bytes()
                        break
                if data:
                    break
            self._cache[key] = data
        return self._cache[key]

    # ---- dibujo ----
    def _render(self):
        r = self.rows.loc[self.i]
        oid = int(r["observation_id"])
        url = f"https://www.inaturalist.org/observations/{oid}"
        self.header.value = (f"<h3 style='margin:4px 0'>{self.i + 1}/{len(self.rows)} · "
                             f"<a href='{url}' target='_blank'>{oid}</a> · photo "
                             f"{int(r['position'])} (id {int(r['photo_id'])})</h3>")
        self._show_big(r)
        self._render_strip(r)
        self._render_table(r)
        self._render_record(r)
        self._render_verdicts(r)

    def _show_big(self, photo_row, reviewed_row=None):
        self.image.value = self._read(photo_row["file_name"], self.big_dirs)
        txt = f"Photo {int(photo_row['position'])}"
        if reviewed_row is not None and int(photo_row["photo_id"]) != int(reviewed_row["photo_id"]):
            txt += (f" <b style='color:#b00'>(not the photo under review: "
                    f"photo {int(reviewed_row['position'])})</b>")
        self.image_label.value = txt

    def _render_strip(self, r):
        w = self.w
        rec = self.all_photos[self.all_photos["observation_id"] == r["observation_id"]] \
            .sort_values("position")
        if len(rec) < 2:
            self.strip.children = ()
            return
        items = []
        for _, p in rec.iterrows():
            current = int(p["photo_id"]) == int(r["photo_id"])
            img = w.Image(value=self._read(p["file_name"], self.thumb_dirs), format="jpeg",
                          layout=w.Layout(max_width=self.thumb_px, max_height=self.thumb_px,
                                          object_fit="contain",
                                          border="3px solid #1a73e8" if current
                                          else "3px solid transparent"))
            b = w.Button(description=f"Photo {int(p['position'])}",
                         button_style="info" if current else "",
                         layout=w.Layout(width=self.thumb_px))
            b.on_click(lambda _, p=p: self._show_big(p, self.rows.loc[self.i]))
            items.append(w.VBox([img, b], layout=w.Layout(margin="0 6px 6px 0")))
        self.strip.children = items

    def _render_table(self, r):
        cols = self.versions + ["human"]
        head = "".join(f"<th style='padding:3px 7px'>{html.escape(c.replace('photo_traits_', ''))}</th>"
                       for c in cols)
        body = ""
        for field in COMPARE_FIELDS + INFO_FIELDS:
            if not any(f"{field}__{c}" in self.rows for c in cols):
                continue
            vals = [_norm(r.get(f"{field}__{c}")) for c in cols]
            human = vals[-1]
            cells = ""
            for c, v in zip(cols, vals):
                st = "padding:3px 7px;vertical-align:top"
                if field in INFO_FIELDS:
                    st += ";font-size:85%;max-width:180px"
                elif c != "human" and human and v not in NOT_COMPARED and v != human:
                    st += ";background:#f8d7da;color:#000"
                elif c == "human":
                    st += ";font-weight:bold"
                cells += f"<td style='{st}'>{html.escape(v)}</td>"
            n = ""
            if field in COMPARE_FIELDS:
                n = f" <small>({int(r[f'n_disagree__{field}'])}/{int(r[f'n_models__{field}'])})</small>"
            body += f"<tr><th style='text-align:right;padding:3px 7px'>{field}{n}</th>{cells}</tr>"
        legend = ("<small>red = differs from me · (x/y) = models that differ / models that "
                  "decided · unclear and not_applicable are not counted · my side/erosion/quality "
                  "only for photos with 1 individual</small>")
        self.table.value = (f"<table style='border-collapse:collapse'><tr><th></th>{head}</tr>"
                            f"{body}</table>{legend}")

    def _render_record(self, r):
        """Todas las fotos del registro: individuos anotados y lado de cada version."""
        rec = self.all_photos[self.all_photos["observation_id"] == r["observation_id"]] \
            .sort_values("position")
        if len(rec) < 2:
            self.record.value = ""
            return
        ind = self.individuals
        vs = [v for v in self.versions if f"side__{v}" in rec]
        head = "<th>photo</th><th>me (letter: side, erosion, quality)</th>" + "".join(
            f"<th>{html.escape(v.replace('photo_traits_', ''))}</th>" for v in vs)
        body = ""
        for _, p in rec.iterrows():
            mine = ind[ind["photo_id"] == int(p["photo_id"])].sort_values("individual")
            me = "; ".join(f"{i['individual']}: {_norm(i.get('side'))}, {_norm(i.get('erosion'))}, "
                           f"{_norm(i.get('photo_quality'))}" for _, i in mine.iterrows()) or "—"
            bold = "font-weight:bold;" if int(p["photo_id"]) == int(r["photo_id"]) else ""
            body += (f"<tr style='{bold}'><td>{int(p['position'])}</td><td>{html.escape(me)}</td>"
                     + "".join(f"<td>{html.escape(_norm(p.get(f'side__{v}')))}</td>" for v in vs)
                     + "</tr>")
        self.record.value = ("<p style='margin:10px 0 2px'><b>Whole record (side)</b></p>"
                             "<table style='border-collapse:collapse;font-size:90%' "
                             f"cellpadding='3'><tr>{head}</tr>{body}</table>")

    def _render_verdicts(self, r):
        w = self.w
        prev = load_verdicts(self.cfg, self.annotator)
        prev = prev[prev["photo_id"] == int(r["photo_id"])] if len(prev) else prev
        self._verdict_widgets = {}
        done = len(prev) and (prev["field"] == PHOTO_CHECKED).any()
        status = ("<span style='color:#080'>✓ reviewed</span>" if done
                  else "<span style='color:#888'>not reviewed</span>")
        rows = [w.HTML(f"<p style='margin:10px 0 2px'><b>Verdict</b> · {status} · "
                       "<small>leave — if the field is fine</small></p>")]
        for f in COMPARE_FIELDS:
            old = prev[prev["field"] == f] if len(prev) else prev
            tb = w.ToggleButtons(options=VERDICTS,
                                 value=old["verdict"].iloc[0] if len(old) else "none",
                                 style={"button_width": "auto"})
            note = w.Text(value=str(old["note"].iloc[0] or "") if len(old) else "",
                          placeholder="note (what is right, why)",
                          layout=w.Layout(width="330px"))
            self._verdict_widgets[f] = (tb, note)
            color = "#b00" if int(r[f"n_disagree__{f}"]) else "#000"
            rows.append(w.HBox([w.HTML(f"<b style='display:inline-block;width:160px;"
                                       f"color:{color}'>{f}</b>"),
                                tb, note]))
        self.verdict_box.children = rows

    def _save(self):
        r = self.rows.loc[self.i]
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        n = 0
        prev = load_verdicts(self.cfg, self.annotator)
        prev = prev[prev["photo_id"] == int(r["photo_id"])] if len(prev) else prev
        for f, (tb, note) in self._verdict_widgets.items():
            had = len(prev) and (prev["field"] == f).any()
            if tb.value == "none" and not had:
                continue                       # nada que guardar ni que borrar
            append_jsonl({
                "observation_id": int(r["observation_id"]), "photo_id": int(r["photo_id"]),
                "field": f, "verdict": tb.value, "note": note.value.strip() or None,
                "human_value": _norm(r.get(f"{f}__human")),
                "model_values": {v: _norm(r.get(f"{f}__{v}")) for v in self.versions},
                "reviewer": self.annotator, "saved_at": now,
            }, verdict_file(self.cfg, self.annotator))
            n += 1
        # la foto queda marcada como revisada aunque no haya ningun error
        append_jsonl({"observation_id": int(r["observation_id"]), "photo_id": int(r["photo_id"]),
                      "field": PHOTO_CHECKED, "verdict": "checked", "note": None,
                      "versions": self.versions, "reviewer": self.annotator, "saved_at": now},
                     verdict_file(self.cfg, self.annotator))
        self.msg.value = (f"<span style='color:#080'>Photo marked as reviewed · "
                          f"{n} verdict(s) saved</span>")

    def show(self):
        from IPython.display import display
        display(self.box)
