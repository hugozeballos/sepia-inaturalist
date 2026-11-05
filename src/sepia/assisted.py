"""
Anotacion ASISTIDA: la persona revisa foto por foto los valores de los prompts y deja su
anotacion, mas rapido que en AnnotationApp. NO es anotacion a ciegas.

Reglas por atributo (conteo, lado, erosion, calidad), en orden de prioridad:
  1. manual: la persona escribe el valor            -> source = "manual"
  2. clic:   la persona elige el valor de un prompt  -> source = "click:<version>"
  3. todos los prompts coinciden (tambien 'unclear') -> source = "all_agree"
  Si los prompts no coinciden y no hay clic ni valor manual, la foto no se puede guardar.
Nada se guarda solo: se guarda al apretar Save (la persona miro la foto).

Otros campos (estado, picoteo, epibiontes, certeza, notas, picorocos) se llenan a mano, con
valores por defecto: estado cuttlebone, epibiontes no, certeza 1, picoteo segun el lado
(dorsal -> no, ventral -> yes). Si no se tocan, su source es "default".

Solo fotos con 1 hueso. Si algun prompt ve 1 hueso pero el conteo final no es 1, se guarda
solo el conteo y la foto queda para anotarla despues en AnnotationApp.

Archivo aparte: annotations_assisted/<anotador>.jsonl (una linea por foto, vale la ultima).
No esta en annotations/, asi que load_events y las evaluaciones no lo leen.

Uso (Colab):
    from sepia.assisted import AssistedApp, load_assisted
    AssistedApp(cfg, VERSIONS, annotator="HZ").show()
"""
from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import pandas as pd

from .config import Config
from .io import paths, read_jsonl, append_jsonl
from .review import review_table, _norm

# Atributos que dan los prompts y valores validos para escribir a mano
MODEL_FIELDS = {
    "n_cuttlebones_visible": None,                      # entero >= 0
    "side": ["dorsal", "ventral", "unclear"],
    "erosion": ["1", "2", "3", "unclear"],
    "photo_quality": ["high", "medium", "low"],
}
# Evidencia de cada prompt (solo se muestra)
EVIDENCE = ["side_basis", "side_evidence", "cover", "focus", "quality_evidence", "notes"]
# Campos manuales: (nombre, opciones o "text", valor por defecto)
MANUAL_FIELDS = [
    ("state", ["cuttlebone", "live animal", "egg case"], "cuttlebone"),
    ("bitemarks", ["", "yes", "no", "(yes)"], None),          # por defecto segun el lado
    ("epibionts", ["yes", "no"], "no"),
    ("certainty", [1, 0], 1),
    ("taxonomic_groups", "text", ""),
    ("barnacles_n", ["", "0", "1-10", "10-100", ">100"], ""),
    ("barnacle_size", ["", "small", "medium", "large"], ""),
    ("barnacle_taxon", "text", ""),
    ("notes", "text", ""),
]
BARNACLE = {"taxonomic_groups", "barnacles_n", "barnacle_size", "barnacle_taxon"}
BITEMARKS_BY_SIDE = {"dorsal": "no", "ventral": "yes"}
NO_VALUE = {"", "not_applicable"}


def assisted_file(cfg: Config, annotator: str) -> Path:
    d = paths(cfg)["base"] / "annotations_assisted"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{annotator}.jsonl"


def load_assisted(cfg: Config, annotator: str) -> pd.DataFrame:
    """Ultima linea guardada por foto."""
    a = read_jsonl(assisted_file(cfg, annotator))
    if a.empty:
        return a
    return (a.sort_values("saved_at", kind="stable")
            .drop_duplicates("photo_id", keep="last").reset_index(drop=True))


def resolve(model_values: dict, click: str | None = None, manual: str = "") -> tuple[str, str]:
    """
    Valor final y su origen para un atributo.
    model_values: {version: valor} (vacios y not_applicable no cuentan).
    Prioridad: manual > clic > todos coinciden. Si no hay nada: ("", "").
    """
    if manual.strip():
        return manual.strip(), "manual"
    if click and click in model_values:
        return model_values[click], f"click:{click}"
    vals = [v for v in model_values.values() if v not in NO_VALUE]
    if vals and len(vals) == len([v for v in model_values.values()]) and len(set(vals)) == 1:
        return vals[0], "all_agree"
    return "", ""


def validate(field: str, value: str) -> str | None:
    """Mensaje de error si un valor escrito a mano no es valido; None si esta bien."""
    if field == "n_cuttlebones_visible":
        return None if value.isdigit() else "count must be a whole number"
    ok = MODEL_FIELDS[field]
    return None if value in ok else f"{field} must be one of {ok}"


def default_bitemarks(side: str) -> str:
    return BITEMARKS_BY_SIDE.get(side, "")


class AssistedApp:
    """Foto + tabla clicable de valores por prompt + columna 'human' + campos manuales."""

    def __init__(self, cfg: Config, versions: list[str], annotator: str = "HZ",
                 image_width: str = "520px", thumb_px: str = "120px"):
        import ipywidgets as w
        self.w, self.cfg, self.versions, self.annotator = w, cfg, versions, annotator
        base = paths(cfg)["base"] / "photos"
        self.big_dirs = [base / "1024px", base / "original"]
        self.thumb_dirs = [base / "512px"] + self.big_dirs
        self.thumb_px = thumb_px
        self._cache = {}
        self.all_photos = review_table(cfg, versions)
        n_cols = [f"n_cuttlebones_visible__{v}" for v in versions
                  if f"n_cuttlebones_visible__{v}" in self.all_photos]
        one = pd.Series(False, index=self.all_photos.index)
        for c in n_cols:                                   # algun prompt ve 1 hueso
            one |= self.all_photos[c].map(_norm) == "1"
        self.rows = self.all_photos[one].reset_index(drop=True)

        prev_b, next_b = w.Button(description="◀ Prev"), w.Button(description="Next ▶")
        prev_b.on_click(lambda _: self._go(self.i - 1))
        next_b.on_click(lambda _: self._go(self.i + 1))
        self.jump = w.Text(placeholder="iNaturalist ID", layout=w.Layout(width="150px"))
        go_b = w.Button(description="Go to ID")
        go_b.on_click(lambda _: self._jump())
        save_b = w.Button(description="Save", button_style="success")
        save_b.on_click(lambda _: self._save())
        save_next_b = w.Button(description="Save + next ▶", button_style="success")
        save_next_b.on_click(lambda _: self._save(go_next=True))

        self.header = w.HTML()
        self.image = w.Image(format="jpeg", layout=w.Layout(max_width=image_width,
                                                            max_height="520px",
                                                            object_fit="contain"))
        self.image_label = w.HTML()
        self.strip = w.HBox(layout=w.Layout(flex_flow="row wrap"))
        self.grid = w.VBox()
        self.evidence = w.HTML()
        self.msg = w.HTML()
        # campos manuales
        self.manual = {}
        for name, opts, _ in MANUAL_FIELDS:
            if opts == "text":
                wid = w.Text(description=name, layout=w.Layout(width="360px"),
                             style={"description_width": "120px"})
            else:
                wid = w.Dropdown(options=opts, description=name, layout=w.Layout(width="300px"),
                                 style={"description_width": "120px"})
            wid.observe(lambda ch, n=name: self._touched(n), names="value")
            self.manual[name] = wid
        self.manual["epibionts"].observe(lambda _: self._barnacle_visibility(), names="value")
        left = w.VBox([self.image_label, self.image, self.strip],
                      layout=w.Layout(width=image_width))
        right = w.VBox([self.grid, self.evidence,
                        w.HTML("<b>Manual fields</b>"), *self.manual.values(),
                        w.HBox([save_b, save_next_b]), self.msg],
                       layout=w.Layout(margin="0 0 0 12px"))
        self.box = w.VBox([w.HBox([prev_b, next_b, self.jump, go_b]), self.header,
                           w.HBox([left, right], layout=w.Layout(align_items="flex-start"))])
        self._go(0)

    # ---- navegacion ----
    def _go(self, i):
        if self.rows.empty:
            self.header.value = "<b>No photos with 1 cuttlebone for any prompt</b>"
            return
        self.i = max(0, min(i, len(self.rows) - 1))
        self._render()

    def _jump(self):
        try:
            oid = int(self.jump.value.strip())
        except ValueError:
            return
        idx = self.rows.index[self.rows["observation_id"] == oid]
        if len(idx):
            self._go(int(idx[0]))
        else:
            self.msg.value = f"<b style='color:#b00'>ID {oid} not in the list</b>"

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

    # ---- valores ----
    def _model_values(self, field) -> dict:
        r = self.rows.loc[self.i]
        return {v: _norm(r.get(f"{field}__{v}")) for v in self.versions}

    def _final(self, field) -> tuple[str, str]:
        return resolve(self._model_values(field), self.clicks.get(field),
                       self.manual_inputs[field].value)

    # ---- dibujo ----
    def _render(self):
        w = self.w
        r = self.rows.loc[self.i]
        oid, pid = int(r["observation_id"]), int(r["photo_id"])
        saved = load_assisted(self.cfg, self.annotator)
        saved = saved[saved["photo_id"] == pid] if len(saved) else saved
        self.saved = saved.iloc[0].to_dict() if len(saved) else None
        url = f"https://www.inaturalist.org/observations/{oid}"
        status = ("<span style='color:#080'>✓ saved</span>" if self.saved
                  else "<span style='color:#888'>not saved</span>")
        self.header.value = (f"<h3 style='margin:4px 0'>{self.i + 1}/{len(self.rows)} · "
                             f"<a href='{url}' target='_blank'>{oid}</a> · photo "
                             f"{int(r['position'])} (id {pid}) · {status}</h3>")
        self.msg.value = ""
        self.image.value = self._read(r["file_name"], self.big_dirs)
        self.image_label.value = f"Photo {int(r['position'])}"
        self._render_strip(r)

        # estado de esta foto: clics y valores manuales (restaurados si ya se guardo)
        src = (self.saved or {}).get("sources") or {}
        vals = (self.saved or {})
        self.clicks = {f: s.split(":", 1)[1] for f, s in src.items()
                       if f in MODEL_FIELDS and isinstance(s, str) and s.startswith("click:")}
        self.manual_inputs, self.cells, self.final_html = {}, {}, {}
        head = [w.HTML("<b style='display:inline-block;width:170px'></b>")] + [
            w.HTML(f"<b style='display:inline-block;width:100px;text-align:center'>"
                   f"{html.escape(v.replace('photo_traits_', ''))}</b>") for v in self.versions] + [
            w.HTML("<b style='display:inline-block;width:110px'>write</b>"),
            w.HTML("<b>human</b>")]
        rows = [w.HBox(head)]
        for f in MODEL_FIELDS:
            mv = self._model_values(f)
            btns = []
            for v in self.versions:
                b = w.Button(description=mv[v] or "—", layout=w.Layout(width="100px"),
                             disabled=mv[v] in NO_VALUE, tooltip=f"use {v}")
                b.on_click(lambda _, f=f, v=v: self._click(f, v))
                btns.append(b)
            self.cells[f] = btns
            txt = w.Text(value=str(vals.get(f, "")) if src.get(f) == "manual" else "",
                         placeholder="type", layout=w.Layout(width="110px"))
            txt.observe(lambda _, f=f: self._refresh(f), names="value")
            self.manual_inputs[f] = txt
            self.final_html[f] = w.HTML()
            rows.append(w.HBox([w.HTML(f"<b style='display:inline-block;width:170px'>{f}</b>"),
                                *btns, txt, self.final_html[f]]))
        legend = ("<small>click = use that prompt's value · type = your own value (wins over "
                  "click) · if all prompts agree their value is proposed · nothing is saved "
                  "until you press Save</small>")
        self.grid.children = rows + [w.HTML(legend)]
        for f in MODEL_FIELDS:
            self._refresh(f, update_defaults=False)

        # evidencia
        ev = "".join(
            f"<tr><th style='text-align:right;padding:2px 6px'>{e}</th>" + "".join(
                f"<td style='padding:2px 6px;font-size:85%;max-width:160px;vertical-align:top'>"
                f"{html.escape(_norm(r.get(f'{e}__{v}')))}</td>" for v in self.versions) + "</tr>"
            for e in EVIDENCE if any(f"{e}__{v}" in self.rows for v in self.versions))
        hv = "".join(f"<th style='padding:2px 6px'>{html.escape(v.replace('photo_traits_', ''))}</th>"
                     for v in self.versions)
        self.evidence.value = (f"<details open><summary><b>Evidence</b></summary><table "
                               f"style='border-collapse:collapse'><tr><th></th>{hv}</tr>{ev}"
                               f"</table></details>")

        # campos manuales: lo guardado o los valores por defecto
        self._loading = True
        self.touched = set()
        side = self._final("side")[0]
        for name, opts, default in MANUAL_FIELDS:
            if self.saved is not None and name in self.saved and self.saved[name] is not None \
                    and not (isinstance(self.saved[name], float) and self.saved[name] != self.saved[name]):
                value = self.saved[name]
                if (src.get(name) or "default") != "default":
                    self.touched.add(name)
            else:
                value = default_bitemarks(side) if name == "bitemarks" else default
            wid = self.manual[name]
            if opts != "text" and value not in wid.options:
                value = default if default in wid.options else wid.options[0]
            wid.value = value
        self._loading = False
        self._barnacle_visibility()

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
            b.on_click(lambda _, p=p: self._peek(p))
            items.append(w.VBox([img, b], layout=w.Layout(margin="0 6px 6px 0")))
        self.strip.children = items

    def _peek(self, p):
        """Ver otra foto del registro en grande (la tabla sigue siendo de la foto en revision)."""
        self.image.value = self._read(p["file_name"], self.big_dirs)
        cur = self.rows.loc[self.i]
        label = f"Photo {int(p['position'])}"
        if int(p["photo_id"]) != int(cur["photo_id"]):
            label += (f" <b style='color:#b00'>(not the photo under review: "
                      f"photo {int(cur['position'])})</b>")
        self.image_label.value = label

    # ---- eventos ----
    def _click(self, field, version):
        if self.clicks.get(field) == version:
            self.clicks.pop(field)                       # segundo clic: deshace
        else:
            self.clicks[field] = version
        self._refresh(field)

    def _refresh(self, field, update_defaults=True):
        value, source = self._final(field)
        chosen = source.split(":", 1)[1] if source.startswith("click:") else None
        for v, b in zip(self.versions, self.cells[field]):
            b.button_style = "info" if v == chosen else ""
        mv = self._model_values(field)
        decided = [x for x in mv.values() if x not in NO_VALUE]
        differ = len(set(decided)) > 1 or len(decided) < len(mv)
        if not value:
            txt = "<b style='color:#b00'>choose or type</b>" if differ else ""
        else:
            err = validate(field, value) if source == "manual" else None
            color = "#b00" if err else "#000"
            txt = (f"<b style='color:{color}'>{html.escape(value)}</b> "
                   f"<small style='color:#666'>({source})</small>"
                   + (f"<br><small style='color:#b00'>{html.escape(err)}</small>" if err else ""))
        self.final_html[field].value = txt
        if field == "side" and update_defaults and "bitemarks" not in self.touched:
            self._loading = True
            self.manual["bitemarks"].value = default_bitemarks(value)
            self._loading = False

    def _touched(self, name):
        if not getattr(self, "_loading", False):
            self.touched.add(name)

    def _barnacle_visibility(self):
        show = self.manual["epibionts"].value == "yes"
        for n in BARNACLE:
            self.manual[n].layout.display = "" if show else "none"

    # ---- guardar ----
    def record(self) -> tuple[dict | None, list[str]]:
        """Linea a guardar para la foto actual, o (None, errores)."""
        r = self.rows.loc[self.i]
        final = {f: self._final(f) for f in MODEL_FIELDS}
        errors = []
        count, csrc = final["n_cuttlebones_visible"]
        if not count:
            errors.append("n_cuttlebones_visible: choose or type")
        elif csrc == "manual" and validate("n_cuttlebones_visible", count):
            errors.append(validate("n_cuttlebones_visible", count))
        one = count == "1"
        fields = list(MODEL_FIELDS) if one else ["n_cuttlebones_visible"]
        for f in fields[1:]:
            v, s = final[f]
            if not v:
                errors.append(f"{f}: choose or type")
            elif s == "manual" and validate(f, v):
                errors.append(validate(f, v))
        if errors:
            return None, errors
        rec = {"observation_id": int(r["observation_id"]), "photo_id": int(r["photo_id"]),
               "position": int(r["position"]), "individual": "a" if one else None,
               "n_cuttlebones_visible": int(count),
               "sources": {f: final[f][1] for f in fields},
               "model_values": {f: self._model_values(f) for f in MODEL_FIELDS},
               "versions": self.versions, "annotator": self.annotator,
               "label_source": "assisted",
               "needs_manual_annotation": not one,
               "saved_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        if one:
            for f in fields[1:]:
                v = final[f][0]
                rec[f] = int(v) if f == "erosion" and v.isdigit() else v
            for name, opts, _ in MANUAL_FIELDS:
                if name in BARNACLE and self.manual["epibionts"].value != "yes":
                    continue
                rec[name] = self.manual[name].value
                rec["sources"][name] = "manual" if name in self.touched else "default"
        return rec, []

    def _save(self, go_next=False):
        rec, errors = self.record()
        if errors:
            self.msg.value = ("<b style='color:#b00'>Not saved: "
                              + html.escape("; ".join(errors)) + "</b>")
            return
        append_jsonl(rec, assisted_file(self.cfg, self.annotator))
        note = (" · count ≠ 1: only the count was saved, annotate this photo later in "
                "AnnotationApp" if rec["needs_manual_annotation"] else "")
        if go_next and self.i < len(self.rows) - 1:
            self._go(self.i + 1)
            self.msg.value = f"<span style='color:#080'>Previous photo saved{note}</span>"
        else:
            self._render()
            self.msg.value = f"<span style='color:#080'>Saved{note}</span>"

    def show(self):
        from IPython.display import display
        display(self.box)
