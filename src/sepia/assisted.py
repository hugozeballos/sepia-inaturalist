"""
Anotacion ASISTIDA: la persona revisa foto por foto los valores de los prompts y deja su
anotacion, mas rapido que en AnnotationApp. NO es anotacion a ciegas.

Reglas por atributo (conteo, lado, erosion, calidad), en orden de prioridad:
  1. manual: la persona escribe el valor            -> source = "manual"
  2. clic:   la persona elige el valor de un prompt  -> source = "click:<version>"
  3. todos los prompts coinciden (tambien 'unclear') -> source = "all_agree"
  4. moda: el valor que da la mayoria de los prompts (mas votos, sin empate)
                                                     -> source = "mode:<votos>/<prompts con valor>"
  Se muestra como sugerencia explicita (celdas en verde). Si hay empate, no hay sugerencia y hay
  que elegir; sin clic ni valor manual la foto no se puede guardar.
Nada se guarda solo: se guarda al apretar Save (la persona miro la foto).

Otros campos (estado, picoteo, epibiontes, certeza, notas, picorocos) se llenan a mano, con
valores por defecto: estado cuttlebone, epibiontes no, certeza 1, picoteo segun el lado
(dorsal -> no, ventral -> yes). Si no se tocan, su source es "default".

Triaje (cfg.triage): se muestra arriba, fuera de la tabla (estado del registro, huesos y lo que
ve en esta foto); no cuenta para "todos coinciden". Su contenido por foto es el valor por
defecto del estado. La columna "manual" muestra la anotacion de AnnotationApp, si existe.

Solo fotos con 1 hueso. Si algun prompt (o el triaje) ve 1 hueso pero el conteo final no es 1, se guarda
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
# Opciones de la lista "write" (valor propio de la persona). "" = no elegir nada.
WRITE_OPTIONS = {f: [""] + (opts if opts else [str(i) for i in range(0, 16)])
                 for f, opts in MODEL_FIELDS.items()}
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
TRIAGE = "triage"
# contenido del triaje -> estado de la anotacion
TRIAGE_STATE = {"cuttlebone": "cuttlebone", "live_animal": "live animal", "egg_case": "egg case"}


def triage_by_photo(cfg: Config) -> tuple[dict, dict]:
    """
    Resultados del triaje (cfg.triage): ({(observation_id, position): {n, content}},
    {observation_id: datos del registro}). Vacios si no hay triaje.
    """
    if not cfg.triage:
        return {}, {}
    from .triage import load_triage
    t = cfg.triage
    tri = load_triage(cfg, t["provider"], t["model"], t.get("prompt_version"))
    photos, records = {}, {}
    for _, r in tri.iterrows():
        oid = int(r["observation_id"])
        records[oid] = {k: r.get(k) for k in ("record_state", "n_cuttlebones", "confidence",
                                               "count_possible", "fragments_present", "notes")}
        for ph in (r.get("photos") or []):
            photos[(oid, int(ph["photo_index"]))] = {"n": ph.get("n_cuttlebones_visible"),
                                                     "content": ph.get("content")}
    return photos, records
NO_VALUE = {"", "not_applicable"}


def assisted_file(cfg: Config, annotator: str) -> Path:
    """Archivo de la anotacion asistida: annotations_assisted/<anotador>.jsonl."""
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
    if vals:                                   # moda: mas votos, sin empate
        from collections import Counter
        top = Counter(vals).most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            return top[0][0], f"mode:{top[0][1]}/{len(vals)}"
    return "", ""


def validate(field: str, value: str) -> str | None:
    """Mensaje de error si un valor escrito a mano no es valido; None si esta bien."""
    if field == "n_cuttlebones_visible":
        return None if value.isdigit() else "count must be a whole number"
    ok = MODEL_FIELDS[field]
    return None if value in ok else f"{field} must be one of {ok}"


def default_bitemarks(side: str) -> str:
    """Picoteo por defecto segun el lado final: dorsal -> no, ventral -> yes, otro -> vacio."""
    return BITEMARKS_BY_SIDE.get(side, "")


class AssistedApp:
    """Foto + tabla clicable de valores por prompt + columna 'human' + campos manuales."""

    def __init__(self, cfg: Config, versions: list[str], annotator: str = "HZ",
                 image_width: str = "520px", thumb_px: str = "120px",
                 model_boxes: str | None = None):
        """
        Arma la lista de fotos (alguna version o el triaje ve 1 hueso) y los widgets.
        model_boxes (p. ej. "bonebox_v15"): muestra el recuadro del modelo sobre la foto. Con
        jupyter-bbox-widget instalado se puede rehacer ("Redraw box") o quitar (✕); el recuadro
        final se guarda con la foto (crop_box, model_box, box_source) y AnnotationApp
        (prefill="assisted") parte con el. Sin el widget: solo se dibuja en rojo, para verlo.
        """
        import ipywidgets as w
        self.w, self.cfg, self.versions, self.annotator = w, cfg, versions, annotator
        self.model_boxes = {}
        if model_boxes:
            from .bonebox import model_boxes as _load_boxes
            self.model_boxes = _load_boxes(cfg, model_boxes)
        base = paths(cfg)["base"] / "photos"
        self.big_dirs = [base / "1024px", base / "original"]
        self.thumb_dirs = [base / "512px"] + self.big_dirs
        self.thumb_px = thumb_px
        self._cache = {}
        self.all_photos = review_table(cfg, versions)
        self.tri_photo, self.tri_record = triage_by_photo(cfg)
        self.cols = versions + ([TRIAGE] if self.tri_photo else [])
        self.all_photos[f"n_cuttlebones_visible__{TRIAGE}"] = [
            (self.tri_photo.get((int(o), int(ps))) or {}).get("n")
            for o, ps in zip(self.all_photos["observation_id"], self.all_photos["position"])]
        n_cols = [f"n_cuttlebones_visible__{v}" for v in self.cols
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
        # Recuadro editable (opcional): BBoxWidget en vez de la imagen fija
        self.bbox = None
        self.box_value, self.box_model = None, None       # recuadro final y el del modelo (0-1)
        if self.model_boxes:
            try:
                from jupyter_bbox_widget import BBoxWidget
                self.bbox = BBoxWidget(hide_buttons=True, view_only=True, classes=["bone"],
                                       label="bone", layout=w.Layout(width=image_width))
                self.bbox.observe(lambda ch: self._on_box_drawn(ch["new"]), names="bboxes")
            except ImportError:
                self.bbox = None
        self._box_updating = False
        self.redraw_btn = w.ToggleButton(description="▭ Redraw box",
                                         tooltip="drag a new rectangle around the bone",
                                         layout=w.Layout(width="130px"))
        self.redraw_btn.observe(lambda ch: self._toggle_redraw(ch["new"]), names="value")
        clear_box_b = w.Button(description="✕ no box", layout=w.Layout(width="90px"),
                               tooltip="remove the box for this photo")
        clear_box_b.on_click(lambda _: self._set_box(None))
        self.box_info = w.HTML()
        self.box_controls = w.HBox([self.redraw_btn, clear_box_b, self.box_info])
        self.strip = w.HBox(layout=w.Layout(flex_flow="row wrap"))
        self.grid = w.VBox()
        self.evidence = w.HTML()
        self.msg = w.HTML()
        # campos manuales
        self.manual = {}
        for name, opts, _ in MANUAL_FIELDS:
            if opts == "text":
                wid = w.Text(description=name, layout=w.Layout(width="315px"),
                             style={"description_width": "110px"})
            else:
                wid = w.Dropdown(options=opts, description=name, layout=w.Layout(width="315px"),
                                 style={"description_width": "110px"})
            wid.observe(lambda ch, n=name: self._touched(n), names="value")
            self.manual[name] = wid
        self.manual["epibionts"].observe(lambda _: self._barnacle_visibility(), names="value")
        viewer = [self.bbox, self.box_controls] if self.bbox is not None else [self.image]
        left = w.VBox([self.image_label, *viewer, self.strip],
                      layout=w.Layout(width=image_width))
        self.manual_box = w.GridBox(list(self.manual.values()), layout=w.Layout(
            grid_template_columns="repeat(2, 320px)", grid_gap="2px 8px"))
        right = w.VBox([self.grid,
                        w.HTML("<b>Manual fields</b>"), self.manual_box,
                        w.HBox([save_b, save_next_b]), self.msg,
                        self.evidence],                 # solo lectura: al final
                       layout=w.Layout(margin="0 0 0 12px"))
        pend_b = w.Button(description="Pending ⏭",
                          tooltip="next photo not saved here and not annotated manually")
        pend_b.on_click(lambda _: self._go_pending())
        self.box = w.VBox([w.HBox([prev_b, next_b, pend_b, self.jump, go_b]), self.header,
                           w.HBox([left, right], layout=w.Layout(align_items="flex-start"))])
        self._go(self._next_pending(0))              # retoma donde quedo

    # ---- navegacion ----
    def _go_pending(self):
        """Siguiente foto pendiente despues de la actual; si no hay, vuelve a buscar desde el inicio."""
        n = len(self.rows)
        j = self._next_pending(self.i + 1, wrap=True)
        if j is None:
            self.msg.value = "<b>All photos done 🎉</b>"
            return
        self._go(j)

    def _next_pending(self, start: int = 0, wrap: bool = False):
        """
        Primera foto desde 'start' que no esta guardada en la asistida ni anotada a mano
        (AnnotationApp). Si no queda ninguna, la ultima foto.
        """
        if self.rows.empty:
            return 0
        done = set()
        a = load_assisted(self.cfg, self.annotator)
        if len(a):
            done |= set(a["photo_id"].astype("int64"))
        from .annotate import load_events
        ev = load_events(self.cfg)
        for k in ("individual", "photo_info", "empty_photo", "duplicate_photo"):
            if len(ev[k]):
                done |= set(ev[k]["photo_id"].dropna().astype("int64"))
        pids = self.rows["photo_id"].astype("int64").tolist()
        order = list(range(max(0, start), len(pids)))
        if wrap:
            order += list(range(0, min(start, len(pids))))
        for i in order:
            if pids[i] not in done:
                return i
        return None if wrap else len(pids) - 1
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

    # ---- recuadro editable ----
    def _photo_path(self, file_name):
        """Foto para el recuadro: la version de 512 px (cabe entera en pantalla y el recuadro
        se ve completo); si no existe, la de 1024 px u original. El recuadro es relativo (0-1),
        asi que vale para cualquier tamano."""
        name = Path(file_name).stem + ".jpg"
        for d in [self.thumb_dirs[0]] + self.big_dirs:
            for f in (d / name, d / file_name):
                if f.exists():
                    return f
        return None

    def _show_in_bbox(self, file_name, box, editable):
        """Foto en el BBoxWidget con el recuadro dado (o sin recuadro)."""
        from PIL import Image, ImageOps
        from .annotate import from_crop_box
        path = self._photo_path(file_name)
        self.redraw_btn.value = False
        self.box_controls.layout.display = None if editable else "none"
        if path is None:
            return
        with Image.open(path) as im:
            self._box_size = ImageOps.exif_transpose(im).size
        self._box_updating = True
        self.bbox.image = str(path)
        self.bbox.bboxes = ([{**from_crop_box(box, *self._box_size), "label": "bone"}]
                            if isinstance(box, dict) else [])
        self._box_updating = False
        self._box_status()

    def _box_status(self):
        from .annotate import box_source
        src = box_source(self.box_value, self.box_model)
        txt = {"model": "box from the model", "model_edited": "box redrawn by you",
               "manual": "box drawn by you", None: "no box"}[src]
        self.box_info.value = f"<small style='color:#555'>{txt}</small>"

    def _toggle_redraw(self, on):
        if self.bbox is None:
            return
        self.bbox.view_only = not on
        self.redraw_btn.button_style = "warning" if on else ""
        if on:
            self.box_info.value = "<small><b>drag a rectangle around the bone</b></small>"

    def _on_box_drawn(self, boxes):
        """Al soltar: queda solo el recuadro nuevo; se guarda al apretar Save."""
        if self._box_updating or not self.redraw_btn.value or not boxes:
            return
        from .annotate import to_crop_box
        box = to_crop_box(boxes[-1], *self._box_size)
        if box["x1"] - box["x0"] < 0.01 or box["y1"] - box["y0"] < 0.01:
            return
        self._set_box(box)

    def _set_box(self, box):
        """Recuadro final de esta foto (None = sin recuadro); se muestra y se guarda con Save."""
        from .annotate import from_crop_box
        self.box_value = box
        if self.bbox is not None and getattr(self, "_box_size", None):
            self._box_updating = True
            self.bbox.bboxes = ([{**from_crop_box(box, *self._box_size), "label": "bone"}]
                                if box else [])
            self._box_updating = False
        self.redraw_btn.value = False
        self._box_status()

    def _manual_values(self, oid: int, pid: int) -> dict:
        """
        Anotacion manual de esta foto (AnnotationApp), si existe, para mostrarla al lado:
        conteo de la foto y, si hay un solo individuo, lado, erosion y calidad. Se lee en cada
        foto, asi refleja lo anotado despues de abrir la asistida. Solo se muestra; no se cambia.
        """
        from .annotate import load_events
        ev = load_events(self.cfg)
        out = {}
        info = ev["photo_info"]
        n = info.loc[info["photo_id"].astype("int64") == pid, "n_in_photo"] if len(info) else []
        ind = ev["individual"]
        here = ind[ind["photo_id"].astype("int64") == pid] if len(ind) else ind
        if len(n):
            out["n_cuttlebones_visible"] = int(n.iloc[0])
        elif len(here):
            out["n_cuttlebones_visible"] = len(here)
        if len(here) == 1:
            row = here.iloc[0]
            for f in ("side", "erosion", "photo_quality"):
                v = row.get(f)
                if v is not None and v == v:
                    out[f] = v
        elif len(here) > 1:
            for f in ("side", "erosion", "photo_quality"):
                out[f] = f"{len(here)} individuals"
        return out

    def _with_box(self, data: bytes, photo_id: int) -> bytes:
        """La foto con el recuadro del modelo dibujado en rojo (si hay); si no, tal cual."""
        box = self.model_boxes.get(photo_id)
        if not box or not data:
            return data
        import io
        from PIL import Image, ImageDraw, ImageOps
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        W, H = im.size
        ImageDraw.Draw(im).rectangle([box["x0"] * W, box["y0"] * H, box["x1"] * W, box["y1"] * H],
                                     outline=(220, 0, 0), width=max(3, W // 150))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=90)
        return out.getvalue()

    # ---- valores ----
    def _model_values(self, field) -> dict:
        r = self.rows.loc[self.i]
        # Solo los prompts: el triaje se muestra arriba (recuadro "Triage"), fuera de la tabla
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
        tr = self.tri_record.get(oid)
        if tr:
            tp = self.tri_photo.get((oid, int(r["position"]))) or {}
            self.header.value += (
                "<div style='background:#eef;color:#000;padding:3px 8px;margin:2px 0'>"
                f"<b>Triage</b> · record: {html.escape(_norm(tr.get('record_state')))}, "
                f"{html.escape(_norm(tr.get('n_cuttlebones')))} bone(s) in the record"
                f" (count possible: {html.escape(_norm(tr.get('count_possible')))}, "
                f"confidence {html.escape(_norm(tr.get('confidence')))}) · this photo: "
                f"{html.escape(_norm(tp.get('content')))}, {html.escape(_norm(tp.get('n')))} visible"
                + (f" · notes: {html.escape(_norm(tr.get('notes')))}" if tr.get("notes") else "")
                + "</div>")
        self.msg.value = ""
        self.box_model = self.model_boxes.get(pid)
        # Recuadro ya revisado aqui (aceptado, rehecho o quitado): ese. Si no (foto sin guardar o
        # guardada antes de los recuadros, con NaN en la columna): el del modelo.
        if self.saved and self.saved.get("box_reviewed") == True:   # noqa: E712
            saved_box = self.saved.get("crop_box")
            self.box_value = saved_box if isinstance(saved_box, dict) else None
        else:
            self.box_value = self.box_model
        if self.bbox is not None:
            self._show_in_bbox(r["file_name"], self.box_value, editable=True)
        else:
            self.image.value = self._with_box(self._read(r["file_name"], self.big_dirs), pid)
        self.image_label.value = f"Photo {int(r['position'])}" + (
            " · <span style='color:#d00'>red box = model</span>"
            if (self.bbox is None and pid in self.model_boxes) else "")
        self._render_strip(r)

        # estado de esta foto: clics y valores manuales (restaurados si ya se guardo)
        src = (self.saved or {}).get("sources") or {}
        vals = (self.saved or {})
        self.clicks = {f: s.split(":", 1)[1] for f, s in src.items()
                       if f in MODEL_FIELDS and isinstance(s, str) and s.startswith("click:")}
        self.manual_inputs, self.cells, self.final_html, self.labels = {}, {}, {}, {}
        hand = self._manual_values(oid, pid)
        # Tabla en una grilla: los titulos quedan siempre alineados con sus columnas
        center = "<b style='display:block;text-align:center'>{}</b>"
        cells = [w.HTML("")] + [
            w.HTML(center.format(html.escape(v.replace("photo_traits_", ""))))
            for v in self.versions] + [
            w.HTML(center.format("my value")), w.HTML(center.format("final")),
            w.HTML("<b style='display:block;text-align:center' "
                   "title='manual annotation (AnnotationApp), if any'>manual</b>")]
        for f in MODEL_FIELDS:
            mv = self._model_values(f)
            btns = []
            self.cells[f] = {}
            for v in self.versions:
                val = mv.get(v, "")
                b = w.Button(description=val or "—", layout=w.Layout(width="auto"),
                             disabled=val in NO_VALUE, tooltip=f"use {v}")
                b.on_click(lambda _, f=f, v=v: self._click(f, v))
                btns.append(b)
                if v in mv:
                    self.cells[f][v] = b
            # lista cerrada (no texto libre): asi no se puede escribir un valor invalido
            prev = str(vals.get(f, "")) if src.get(f) == "manual" else ""
            txt = w.Dropdown(options=WRITE_OPTIONS[f],
                             value=prev if prev in WRITE_OPTIONS[f] else "",
                             layout=w.Layout(width="auto"))
            txt.observe(lambda _, f=f: self._refresh(f), names="value")
            self.manual_inputs[f] = txt
            self.final_html[f] = w.HTML()
            self.labels[f] = w.HTML()
            hv = hand.get(f)
            if isinstance(hv, float) and hv == hv and hv.is_integer():
                hv = int(hv)                                    # 3.0 -> 3
            manual_cell = w.HTML(
                f"<span style='display:block;padding:4px 6px;background:#d4edda;color:#000;"
                f"text-align:center'>{html.escape(str(hv))}</span>"
                if hv not in (None, "") else
                "<span style='display:block;text-align:center;color:#888'>—</span>")
            cells += [self.labels[f], *btns, txt, self.final_html[f], manual_cell]
        legend = ("<small><span style='background:#ffe08a;color:#000;padding:0 4px'>yellow"
                  "</span> = prompts disagree · <span style='background:#5cb85c;color:#fff;"
                  "padding:0 4px'>green</span> = suggested value (mode of the prompts; kept if "
                  "you do nothing) · blue = your choice · click = use that prompt's value · "
                  "my value = choose your own value from the list (wins over click) · a tie "
                  "has no suggestion: choose · nothing is saved until you press Save</small>")
        table = w.GridBox(cells, layout=w.Layout(
            grid_template_columns=f"150px repeat({len(self.versions)}, 72px) 92px 110px 64px",
            grid_gap="4px 4px", align_items="center"))
        self.grid.children = [table, w.HTML(legend)]
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
                if name == "state":
                    tp = self.tri_photo.get((oid, int(r["position"]))) or {}
                    value = TRIAGE_STATE.get(tp.get("content"), default)
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
        cur = self.rows.loc[self.i]
        if self.bbox is not None:
            same = int(p["photo_id"]) == int(cur["photo_id"])
            self._show_in_bbox(p["file_name"], self.box_value if same
                               else self.model_boxes.get(int(p["photo_id"])), editable=same)
        else:
            self.image.value = self._with_box(self._read(p["file_name"], self.big_dirs),
                                              int(p["photo_id"]))
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
        mv = self._model_values(field)
        decided = [x for x in mv.values() if x not in NO_VALUE]
        differ = len(set(decided)) > 1 or len(decided) < len(mv)
        is_mode = source.startswith("mode:")
        for v, b in self.cells[field].items():
            if v == chosen:
                b.button_style = "info"                  # tu clic
            elif is_mode and mv.get(v) == value:
                b.button_style = "success"               # forman la moda sugerida
            else:
                b.button_style = "warning" if differ else ""
        self.labels[field].value = (
            "<b style='display:inline-block;width:170px;padding:2px 4px;"
            + ("background:#ffe08a;color:#000" if differ else "") + f"'>{field}"
            + (" ⚠" if differ else "") + "</b>")
        if not value:
            txt = "<b style='color:#b00'>choose a value</b>" if differ else ""
        else:
            err = validate(field, value) if source == "manual" else None
            color = "#b00" if err else "#000"
            shown = (f"suggested: mode {source.split(':', 1)[1]}" if source.startswith("mode:")
                     else source)
            txt = (f"<b style='color:{color}'>{html.escape(value)}</b> "
                   f"<small style='color:#666'>({shown})</small>"
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
            errors.append("n_cuttlebones_visible: choose a value")
        elif csrc == "manual" and validate("n_cuttlebones_visible", count):
            errors.append(validate("n_cuttlebones_visible", count))
        one = count == "1"
        fields = list(MODEL_FIELDS) if one else ["n_cuttlebones_visible"]
        for f in fields[1:]:
            v, s = final[f]
            if not v:
                errors.append(f"{f}: choose a value")
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
               "triage_version": (self.cfg.triage or {}).get("prompt_version")
               if TRIAGE in self.cols else None,
               "label_source": "assisted",
               "needs_manual_annotation": not one,
               "saved_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        if one and self.model_boxes:
            from .annotate import box_source
            rec["crop_box"] = self.box_value
            rec["model_box"] = self.box_model
            rec["box_source"] = box_source(self.box_value, self.box_model)
            rec["box_reviewed"] = True          # recuadro mirado aqui (aceptado, rehecho o quitado)
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
        """Muestra la interfaz en la celda del notebook."""
        from IPython.display import display
        display(self.box)
