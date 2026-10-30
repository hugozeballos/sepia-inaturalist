"""
Anotacion manual: una fila por foto x individuo, guardada en JSON Lines.

Archivo: annotations/manual_<anotador>.jsonl (uno por anotador, solo se agregan lineas).
Cada linea es un evento con 'kind':
  - "individual":   un individuo en una foto (clave observation_id + photo_id + individual).
                    Con deleted=True se anula (p. ej. un individuo agregado por error).
  - "empty_photo":  la foto no muestra ningun individuo evaluable (clave observation_id + photo_id).
                    empty=False lo deshace.
  - "exclude_record": excluye el registro con un motivo (clave observation_id).
                    Un motivo vacio deshace la exclusion.
Para cada clave vale la ultima linea guardada: corregir = guardar de nuevo.

La logica (cola de trabajo, guardar, leer) no depende de la interfaz y tiene pruebas.
La interfaz (AnnotationApp) es una capa delgada con ipywidgets para Colab.
"""
from __future__ import annotations

import datetime as dt
import string
from pathlib import Path

import pandas as pd

from .config import Config
from .io import paths, read_jsonl, append_jsonl
from .inat import load_tables

SCHEMA_VERSION = 1

# Campos por individuo: (nombre interno, etiqueta, opciones o "text", grupo)
# grupo "all": siempre; "bone": solo si state = cuttlebone; "barnacle": solo si epibionts = yes
FIELDS = [
    ("state", "State", ["cuttlebone", "live animal", "egg case"], "all"),
    ("side", "Side", ["dorsal", "ventral", "unclear"], "bone"),
    ("photo_quality", "Photo quality", ["high", "medium", "low"], "bone"),
    ("erosion", "Erosion", [1, 2, 3], "bone"),
    ("bitemarks", "Bitemarks", ["yes", "no", "(yes)"], "bone"),
    ("epibionts", "Epibionts", ["yes", "no"], "bone"),
    ("taxonomic_groups", "Taxonomic groups", "text", "barnacle"),
    ("barnacles_n", "N° barnacles", ["0", "1-10", "10-100", ">100"], "barnacle"),
    ("barnacle_size", "Barnacle size", ["small", "medium", "large"], "barnacle"),
    ("barnacle_taxon", "Barnacle genus/species", "text", "barnacle"),
    ("certainty", "Certainty", [1, 0], "all"),
    ("possible_duplicate_of", "Same bone as (ID:letter)", "text", "all"),
    ("notes", "Notes", "text", "all"),
]
FIELD_NAMES = [f[0] for f in FIELDS]

# Mismo hueso en OTRO registro: "<observation_id>" o "<observation_id>:<letra>",
# varios separados por coma. Solo marca (no excluye); pendiente con Martin que hacer con ellos.
import re as _re
_DUP = _re.compile(r"^\d+(:[a-z][a-z0-9]*)?$")


def parse_duplicates(text):
    """'405594162:a, 409584581' -> '405594162:a, 409584581' (normalizado) o None. Error si no es valido."""
    if not text or not str(text).strip():
        return None
    # tolerante con espacios: "337987515 :a", "337987515 : a" -> "337987515:a"
    items = [_re.sub(r"\s+", "", t).lower() for t in str(text).replace(";", ",").split(",")
             if t.strip()]
    bad = [t for t in items if not _DUP.match(t)]
    if bad:
        raise ValueError(f"'Same bone as' must be IDs like 405594162 or 405594162:a (not {bad})")
    return ", ".join(items)

# Definiciones (las mismas del prompt photo_traits_v2): se muestran al pasar el mouse
HELP = {
    "side": "dorsal: smoother, harder, granular, no growth lines · ventral: laminated, "
            "fine arc-shaped growth lines, often bite marks · unclear: edge-on, covered, blurry",
    "photo_quality": "only what HIDES the bone lowers it · high: whole bone visible, in focus, "
                     "nothing on top · medium: small areas hidden (grass, seaweed, fingers, wire, "
                     "litter, sand layer) · low: more than half hidden, OR blurry, OR too small/far. "
                     "Marks, bite marks, epibionts, stains, isolated grains do NOT lower it; "
                     "a patch/film of sand or seaweed ON the bone DOES",
    "erosion": "1: outline complete (edge at most slightly worn) · 2: pieces missing, in total "
               "<10% · 3: 10% or more missing (incl. fragments) [threshold from Martin's guide]",
    "possible_duplicate_of": "the same bone appears in ANOTHER iNaturalist record: its ID, "
                             "optionally with the letter there (405594162:a). Several: comma-separated",
    "bitemarks": "yes: clear (triangular nicks, beak shape) · (yes): probable · no: ventral side "
                 "visible, no marks · —: cannot be assessed",
}

# Valores sugeridos al marcar un hueso nuevo (se pueden cambiar). "—" (vacio) queda
# reservado para "no se puede evaluar" (lado tapado, foto borrosa...): no es lo mismo que "no".
BONE_DEFAULTS = {"bitemarks": "no", "epibionts": "no"}

KEYS = {"individual": ["observation_id", "photo_id", "individual"],
        "empty_photo": ["observation_id", "photo_id"],
        "photo_info": ["observation_id", "photo_id"],
        "duplicate_photo": ["observation_id", "photo_id"],
        "exclude_record": ["observation_id"]}

# Valor por defecto del estado para un individuo nuevo (casi todo son huesos)
DEFAULT_STATE = "cuttlebone"


# --------------------------------------------------------------------------
# Recuadros (crop_box): coordenadas relativas 0-1 sobre la foto ya enderezada
# (EXIF aplicado), asi sirven para cualquier tamano: 512 px, 1024 px u original.
# --------------------------------------------------------------------------

def to_crop_box(bbox: dict, width: int, height: int) -> dict:
    """
    Recuadro en pixeles {x, y, width, height} -> relativo {x0, y0, x1, y1} en 0-1.
    Acepta recuadros dibujados en cualquier direccion (ancho o alto negativos) y
    recorta lo que se sale de la foto.
    """
    xa, xb = sorted((bbox["x"], bbox["x"] + bbox["width"]))
    ya, yb = sorted((bbox["y"], bbox["y"] + bbox["height"]))
    clip = lambda v: min(1.0, max(0.0, v))
    return {"x0": round(clip(xa / width), 4), "y0": round(clip(ya / height), 4),
            "x1": round(clip(xb / width), 4), "y1": round(clip(yb / height), 4)}


def from_crop_box(box: dict, width: int, height: int) -> dict:
    """Relativo {x0, y0, x1, y1} -> pixeles {x, y, width, height} para una foto de ese tamano."""
    return {"x": box["x0"] * width, "y": box["y0"] * height,
            "width": (box["x1"] - box["x0"]) * width, "height": (box["y1"] - box["y0"]) * height}


def format_crop_box(box) -> str:
    """{x0, y0, x1, y1} -> 'x0, y0, x1, y1' (texto del campo de la interfaz)."""
    if not isinstance(box, dict):
        return ""
    return ", ".join(f"{box[k]:.4f}" for k in ("x0", "y0", "x1", "y1"))


def parse_crop_box(text: str):
    """'x0, y0, x1, y1' -> dict, o None si el campo esta vacio. Error si no es valido."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in text.replace(";", ",").split(","))
    except ValueError:
        raise ValueError("crop box must be 'x0, y0, x1, y1' (numbers 0-1) or empty")
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"crop box '{text}': need 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 "
                         "(draw again or clear with ✕)")
    return {"x0": round(x0, 4), "y0": round(y0, 4), "x1": round(x1, 4), "y1": round(y1, 4)}


def crop_photo(path, box: dict):
    """Recorta un individuo de una foto (cualquier tamano). Devuelve una imagen PIL."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")   # igual que las versiones reducidas
        W, H = im.size
        return im.crop((round(box["x0"] * W), round(box["y0"] * H),
                        round(box["x1"] * W), round(box["y1"] * H)))


# --------------------------------------------------------------------------
# Lectura y escritura
# --------------------------------------------------------------------------

def annotation_file(cfg: Config, annotator: str) -> Path:
    return paths(cfg)["annotations"] / f"manual_{annotator}.jsonl"


def _now() -> str:
    # Siempre UTC y con milisegundos: el orden de 'saved_at' es el orden real
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _write(cfg, annotator, kind, record, round_=None):
    line = {"schema_version": SCHEMA_VERSION, "kind": kind, **record,
            "annotator": annotator, "label_source": "annotator", "round": round_,
            "saved_at": _now()}
    append_jsonl(line, annotation_file(cfg, annotator))
    return line


def save_individual(cfg, annotator, observation_id, photo_id, individual, values, round_=None,
                    crop_box=None):
    """
    Guarda (o corrige) un individuo en una foto. values: dict con campos de FIELDS.
    crop_box (opcional): recuadro relativo {x0, y0, x1, y1}, ver to_crop_box.
    """
    values = {k: v for k, v in values.items() if k in FIELD_NAMES}
    return _write(cfg, annotator, "individual",
                  {"observation_id": int(observation_id), "photo_id": int(photo_id),
                   "individual": individual, "deleted": False, **values,
                   "crop_box": crop_box}, round_)


def delete_individual(cfg, annotator, observation_id, photo_id, individual, round_=None):
    return _write(cfg, annotator, "individual",
                  {"observation_id": int(observation_id), "photo_id": int(photo_id),
                   "individual": individual, "deleted": True}, round_)


def mark_duplicate_photo(cfg, annotator, observation_id, photo_id, of_photo_id, round_=None):
    """
    La foto repite a otra del MISMO registro (mismo lado, mismo encuadre, no aporta nada):
    no se anotan individuos en ella. of_photo_id=None deshace la marca.
    """
    if of_photo_id is not None and int(of_photo_id) == int(photo_id):
        raise ValueError("a photo cannot be a duplicate of itself")
    return _write(cfg, annotator, "duplicate_photo",
                  {"observation_id": int(observation_id), "photo_id": int(photo_id),
                   "duplicate_of_photo": None if of_photo_id is None else int(of_photo_id)}, round_)


def save_photo_count(cfg, annotator, observation_id, photo_id, n_in_photo, round_=None):
    """
    Numero de huesos (individuos) que SE VEN en la foto, anotados o no en detalle.
    Sirve para comparar con el conteo por foto del triaje.
    """
    n = int(n_in_photo)
    if n < 0:
        raise ValueError("individuals in photo must be >= 0")
    return _write(cfg, annotator, "photo_info",
                  {"observation_id": int(observation_id), "photo_id": int(photo_id),
                   "n_in_photo": n}, round_)


def mark_empty_photo(cfg, annotator, observation_id, photo_id, empty=True, round_=None):
    return _write(cfg, annotator, "empty_photo",
                  {"observation_id": int(observation_id), "photo_id": int(photo_id),
                   "empty": bool(empty)}, round_)


def exclude_record(cfg, annotator, observation_id, reason, round_=None):
    """Excluye un registro. reason='' deshace la exclusion."""
    return _write(cfg, annotator, "exclude_record",
                  {"observation_id": int(observation_id), "excluded_reason": reason or ""},
                  round_)


def load_events(cfg: Config) -> dict[str, pd.DataFrame]:
    """
    Lee todas las anotaciones (annotations/*.jsonl) y aplica 'vale la ultima' por clave.
    Devuelve {'individual': ..., 'empty_photo': ..., 'exclude_record': ...}, ya filtrados:
    sin individuos borrados, sin fotos des-marcadas, sin exclusiones deshechas.
    Lineas antiguas sin 'kind' se tratan como 'individual'.
    """
    p = paths(cfg)
    frames = [read_jsonl(f) for f in sorted(p["annotations"].glob("*.jsonl"))]
    frames = [f for f in frames if len(f)]
    extra = {"individual": ["deleted"] + FIELD_NAMES, "empty_photo": ["empty"],
             "photo_info": ["n_in_photo"], "duplicate_photo": ["duplicate_of_photo"],
             "exclude_record": ["excluded_reason"]}
    out = {k: pd.DataFrame(columns=v + extra[k]) for k, v in KEYS.items()}
    if not frames:
        return out
    ann = pd.concat(frames, ignore_index=True)
    if "kind" not in ann:
        ann["kind"] = "individual"
    ann["kind"] = ann["kind"].fillna("individual")
    if "saved_at" in ann:
        ann = ann.sort_values("saved_at", kind="stable")   # empates: orden del archivo

    for kind, key in KEYS.items():
        d = ann[ann["kind"] == kind].copy()
        for k in key:
            if k not in d:
                d[k] = None
        if kind != "exclude_record":
            missing = d["photo_id"].isna()
            if missing.any():
                print(f"AVISO: {int(missing.sum())} anotaciones '{kind}' sin photo_id, no se usan")
                d = d[~missing]
            d["photo_id"] = d["photo_id"].astype("int64")
        d["observation_id"] = d["observation_id"].astype("int64")
        for c in ("deleted", "empty", "excluded_reason", "n_in_photo", "duplicate_of_photo"):
            if c not in d:
                d[c] = None
        d = d.drop_duplicates(subset=key, keep="last")
        if kind == "individual":
            d = d[d["deleted"] != True]          # noqa: E712 (puede ser NaN en lineas viejas)
        if kind == "empty_photo":
            d = d[d["empty"] == True]            # noqa: E712
        if kind == "duplicate_photo":
            d = d[d["duplicate_of_photo"].notna()]
        if kind == "exclude_record":
            d = d[d["excluded_reason"].fillna("") != ""]
        out[kind] = d.reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Cola de trabajo
# --------------------------------------------------------------------------

def work_queue(cfg: Config, ids=None) -> pd.DataFrame:
    """Una fila por foto, en orden: registro (ID) y posicion de la foto en el registro."""
    obs, pho = load_tables(cfg)
    q = (pho[["observation_id", "photo_id", "position", "file_name"]]
         .merge(obs[["observation_id", "url", "observed_on", "place_guess", "n_photos"]],
                on="observation_id", how="left"))
    if ids is not None:
        q = q[q["observation_id"].isin({int(i) for i in ids})]
    return q.sort_values(["observation_id", "position"]).reset_index(drop=True)


def done_photos(events) -> set[tuple[int, int]]:
    """
    Fotos ya anotadas: con al menos un individuo, marcadas sin individuos, o con
    solo el conteo guardado (fotos con muchos huesos).
    """
    done = set()
    for k in ("individual", "empty_photo", "photo_info", "duplicate_photo"):
        d = events[k]
        done |= set(zip(d["observation_id"].astype(int), d["photo_id"].astype(int)))
    return done


def first_pending(queue: pd.DataFrame, events) -> int | None:
    """Indice de la primera foto pendiente (sin anotar y de un registro no excluido)."""
    done = done_photos(events)
    excluded = set(events["exclude_record"]["observation_id"].astype(int))
    for i, r in enumerate(queue.itertuples(index=False)):
        if r.observation_id in excluded:
            continue
        if (r.observation_id, r.photo_id) not in done:
            return i
    return None


def letters_in_record(events, observation_id) -> list[str]:
    d = events["individual"]
    return sorted(set(d.loc[d["observation_id"] == observation_id, "individual"].dropna()))


def next_letter(existing) -> str:
    """Siguiente letra libre: a, b, c ... z, luego z1, z2 ..."""
    for c in string.ascii_lowercase:
        if c not in existing:
            return c
    n = 1
    while f"z{n}" in existing:
        n += 1
    return f"z{n}"


# --------------------------------------------------------------------------
# Interfaz (Colab)
# --------------------------------------------------------------------------

# Atajos de teclado: Alt + tecla (no chocan con lo que se escribe en los campos de texto).
# Se usa e.code (posicion fisica de la tecla): funciona igual con teclado en espanol o ingles.
SHORTCUTS = {"KeyS": "save_next", "KeyA": "save_here", "KeyN": "next", "KeyP": "prev"}

_KEYS_ESM = """
function toTop() {
  // ir al inicio de la interfaz y dejar el foco en el primer campo del formulario
  const app = document.querySelector(".sepia-app");
  if (app) app.scrollIntoView({ block: "start", behavior: "instant" });
  const first = document.querySelector(".sepia-form select, .sepia-form input");
  if (first) first.focus({ preventScroll: true });
}
export default {
  render({ model, el }) {
    el.innerHTML = "<small style='opacity:.7'>⌨ Alt+S save ▶ · Alt+A save + another · "
                 + "Alt+N next · Alt+P prev · Alt+T top of form</small>";
    const codes = model.get("codes");
    const onKey = (e) => {
      if (!e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.code === "KeyT") { e.preventDefault(); toTop(); return; }
      if (!codes.includes(e.code)) return;
      e.preventDefault();
      e.stopPropagation();
      model.send({ code: e.code });
    };
    document.addEventListener("keydown", onKey, true);
    // Python sube 'tick' al cambiar de foto: volver arriba automaticamente
    model.on("change:tick", () => setTimeout(toTop, 50));
    return () => document.removeEventListener("keydown", onKey, true);
  }
};
"""


def _make_key_listener():
    """Control invisible que escucha el teclado. None si anywidget no esta instalado."""
    try:
        import anywidget
        import traitlets
    except ImportError:
        return None

    class KeyListener(anywidget.AnyWidget):
        _esm = _KEYS_ESM
        codes = traitlets.List(traitlets.Unicode()).tag(sync=True)
        tick = traitlets.Int(0).tag(sync=True)      # cambia al pasar de foto -> volver arriba

    return KeyListener(codes=list(SHORTCUTS))

class AnnotationApp:
    """
    Interfaz de anotacion para Colab. Uso:
        app = AnnotationApp(cfg, annotator="HZ", round_=1)
        app.show()
    Retoma en la primera foto pendiente. No muestra resultados de modelos (anotacion a ciegas).
    Tamano de la foto: image_width / image_height (maximos, p. ej. "400px", "480px").
    Registros con varias fotos: arriba se ven todas en miniatura (thumb_px); clic = ir a esa foto.
    Recuadros (opcional, boxes=True): elegir la letra, boton 'Draw box' y arrastrar sobre la foto;
    al soltar, las coordenadas aparecen en el campo 'Crop box' y se guardan con el individuo.
    Campo vacio = sin recuadro.
    """

    def __init__(self, cfg: Config, annotator: str, round_=None, ids=None, image_px=1024,
                 image_width="480px", image_height="560px", thumb_px="140px", boxes=True):
        import ipywidgets as w
        self.w, self.cfg, self.annotator, self.round = w, cfg, annotator, round_
        self.queue = work_queue(cfg, ids)
        if self.queue.empty:
            raise RuntimeError("No hay fotos: correr la descarga (tables y photos) primero")
        base = paths(cfg)["base"] / "photos"
        self.photo_dirs = [base / f"{image_px}px", paths(cfg)["photos"]]
        self.thumb_dirs = [base / "512px"] + self.photo_dirs      # miniaturas: la version liviana
        self.thumb_px = thumb_px
        self._thumb_cache = {}
        self.events = load_events(cfg)
        self.i = first_pending(self.queue, self.events) or 0

        # --- widgets ---
        self.header = w.HTML()
        self.strip = w.HBox(layout=w.Layout(flex_flow="row wrap"))   # todas las fotos del registro
        # Ancho y alto maximos: las fotos verticales tampoco se salen de la pantalla
        self.image = w.Image(format="jpeg", layout=w.Layout(
            max_width=image_width, max_height=image_height, object_fit="contain"))
        # Recuadros opcionales (jupyter-bbox-widget). Si no esta instalado: foto sin recuadros.
        self.bbox = None
        if boxes:
            try:
                from jupyter_bbox_widget import BBoxWidget
                try:                                   # Colab: habilitar widgets externos
                    from google.colab import output
                    output.enable_custom_widget_manager()
                except ImportError:
                    pass
                self.bbox = BBoxWidget(hide_buttons=True, view_only=True,
                                       layout=w.Layout(width=image_width))
            except ImportError:
                self._no_box_reason = ("jupyter-bbox-widget is not installed: run "
                                       "!pip install -q jupyter-bbox-widget and restart the app")
                print("AVISO:", self._no_box_reason)
        self.viewer = self.bbox if self.bbox is not None else self.image
        self._box_updating = False
        self.table = w.HTML()
        # A nivel de FOTO: cuantos huesos se ven (anotados o no en detalle). 1 por defecto.
        # Misma regla que photo_traits_v4: cuenta si se ve al menos ~la mitad del hueso.
        self.n_photo = w.BoundedIntText(value=1, min=0, max=999, description="Individuals in photo",
                                        tooltip="bones with at least ~half visible (whole, half-bone "
                                                "fragment, or cut by the photo edge with half inside); "
                                                "small pieces at the edge do not count",
                                        layout=w.Layout(width="260px"))
        self.n_photo.style.description_width = "160px"
        self.letter = w.Dropdown(description="Individual", layout=w.Layout(width="260px"))
        self.inputs = {}
        for name, label, opts, group in FIELDS:
            if opts == "text":
                wid = w.Text(description=label, layout=w.Layout(width="320px"),
                             tooltip=HELP.get(name, ""),
                             placeholder="e.g. 405594162:a" if name == "possible_duplicate_of" else "")
            else:
                wid = w.Dropdown(description=label, options=[("—", None)] + [(str(o), o) for o in opts],
                                 tooltip=HELP.get(name, ""),
                                 layout=w.Layout(width="260px"))
            wid.style.description_width = "120px"
            self.inputs[name] = wid
        self.letter.style.description_width = "120px"
        self.msg = w.HTML()
        self.reason = w.Text(placeholder="exclusion reason", layout=w.Layout(width="220px"))
        self.jump = w.Text(placeholder="iNaturalist ID", layout=w.Layout(width="220px"))

        def btn(text, cb, style=""):
            b = w.Button(description=text, button_style=style, layout=w.Layout(width="100%"))
            b.on_click(lambda _: self._safe(cb))
            return b

        # Foto repetida: "igual que la foto N" (mismo lado y encuadre; no se anota)
        self.dup_of = w.Dropdown(options=[], layout=w.Layout(width="110px"),
                                 tooltip="this photo repeats another photo of the record: "
                                         "same side and framing, adds nothing new")
        # Tres columnas: foto | formulario | botones (todo a la vista, sin bajar).
        # Save = guardar y pasar a la siguiente foto (lo habitual).
        # Para otro individuo en la MISMA foto: "Save + another here".
        col = w.Layout(width="220px")
        self.buttons = w.VBox([
            btn("Save ▶", self.on_save_next, "success"),
            btn("Save + another here", self.on_save_here, "info"),
            btn("Delete individual", self.on_delete, "danger"),
            w.HTML("<hr style='margin:6px 0'>"),
            w.HBox([btn("◀ Prev", self.on_prev), btn("Next ▶", self.on_next)]),
            btn("Next pending ⏭", self.on_pending),
            btn("Photo without individuals", self.on_empty, "warning"),
            btn("Save count only ▶", self.on_count_only, "warning"),
            w.HBox([self.dup_of, btn("Same as ▶", self.on_duplicate, "warning")]),
            w.HTML("<hr style='margin:6px 0'>"),
            self.reason,
            w.HBox([btn("Exclude record", self.on_exclude, "danger"),
                    btn("Undo", self.on_unexclude)]),
            w.HTML("<hr style='margin:6px 0'>"),
            w.HBox([self.jump]), btn("Go to ID", self.on_jump),
            self.msg,
        ], layout=col)
        self.letter.observe(lambda ch: self._safe(self._load_letter), names="value")
        # Recuadro: boton "Draw box" -> arrastrar sobre la foto -> al soltar, las
        # coordenadas (relativas 0-1) aparecen en el campo "Crop box" (editable; vacio = sin recuadro)
        self.crop = w.Text(description="Crop box", placeholder="x0, y0, x1, y1 (optional)",
                           layout=w.Layout(width="320px"))
        self.crop.style.description_width = "120px"
        self.draw_btn = w.ToggleButton(description="▭ Draw box", layout=w.Layout(width="110px"),
                                       disabled=self.bbox is None,
                                       tooltip="click, then drag a rectangle on the photo")
        if self.bbox is None:           # que se vea en la interfaz por que no se puede dibujar
            self.draw_btn.tooltip = getattr(self, "_no_box_reason", "boxes disabled (boxes=False)")
            self.draw_btn.description = "▭ (no boxes)"
        clear_btn = w.Button(description="✕", layout=w.Layout(width="40px"), tooltip="remove box")
        clear_btn.on_click(lambda _: self._safe(self._clear_box))
        self.draw_btn.observe(lambda ch: self._safe(lambda: self._toggle_draw(ch["new"])), names="value")
        if self.bbox is not None:
            self.bbox.observe(lambda ch: self._safe(lambda: self._on_box_drawn(ch["new"])),
                              names="bboxes")
        self._loading = False
        self.inputs["state"].observe(lambda ch: self._safe(lambda: self._on_state(ch)), names="value")
        self.inputs["epibionts"].observe(lambda ch: self._visibility(), names="value")

        form = w.VBox([self.table, self.n_photo, w.HTML("<hr style='margin:4px 0'>"), self.letter,
                       w.HBox([self.crop, self.draw_btn, clear_btn])] + list(self.inputs.values()),
                      layout=w.Layout(margin="0 12px"))
        # Atajos de teclado (opcional: requiere anywidget, que instala jupyter-bbox-widget)
        self.keys = _make_key_listener()
        actions = {"save_next": self.on_save_next, "save_here": self.on_save_here,
                   "next": self.on_next, "prev": self.on_prev}
        if self.keys is not None:
            self.keys.on_msg(lambda _w, content, _b: self._safe(
                actions[SHORTCUTS[content["code"]]]) if content.get("code") in SHORTCUTS else None)
        extra = [self.keys] if self.keys is not None else []
        form.add_class("sepia-form")
        self.box = w.VBox([self.header, self.strip,
                           w.HBox([self.viewer, form, self.buttons])] + extra)
        self.box.add_class("sepia-app")
        self._render()

    # --- utilidades ---
    def show(self):
        from IPython.display import display
        display(self.box)

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:                       # mostrar el error sin romper la interfaz
            self.msg.value = f"<b style='color:#b00'>Error: {e}</b>"

    @property
    def row(self):
        return self.queue.iloc[self.i]

    def _reload(self):
        self.events = load_events(self.cfg)

    @staticmethod
    def _read_photo(file_name, dirs):
        name = Path(file_name).stem + ".jpg"
        for d in dirs:
            for f in (d / name, d / file_name):
                if f.exists():
                    return f.read_bytes()
        return b""

    def _photo_bytes(self):
        return self._read_photo(self.row["file_name"], self.photo_dirs)

    def _photo_path(self):
        name = Path(self.row["file_name"]).stem + ".jpg"
        for d in self.photo_dirs:
            for f in (d / name, d / self.row["file_name"]):
                if f.exists():
                    return f
        return None

    def _image_size(self, path):
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).size

    def _render_boxes(self, here, options):
        """Muestra en la foto los recuadros ya guardados de esta foto."""
        path = self._photo_path()
        if self.bbox is None or path is None:
            return
        W, H = self._image_size(path)
        self._box_size = (W, H)
        self._box_updating = True
        self.bbox.classes = [v for _, v in options]
        self.bbox.image = str(path)
        boxes = []
        if len(here) and "crop_box" in here:
            for r in here.itertuples(index=False):
                cb = getattr(r, "crop_box", None)
                if isinstance(cb, dict):
                    boxes.append({**from_crop_box(cb, W, H), "label": r.individual})
        self.bbox.bboxes = boxes
        self._box_updating = False

    def _toggle_draw(self, on):
        """Boton 'Draw box': habilita dibujar un recuadro con la letra elegida."""
        if self.bbox is None:
            return
        self.bbox.label = self.letter.value or ""
        self.bbox.view_only = not on
        self.draw_btn.button_style = "warning" if on else ""
        if on:
            self.msg.value = f"Drag a rectangle around individual <b>{self.letter.value}</b>"

    def _on_box_drawn(self, boxes):
        """Al soltar: deja un solo recuadro por letra y muestra sus coordenadas en el campo."""
        if self._box_updating or not self.draw_btn.value:
            return
        letter = self.letter.value
        mine = [b for b in boxes if b.get("label") == letter]
        if not mine or not getattr(self, "_box_size", None):
            return
        new = mine[-1]
        box = to_crop_box(new, *self._box_size)
        if box["x1"] - box["x0"] < 0.01 or box["y1"] - box["y0"] < 0.01:
            self.msg.value = ("<b style='color:#b00'>Box too small or outside the photo: "
                              "drag again</b>")
            return
        self._box_updating = True                     # evitar recursion al reescribir bboxes
        self.bbox.bboxes = [b for b in boxes if b.get("label") != letter] + [new]
        self._box_updating = False
        self.crop.value = format_crop_box(box)
        self.draw_btn.value = False                   # vuelve a modo solo-ver
        self.msg.value = f"Box for <b>{letter}</b> ready: press Save"

    def _clear_box(self):
        letter = self.letter.value
        self.crop.value = ""
        if self.bbox is not None:
            self._box_updating = True
            self.bbox.bboxes = [b for b in self.bbox.bboxes if b.get("label") != letter]
            self._box_updating = False

    def _render_strip(self, oid, done):
        """Miniaturas de todas las fotos del registro; clic = ir a esa foto."""
        w = self.w
        rows = self.queue.index[self.queue["observation_id"] == oid]
        others = [(f"photo {int(self.queue.loc[j, 'position'])}", int(self.queue.loc[j, "photo_id"]))
                  for j in rows if j != self.i]
        self.dup_of.options = others
        self.dup_of.disabled = not others
        if others:                       # por defecto: la foto anterior del registro (o la primera)
            prev = [v for (_, v), j in zip(others, [j for j in rows if j != self.i]) if j < self.i]
            self.dup_of.value = prev[-1] if prev else others[0][1]
        if len(rows) < 2:
            self.strip.children = ()
            return
        items = []
        for j in rows:
            r = self.queue.loc[j]
            fn = r["file_name"]
            if fn not in self._thumb_cache:
                self._thumb_cache[fn] = self._read_photo(fn, self.thumb_dirs)
            img = w.Image(value=self._thumb_cache[fn], format="jpeg", layout=w.Layout(
                max_width=self.thumb_px, max_height=self.thumb_px, object_fit="contain",
                border="3px solid #1a73e8" if j == self.i else "3px solid transparent"))
            mark = " ✓" if (int(r["observation_id"]), int(r["photo_id"])) in done else ""
            b = w.Button(description=f"Photo {int(r['position'])}{mark}",
                         button_style="info" if j == self.i else "",
                         layout=w.Layout(width=self.thumb_px))
            b.on_click(lambda _, j=j: self._safe(lambda: self._go(int(j))))
            items.append(w.VBox([img, b], layout=w.Layout(margin="0 6px 6px 0")))
        self.strip.children = items

    def _individuals_here(self):
        d = self.events["individual"]
        r = self.row
        return d[(d["observation_id"] == r["observation_id"]) & (d["photo_id"] == r["photo_id"])]

    def _visibility(self):
        is_bone = self.inputs["state"].value == "cuttlebone"
        has_epi = is_bone and self.inputs["epibionts"].value == "yes"
        for name, _, _, group in FIELDS:
            show = group == "all" or (group == "bone" and is_bone) or (group == "barnacle" and has_epi)
            self.inputs[name].layout.display = None if show else "none"

    def _render(self, keep_msg=False):
        r = self.row
        oid, pid = int(r["observation_id"]), int(r["photo_id"])
        done = done_photos(self.events)
        excluded = self.events["exclude_record"]
        ex = excluded.loc[excluded["observation_id"] == oid, "excluded_reason"]
        n_rec = self.queue["observation_id"].nunique()
        rec_i = list(self.queue["observation_id"].unique()).index(oid) + 1
        n_done = sum((int(a), int(b)) in done for a, b in
                     zip(self.queue["observation_id"], self.queue["photo_id"]))
        status = ("<b style='color:#b00'>EXCLUDED: " + ex.iloc[0] + "</b>") if len(ex) else \
                 ("annotated" if (oid, pid) in done else "<b>pending</b>")
        self.header.value = (
            f"<h3 style='margin:4px 0'>Record {rec_i}/{n_rec} · "
            f"<a href='{r['url']}' target='_blank'>{oid}</a> · photo {int(r['position'])}/"
            f"{int(r['n_photos'])} (id {pid}) · {status}</h3>"
            f"{r['observed_on']} · {r['place_guess']} · progress: {n_done}/{len(self.queue)} photos")
        self.image.value = self._photo_bytes()
        self._render_strip(oid, done)

        here = self._individuals_here()
        empty = self.events["empty_photo"]
        is_empty = ((empty["observation_id"] == oid) & (empty["photo_id"] == pid)).any()
        if len(here):
            cols = ["individual"] + [c for c in FIELD_NAMES if c in here and c != "notes"]
            self.table.value = ("<b>In this photo:</b>" +
                                here[cols].fillna("").to_html(index=False, border=0))
        else:
            dups = self.events["duplicate_photo"]
            dup = dups.loc[(dups["observation_id"] == oid) & (dups["photo_id"] == pid),
                           "duplicate_of_photo"]
            if len(dup):
                pos = self.queue.loc[self.queue["photo_id"] == int(dup.iloc[0]), "position"]
                self.table.value = (f"<b>Same as photo {int(pos.iloc[0]) if len(pos) else dup.iloc[0]}"
                                    "</b> (not annotated)")
            else:
                self.table.value = "<b>Photo without individuals</b>" if is_empty else \
                                   "<i>No individuals yet in this photo</i>"

        letters = letters_in_record(self.events, oid)
        here_letters = set(here["individual"]) if len(here) else set()
        new = next_letter(letters)
        opts = [(f"{l} (this photo)" if l in here_letters else f"{l} (other photo)", l)
                for l in letters] + [(f"{new} (new)", new)]
        info = self.events["photo_info"]
        saved_n = info.loc[(info["observation_id"] == oid) & (info["photo_id"] == pid), "n_in_photo"]
        self.n_photo.value = int(saved_n.iloc[0]) if len(saved_n) else (0 if is_empty else 1)
        self.letter.options = opts
        self._render_boxes(here, opts)
        self.letter.value = sorted(here_letters)[0] if here_letters else new
        if self.bbox is not None:
            self.bbox.label = self.letter.value
            self.draw_btn.value = False
        self._load_letter()
        if not keep_msg:
            self.msg.value = ""

    def _apply_defaults(self):
        """Hueso nuevo: sugiere 'no' en picoteo y epibiontes si estan vacios."""
        if self.inputs["state"].value == "cuttlebone":
            for k, v in BONE_DEFAULTS.items():
                if self.inputs[k].value is None:
                    self.inputs[k].value = v

    def _on_state(self, ch):
        if not self._loading:          # solo cuando la persona cambia el estado, no al cargar
            self._apply_defaults()
        self._visibility()

    def _load_letter(self):
        """Carga los valores del individuo elegido (si ya existe en esta foto)."""
        letter = self.letter.value
        here = self._individuals_here()
        cur = here[here["individual"] == letter]
        if len(cur):
            vals = cur.iloc[0].to_dict()
        else:
            # mismo hueso en otra foto: el estado se copia; lo demas depende de la foto
            d = self.events["individual"]
            other = d[(d["observation_id"] == self.row["observation_id"]) & (d["individual"] == letter)]
            vals = {"state": other.iloc[-1].get("state")} if len(other) else {}
            if vals.get("state") is None or vals.get("state") != vals.get("state"):
                vals["state"] = DEFAULT_STATE          # individuo nuevo: hueso por defecto
        self.crop.value = format_crop_box(vals.get("crop_box"))   # recuadro guardado (o vacio)
        if self.bbox is not None:
            self.draw_btn.value = False
            self.bbox.label = letter or ""
        self._loading = True
        for name, wid in self.inputs.items():
            v = vals.get(name)
            v = None if (v is None or (isinstance(v, float) and v != v)) else v
            if isinstance(wid, self.w.Text):
                wid.value = "" if v is None else str(v)
            else:
                valid = [o[1] for o in wid.options]
                wid.value = v if v in valid else (int(v) if str(v).isdigit() and int(v) in valid else None)
        self._loading = False
        if not len(cur):               # individuo nuevo en esta foto: valores sugeridos
            self._apply_defaults()     # (uno ya guardado se muestra tal cual, aunque este vacio)
        self._visibility()

    def _values(self):
        vals = {}
        is_bone = self.inputs["state"].value == "cuttlebone"
        has_epi = is_bone and self.inputs["epibionts"].value == "yes"
        for name, _, _, group in FIELDS:
            visible = group == "all" or (group == "bone" and is_bone) or (group == "barnacle" and has_epi)
            v = self.inputs[name].value if visible else None   # campos ocultos no se guardan
            vals[name] = (v.strip() or None) if isinstance(v, str) else v
        vals["possible_duplicate_of"] = parse_duplicates(vals.get("possible_duplicate_of"))
        own = str(int(self.row["observation_id"]))
        if vals["possible_duplicate_of"] and any(t.split(":")[0] == own
                                                 for t in vals["possible_duplicate_of"].split(", ")):
            raise ValueError("'Same bone as' points to this same record (use the letter instead)")
        return vals

    def _go(self, i):
        self.i = max(0, min(i, len(self.queue) - 1))
        self._render()
        if getattr(self, "keys", None) is not None:
            self.keys.tick += 1                       # el navegador vuelve al inicio del formulario

    # --- acciones ---
    def on_save(self):
        vals = self._values()
        if vals["state"] is None:
            raise ValueError("choose a state first")
        r = self.row
        empty = self.events["empty_photo"]
        if ((empty["observation_id"] == r["observation_id"]) & (empty["photo_id"] == r["photo_id"])).any():
            mark_empty_photo(self.cfg, self.annotator, r["observation_id"], r["photo_id"], False, self.round)
        dups = self.events["duplicate_photo"]
        if ((dups["observation_id"] == r["observation_id"]) & (dups["photo_id"] == r["photo_id"])).any():
            mark_duplicate_photo(self.cfg, self.annotator, r["observation_id"], r["photo_id"],
                                 None, self.round)
        save_individual(self.cfg, self.annotator, r["observation_id"], r["photo_id"],
                        self.letter.value, vals, self.round,
                        crop_box=parse_crop_box(self.crop.value))
        self._save_count()
        letter = self.letter.value
        self._reload()
        n_here = len(self._individuals_here())
        n_saved = self.n_photo.value
        self._render()
        self.msg.value = f"<b style='color:#070'>Saved individual {letter}</b>"
        if n_here > n_saved:
            self.msg.value += (f"<br><b style='color:#b60'>Note: {n_here} individuals annotated but "
                               f"'Individuals in photo' = {n_saved}</b>")

    def _save_count(self):
        """Guarda el conteo por foto solo si cambio (evita lineas repetidas)."""
        r = self.row
        info = self.events["photo_info"]
        prev = info.loc[(info["observation_id"] == r["observation_id"]) &
                        (info["photo_id"] == r["photo_id"]), "n_in_photo"]
        if not len(prev) or int(prev.iloc[0]) != int(self.n_photo.value):
            save_photo_count(self.cfg, self.annotator, r["observation_id"], r["photo_id"],
                             self.n_photo.value, self.round)

    def on_duplicate(self):
        """Marca la foto como repetida de otra del registro y pasa a la siguiente."""
        if len(self._individuals_here()):
            raise ValueError("this photo has individuals: delete them first")
        if self.dup_of.value is None:
            raise ValueError("this record has no other photo")
        r = self.row
        mark_duplicate_photo(self.cfg, self.annotator, r["observation_id"], r["photo_id"],
                             self.dup_of.value, self.round)
        label = self.dup_of.label
        self._reload()
        self._go(self.i + 1)
        self.msg.value = f"<b style='color:#070'>Marked as same as {label} · next photo</b>"

    def on_count_only(self):
        """Foto con muchos huesos: guardar solo el conteo, sin anotar cada individuo."""
        if self.n_photo.value < 1:
            raise ValueError("count only needs 'Individuals in photo' >= 1 (or use 'Photo without individuals')")
        self._save_count()
        self._reload()
        self._go(self.i + 1)
        self.msg.value = "<b style='color:#070'>Count saved · next photo</b>"

    def on_save_here(self):
        """Guarda y se queda en la foto con una letra NUEVA lista para el siguiente individuo."""
        self.on_save()
        msg = self.msg.value
        self.letter.value = self.letter.options[-1][1]     # la ultima opcion es siempre la nueva
        self.msg.value = msg + " · next: new individual"

    def on_save_next(self):
        self.on_save()
        if self.i + 1 < len(self.queue):
            self._go(self.i + 1)
            self.msg.value = "<b style='color:#070'>Saved · next photo</b>"
        else:
            self.msg.value = "<b style='color:#070'>Saved · this was the last photo</b>"

    def on_delete(self):
        r = self.row
        letter = self.letter.value
        if letter not in set(self._individuals_here()["individual"]):
            raise ValueError(f"individual {letter} is not in this photo")
        delete_individual(self.cfg, self.annotator, r["observation_id"], r["photo_id"], letter, self.round)
        self._reload()
        self._render()
        self.msg.value = f"Deleted individual {letter} from this photo"

    def on_empty(self):
        if len(self._individuals_here()):
            raise ValueError("this photo has individuals: delete them first")
        r = self.row
        mark_empty_photo(self.cfg, self.annotator, r["observation_id"], r["photo_id"], True, self.round)
        self.n_photo.value = 0
        self._save_count()
        self._reload()
        self._go(self.i + 1)

    def on_exclude(self):
        reason = self.reason.value.strip()
        if not reason:
            raise ValueError("write the exclusion reason")
        oid = self.row["observation_id"]
        exclude_record(self.cfg, self.annotator, oid, reason, self.round)
        self.reason.value = ""
        self._reload()
        self.on_pending()
        self.msg.value = f"Record {oid} excluded"

    def on_unexclude(self):
        exclude_record(self.cfg, self.annotator, self.row["observation_id"], "", self.round)
        self._reload()
        self._render()

    def on_prev(self):
        self._go(self.i - 1)

    def on_next(self):
        self._go(self.i + 1)

    def on_pending(self):
        i = first_pending(self.queue, self.events)
        if i is None:
            self.msg.value = "<b>All photos annotated 🎉</b>"
            return
        self._go(i)

    def on_jump(self):
        oid = int(self.jump.value.strip())
        idx = self.queue.index[self.queue["observation_id"] == oid]
        if not len(idx):
            raise ValueError(f"ID {oid} is not in the queue")
        self._go(int(idx[0]))
