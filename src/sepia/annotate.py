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
    ("notes", "Notes", "text", "all"),
]
FIELD_NAMES = [f[0] for f in FIELDS]

# Valores sugeridos al marcar un hueso nuevo (se pueden cambiar). "—" (vacio) queda
# reservado para "no se puede evaluar" (lado tapado, foto borrosa...): no es lo mismo que "no".
BONE_DEFAULTS = {"bitemarks": "no", "epibionts": "no"}

KEYS = {"individual": ["observation_id", "photo_id", "individual"],
        "empty_photo": ["observation_id", "photo_id"],
        "exclude_record": ["observation_id"]}


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


def save_individual(cfg, annotator, observation_id, photo_id, individual, values, round_=None):
    """Guarda (o corrige) un individuo en una foto. values: dict con campos de FIELDS."""
    values = {k: v for k, v in values.items() if k in FIELD_NAMES}
    return _write(cfg, annotator, "individual",
                  {"observation_id": int(observation_id), "photo_id": int(photo_id),
                   "individual": individual, "deleted": False, **values}, round_)


def delete_individual(cfg, annotator, observation_id, photo_id, individual, round_=None):
    return _write(cfg, annotator, "individual",
                  {"observation_id": int(observation_id), "photo_id": int(photo_id),
                   "individual": individual, "deleted": True}, round_)


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
        for c in ("deleted", "empty", "excluded_reason"):
            if c not in d:
                d[c] = None
        d = d.drop_duplicates(subset=key, keep="last")
        if kind == "individual":
            d = d[d["deleted"] != True]          # noqa: E712 (puede ser NaN en lineas viejas)
        if kind == "empty_photo":
            d = d[d["empty"] == True]            # noqa: E712
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
    """Fotos ya anotadas: con al menos un individuo, o marcadas sin individuos."""
    done = set()
    for k in ("individual", "empty_photo"):
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

class AnnotationApp:
    """
    Interfaz de anotacion para Colab. Uso:
        app = AnnotationApp(cfg, annotator="HZ", round_=1)
        app.show()
    Retoma en la primera foto pendiente. No muestra resultados de modelos (anotacion a ciegas).
    Tamano de la foto: image_width / image_height (maximos, p. ej. "400px", "480px").
    Registros con varias fotos: arriba se ven todas en miniatura (thumb_px); clic = ir a esa foto.
    """

    def __init__(self, cfg: Config, annotator: str, round_=None, ids=None, image_px=1024,
                 image_width="480px", image_height="560px", thumb_px="140px"):
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
        self.table = w.HTML()
        self.letter = w.Dropdown(description="Individual", layout=w.Layout(width="260px"))
        self.inputs = {}
        for name, label, opts, group in FIELDS:
            if opts == "text":
                wid = w.Text(description=label, layout=w.Layout(width="320px"))
            else:
                wid = w.Dropdown(description=label, options=[("—", None)] + [(str(o), o) for o in opts],
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

        # Tres columnas: foto | formulario | botones (todo a la vista, sin bajar).
        # Save = guardar y pasar a la siguiente foto (lo habitual).
        # Para otro individuo en la MISMA foto: "Save + another here".
        col = w.Layout(width="220px")
        self.buttons = w.VBox([
            btn("Save ▶", self.on_save_next, "success"),
            btn("Save + another here", self.on_save, "info"),
            btn("Delete individual", self.on_delete, "danger"),
            w.HTML("<hr style='margin:6px 0'>"),
            w.HBox([btn("◀ Prev", self.on_prev), btn("Next ▶", self.on_next)]),
            btn("Next pending ⏭", self.on_pending),
            btn("Photo without individuals", self.on_empty, "warning"),
            w.HTML("<hr style='margin:6px 0'>"),
            self.reason,
            w.HBox([btn("Exclude record", self.on_exclude, "danger"),
                    btn("Undo", self.on_unexclude)]),
            w.HTML("<hr style='margin:6px 0'>"),
            w.HBox([self.jump]), btn("Go to ID", self.on_jump),
            self.msg,
        ], layout=col)
        self.letter.observe(lambda ch: self._safe(self._load_letter), names="value")
        self._loading = False
        self.inputs["state"].observe(lambda ch: self._safe(lambda: self._on_state(ch)), names="value")
        self.inputs["epibionts"].observe(lambda ch: self._visibility(), names="value")

        form = w.VBox([self.table, self.letter] + list(self.inputs.values()),
                      layout=w.Layout(margin="0 12px"))
        self.box = w.VBox([self.header, self.strip,
                           w.HBox([self.image, form, self.buttons])])
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

    def _render_strip(self, oid, done):
        """Miniaturas de todas las fotos del registro; clic = ir a esa foto."""
        w = self.w
        rows = self.queue.index[self.queue["observation_id"] == oid]
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
            self.table.value = "<b>Photo without individuals</b>" if is_empty else \
                               "<i>No individuals yet in this photo</i>"

        letters = letters_in_record(self.events, oid)
        here_letters = set(here["individual"]) if len(here) else set()
        new = next_letter(letters)
        opts = [(f"{l} (this photo)" if l in here_letters else f"{l} (other photo)", l)
                for l in letters] + [(f"{new} (new)", new)]
        self.letter.options = opts
        self.letter.value = sorted(here_letters)[0] if here_letters else new
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
        return vals

    def _go(self, i):
        self.i = max(0, min(i, len(self.queue) - 1))
        self._render()

    # --- acciones ---
    def on_save(self):
        vals = self._values()
        if vals["state"] is None:
            raise ValueError("choose a state first")
        r = self.row
        empty = self.events["empty_photo"]
        if ((empty["observation_id"] == r["observation_id"]) & (empty["photo_id"] == r["photo_id"])).any():
            mark_empty_photo(self.cfg, self.annotator, r["observation_id"], r["photo_id"], False, self.round)
        save_individual(self.cfg, self.annotator, r["observation_id"], r["photo_id"],
                        self.letter.value, vals, self.round)
        letter = self.letter.value
        self._reload()
        self._render()
        self.msg.value = f"<b style='color:#070'>Saved individual {letter}</b>"

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
