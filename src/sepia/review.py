"""
Visor de resultados de los modelos: foto + valores de cada version de prompt, lado a lado,
y la anotacion manual si existe. Solo lee archivos; no llama a ningun modelo.

Uso (Colab):
    from sepia.review import ReviewApp
    ReviewApp(cfg, ["photo_traits_v8", "photo_traits_v9"]).show()
    ReviewApp(cfg, [...], only="versions_differ").show()   # solo fotos donde cambia el lado
    ReviewApp(cfg, [...], only="model_wrong").show()       # solo errores de lado vs anotacion

Cuidado: mirar los valores del modelo antes de anotar una foto rompe la anotacion a ciegas.
Usar en fotos ya anotadas (only="annotated" / "model_wrong") o que no se van a anotar.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .config import Config
from .io import paths, read_jsonl
from .photo_traits import load_photo_traits

# Filas de la tabla (en este orden)
FIELDS = ["n_cuttlebones_visible", "side", "side_rule", "side_basis", "side_signs", "side_evidence",
          "erosion", "photo_quality", "cover", "focus", "quality_evidence",
          "bitemarks", "bitemarks_rule", "n_candidates", "candidates",
          "notes"]
FILTERS = ["all", "annotated", "versions_differ", "model_wrong",
           "versions_agree_human_agrees", "versions_agree_human_differs"]


def review_table(cfg: Config, versions: list[str]) -> pd.DataFrame:
    """
    Una fila por foto: datos de la foto, valores de cada version (<campo>__<version>)
    y anotacion manual (<campo>__human; lado, erosion y calidad solo con 1 individuo).
    """
    from .annotate import load_events
    p = paths(cfg)
    t = read_jsonl(p["tables"] / "photos.jsonl").sort_values(["observation_id", "position"])
    t = t[["observation_id", "photo_id", "position", "file_name"]].copy()
    t["photo_id"] = t["photo_id"].astype("int64")
    from .bitemarks import load_bitemarks
    for v in versions:
        # versiones de picoteo (bitemarks_vN) o de rasgos por foto (photo_traits_vN)
        m = load_bitemarks(cfg, v) if v.startswith("bitemarks_") else load_photo_traits(cfg, v)
        if m.empty:
            continue
        cols = [c for c in FIELDS if c in m]
        m = m[["photo_id"] + cols].rename(columns={c: f"{c}__{v}" for c in cols})
        t = t.merge(m.astype({"photo_id": "int64"}), on="photo_id", how="left")

    ev = load_events(cfg)
    info = ev["photo_info"][["photo_id", "n_in_photo"]].dropna(subset=["photo_id"])
    info = info.astype({"photo_id": "int64"}).rename(
        columns={"n_in_photo": "n_cuttlebones_visible__human"})
    t = t.merge(info, on="photo_id", how="left")
    ind = ev["individual"]
    if len(ind):
        one = ind.groupby("photo_id").filter(lambda g: len(g) == 1)
        keep = [c for c in ("side", "erosion", "photo_quality", "bitemarks", "notes") if c in one]
        one = one[["photo_id"] + keep].astype({"photo_id": "int64"})
        t = t.merge(one.rename(columns={c: f"{c}__human" for c in keep}), on="photo_id", how="left")
    if "side__human" in t:
        t["side_rule__human"] = t["side__human"]   # el lado por regla se compara con el mismo lado
    if "bitemarks__human" in t:
        t["bitemarks_rule__human"] = t["bitemarks__human"]
    return t.reset_index(drop=True)


def _norm(v) -> str:
    """Valor como texto para mostrar y comparar (erosion humana viene como numero)."""
    if isinstance(v, (list, tuple)):              # side_signs (v10-), candidates (bitemarks)
        sep = "; " if any(isinstance(x, dict) for x in v) else ", "
        return sep.join(_norm(x) for x in v)
    if isinstance(v, dict):                       # una candidata de picoteo
        flags = [k[3:] for k, x in v.items() if k.startswith("is_") and x]
        return (f"{v.get('shape')}/{v.get('edges')}/{v.get('location')}"
                + (f" [{', '.join(flags)}]" if flags else "") + f": {v.get('description', '')}")
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def filter_rows(t: pd.DataFrame, versions: list[str], only: str) -> pd.DataFrame:
    """Aplica el filtro: all, annotated, versions_differ (lado), model_wrong (lado vs persona)."""
    if only == "all":
        return t
    annotated = t["n_cuttlebones_visible__human"].notna() if "n_cuttlebones_visible__human" in t \
        else pd.Series(False, index=t.index)
    if only == "annotated":
        return t[annotated]
    sides = [f"side__{v}" for v in versions if f"side__{v}" in t]
    if only == "versions_differ":
        s = t[sides].map(_norm).replace("", pd.NA)      # sin resultado en una version: no cuenta
        return t[s.nunique(axis=1) > 1]
    if only in ("versions_agree_human_agrees", "versions_agree_human_differs"):
        # todas las versiones deciden el mismo lado (dorsal/ventral) y la persona anoto
        # dorsal/ventral; luego: la persona coincide o no
        if "side__human" not in t or not sides:
            return t.iloc[0:0]
        s = t[sides].map(_norm)
        decided = s.isin(["dorsal", "ventral"]).all(axis=1)
        same = s.nunique(axis=1) == 1
        h = t["side__human"].map(_norm)
        human_ok = h.isin(["dorsal", "ventral"])
        hit = s.iloc[:, 0] == h
        keep = decided & same & human_ok & (hit if only.endswith("agrees") else ~hit)
        return t[keep]
    if only == "model_wrong":
        if "side__human" not in t:
            return t.iloc[0:0]
        h = t["side__human"].map(_norm)
        wrong = pd.Series(False, index=t.index)
        for c in sides:
            m = t[c].map(_norm)
            wrong |= (h != "") & (m != "") & (m != "not_applicable") & (m != h)
        return t[wrong]
    raise ValueError(f"filtro desconocido: {only} (usar {FILTERS})")


class ReviewApp:
    """Foto + tabla de valores por version y anotacion manual. Prev / Next / Go to ID."""

    def __init__(self, cfg: Config, versions: list[str], only: str = "annotated",
                 image_px: int = 1024, image_width: str = "520px", photo_ids=None):
        """photo_ids: mostrar solo estas fotos (p. ej. una lista armada a mano)."""
        import ipywidgets as w
        self.w, self.cfg, self.versions = w, cfg, versions
        self.img_dir = paths(cfg)["base"] / "photos" / f"{image_px}px"
        self.full = review_table(cfg, versions)
        if photo_ids is not None:
            self.full = self.full[self.full["photo_id"].isin({int(i) for i in photo_ids})]
        self.header = w.HTML()
        self.image = w.Image(format="jpeg", layout=w.Layout(max_width=image_width,
                                                            max_height="600px",
                                                            object_fit="contain"))
        self.table = w.HTML()
        self.filter = w.Dropdown(options=FILTERS, value=only, description="Show")
        self.filter.observe(lambda _: self._apply_filter(), names="value")
        prev_b, next_b = w.Button(description="◀ Prev"), w.Button(description="Next ▶")
        prev_b.on_click(lambda _: self._go(self.i - 1))
        next_b.on_click(lambda _: self._go(self.i + 1))
        self.jump = w.Text(placeholder="iNaturalist ID", layout=w.Layout(width="160px"))
        go_b = w.Button(description="Go to ID")
        go_b.on_click(lambda _: self._jump())
        self.controls = w.HBox([self.filter, prev_b, next_b, self.jump, go_b])
        self.box = w.VBox([self.controls, self.header,
                           w.HBox([self.image, self.table],
                                  layout=w.Layout(align_items="flex-start"))])
        self._apply_filter()

    def _apply_filter(self):
        self.rows = filter_rows(self.full, self.versions, self.filter.value).reset_index(drop=True)
        self._go(0)

    def _go(self, i):
        if self.rows.empty:
            self.header.value = f"<b>No photos for filter '{self.filter.value}'</b>"
            self.image.value, self.table.value = b"", ""
            return
        self.i = max(0, min(i, len(self.rows) - 1))
        self._render()

    def _jump(self):
        oid = int(self.jump.value.strip())
        idx = self.rows.index[self.rows["observation_id"] == oid]
        if not len(idx):
            self.header.value = f"<b style='color:#b00'>ID {oid} not in this filter</b>"
            return
        self._go(int(idx[0]))

    def _render(self):
        r = self.rows.loc[self.i]
        oid = int(r["observation_id"])
        url = f"https://www.inaturalist.org/observations/{oid}"
        self.header.value = (f"<h3 style='margin:4px 0'>{self.i + 1}/{len(self.rows)} · "
                             f"<a href='{url}' target='_blank'>{oid}</a> · photo "
                             f"{int(r['position'])} (id {int(r['photo_id'])})</h3>")
        f = self.img_dir / (Path(r["file_name"]).stem + ".jpg")
        self.image.value = f.read_bytes() if f.exists() else b""
        cols = [v for v in self.versions] + ["human"]
        head = "".join(f"<th style='padding:4px 8px'>{html.escape(c.replace('photo_traits_', ''))}</th>"
                       for c in cols)
        body = ""
        # filas sin ningun valor (p. ej. picoteo cuando solo se ven versiones de lado): no se muestran
        for field in [f for f in FIELDS if any(f"{f}__{c}" in self.rows for c in cols)]:
            vals = [_norm(r.get(f"{field}__{c}")) for c in cols]
            if field in ("bitemarks", "bitemarks_rule"):
                # para comparar: (yes) del anotador = probable del modelo
                vals = ["probable" if v == "(yes)" else v for v in vals]
            model_vals = {v for v in vals[:-1] if v}
            human = vals[-1]
            cells = ""
            for c, v in zip(cols, vals):
                style = "padding:4px 8px"
                if field in ("side", "side_rule", "erosion", "photo_quality", "bitemarks",
                             "bitemarks_rule", "n_cuttlebones_visible") and c != "human":
                    if human and v and v not in ("not_applicable", "not_assessable") and v != human:
                        style += ";background:#f8d7da;color:#000"     # distinto de la persona
                    elif len(model_vals) > 1:
                        style += ";background:#fff3cd;color:#000"     # versiones no coinciden
                cells += f"<td style='{style}'>{html.escape(v)}</td>"
            body += f"<tr><th style='text-align:right;padding:4px 8px'>{field}</th>{cells}</tr>"
        legend = ("<small>red = differs from human · yellow = versions disagree · "
                  "human side/erosion/quality only for photos with 1 individual</small>")
        self.table.value = (f"<table style='border-collapse:collapse'><tr><th></th>{head}</tr>"
                            f"{body}</table>{legend}")

    def show(self):
        """Muestra la interfaz en la celda del notebook."""
        from IPython.display import display
        display(self.box)
