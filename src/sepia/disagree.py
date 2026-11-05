"""
Revision persona vs modelos: fotos donde la anotacion manual no coincide con las versiones
de prompt de photo_traits. El objetivo es encontrar errores de anotacion (o confirmar errores
del modelo) y dejar constancia de cada decision.

Pasos (Colab):
    from sepia.disagree import rank_versions, DisagreementApp, verdict_summary
    table, top = rank_versions(cfg)                  # 1. metricas de todas las versiones, top 5
    summary, per_class, mats = classification_metrics(cfg, top)   # matrices, precision, recall, F1
    plot_confusion(mats, "side")
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
from .io import paths, read_jsonl, append_jsonl, save_metrics, ids_sha
from .photo_traits import load_photo_traits, compare_photo_traits
from .review import review_table, _norm

# Campos que se comparan con la anotacion manual (lado, erosion y calidad: solo fotos con
# 1 individuo anotado, como en review_table y evaluate_photo_traits)
COMPARE_FIELDS = ["n_cuttlebones_visible", "side", "erosion", "photo_quality"]
# Campos que solo se muestran (ayudan a entender por que decidio el modelo)
INFO_FIELDS = ["side_basis", "side_signs", "side_evidence", "cover", "focus",
               "quality_evidence", "notes"]
# Valores del modelo que no cuentan como desacuerdo: no aplica o se abstuvo
NOT_COMPARED = {"", "not_applicable", "not_assessable", "unclear"}
# Metricas que entran al puntaje para elegir las mejores versiones (promedio simple)
SCORE_METRICS = ["count_agreement", "side_agreement", "erosion_agreement",
                 "photo_quality_agreement"]
PHOTO_CHECKED = "_photo"      # linea de verdicts que marca la foto como revisada
ORDERS = ["record", "most_disagreement"]
# Filtro extra: fotos donde 'Individuals in photo' no coincide con los individuos anotados
COUNT_MISMATCH = "my_count_mismatch"
# Vistas (para mostrar a Martin). En estas vistas 'unclear' SIEMPRE cuenta como diferencia.
VIEWS = ["all", "any_model_differs", "all_models_agree_differ", "ask_martin"]
# Valores que no se pueden comparar (no hay valor o no aplica)
NO_VALUE = {"", "not_applicable"}
# Notas que mencionan a Martin (sin mayusculas ni tildes): martin, martín, marthin
import re as _re
import unicodedata as _ud
from collections import Counter as _Counter
_MARTIN = _re.compile(r"marth?in")


def mentions_martin(text) -> bool:
    t = _ud.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    return bool(_MARTIN.search(t))


def ask_martin_photos(cfg: Config) -> list[int]:
    """photo_id de las fotos donde alguna nota de un individuo menciona a Martin."""
    from .annotate import load_events
    ind = load_events(cfg)["individual"]
    if not len(ind) or "notes" not in ind:
        return []
    return sorted(set(ind.loc[ind["notes"].map(mentions_martin), "photo_id"].astype(int)))


def draw_boxes(image: bytes, boxes: list[tuple[str, dict]]) -> bytes:
    """Dibuja sobre la foto el recuadro (crop_box relativo 0-1) y la letra de cada individuo."""
    if not image or not boxes:
        return image
    import io
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(io.BytesIO(image)).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, W // 250)
    try:
        font = ImageFont.load_default(size=max(14, W // 30))
    except TypeError:                                   # Pillow antiguo
        font = ImageFont.load_default()
    for letter, b in boxes:
        x0, y0, x1, y1 = b["x0"] * W, b["y0"] * H, b["x1"] * W, b["y1"] * H
        d.rectangle([x0, y0, x1, y1], outline=(255, 140, 0), width=lw)
        tx, ty = x0 + lw, max(0, y0 - font.size - 2 * lw) if hasattr(font, "size") else y0
        d.rectangle(d.textbbox((tx, ty), str(letter), font=font), fill=(255, 140, 0))
        d.text((tx, ty), str(letter), fill=(0, 0, 0), font=font)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=90)
    return out.getvalue()


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

# Version "virtual": la moda de varios prompts (columnas <campo>__mode)
MODE = "mode"


def mode_value(values, numeric: bool = False) -> str:
    """
    Moda de los valores de varios prompts para un atributo (votacion por mayoria).
    - No cuentan los vacios ni 'not_applicable' (el prompt no describio ese atributo).
    - Gana el valor con mas votos; si hay empate (p. ej. los 3 distintos): 'unclear'
      (abstencion). En el conteo (numeric=True), un empate deja la foto sin valor: se salta.
    """
    vals = [v for v in values if v not in NO_VALUE]
    if not vals:
        return ""
    counts = _Counter(vals)
    top = max(counts.values())
    winners = [v for v, c in counts.items() if c == top]
    if len(winners) == 1:
        return winners[0]
    return "" if numeric else "unclear"


def add_mode(t: pd.DataFrame, mode_of: list[str]) -> pd.DataFrame:
    """Agrega las columnas <campo>__mode con la moda de las versiones mode_of."""
    for f in COMPARE_FIELDS:
        cols = [f"{f}__{v}" for v in mode_of if f"{f}__{v}" in t]
        if not cols:
            continue
        num = f == "n_cuttlebones_visible"
        t[f"{f}__{MODE}"] = [mode_value([_norm(x) for x in row], numeric=num)
                             for row in t[cols].itertuples(index=False)]
    return t


def disagreement_table(cfg: Config, versions: list[str],
                       mode_of: list[str] | None = None) -> pd.DataFrame:
    """
    Una fila por foto anotada: valores de cada version y de la persona (review_table) y,
    por campo comparado:
      n_models__<campo>    versiones con un valor comparable (no vacio, no unclear/not_applicable)
      n_disagree__<campo>  de esas, cuantas difieren de la persona
    n_disagree_max = maximo entre campos (para ordenar).
    n_individuals__human = individuos anotados en la foto; count_mismatch = ese numero no
    coincide con 'Individuals in photo' (incoherencia de la anotacion, a revisar).
    Con 'unclear' contando como diferencia (vistas para Martin):
      n_differ__<campo>       versiones con valor que difieren de la persona
      all_agree_differ__<campo>  todas las versiones tienen valor, el mismo, y no es el de la persona
    notes_all__human = notas de todos los individuos de la foto; ask_martin = alguna menciona a Martin.
    """
    from .annotate import load_events
    real = [v for v in versions if v != MODE]
    t = review_table(cfg, list(dict.fromkeys(real + list(mode_of or []))))   # sin repetir
    if mode_of:
        t = add_mode(t, mode_of)
    t = t[t["n_cuttlebones_visible__human"].notna()].copy()    # solo fotos anotadas
    ind = load_events(cfg)["individual"]
    n_ind = ind.groupby("photo_id").size() if len(ind) else pd.Series(dtype="int64")
    t["n_individuals__human"] = t["photo_id"].map(n_ind).fillna(0).astype(int)
    t["count_mismatch"] = t["n_individuals__human"] != t["n_cuttlebones_visible__human"]
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
        # 'unclear' cuenta: basta con que las dos partes tengan un valor
        vals = [t[f"{f}__{v}"].map(_norm) if f"{f}__{v}" in t else pd.Series("", index=t.index)
                for v in versions]
        has_h = h != ""
        t[f"n_differ__{f}"] = sum(((has_h & ~m.isin(NO_VALUE) & (m != h)).astype(int)
                                   for m in vals), pd.Series(0, index=t.index))
        if vals:
            first = vals[0]
            same = pd.Series(True, index=t.index)
            for m in vals:
                same &= ~m.isin(NO_VALUE) & (m == first)
            t[f"all_agree_differ__{f}"] = same & has_h & (first != h)
        else:
            t[f"all_agree_differ__{f}"] = False
    t["n_disagree_max"] = t[[f"n_disagree__{f}" for f in COMPARE_FIELDS]].max(axis=1)
    # notas de todos los individuos de la foto (no solo fotos con 1 individuo)
    if len(ind) and "notes" in ind:
        notes = (ind.dropna(subset=["notes"]).loc[lambda d: d["notes"].astype(str).str.strip() != ""]
                 .sort_values("individual").groupby("photo_id")
                 .apply(lambda g: " | ".join(f"{a}: {n}" for a, n in zip(g["individual"], g["notes"]))))
        t["notes_all__human"] = t["photo_id"].map(notes).fillna("")
    else:
        t["notes_all__human"] = ""
    t["ask_martin"] = t["notes_all__human"].map(mentions_martin)
    return t.sort_values(["observation_id", "position"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2b. Metricas clasicas: matriz de confusion, exactitud, precision, recall, F1
# --------------------------------------------------------------------------

# Clases de cada campo (lo que anota la persona). Otros valores del modelo, como 'unclear',
# aparecen como columna extra en la matriz y cuentan como abstencion.
CLASSES = {"side": ["dorsal", "ventral"], "erosion": ["1", "2", "3"],
           "photo_quality": ["high", "medium", "low"]}
# Valores de la persona que se muestran como fila extra en la matriz, pero NO entran en las
# metricas (la verdad de esas fotos es dudosa): side 'unclear'.
HUMAN_EXTRA_ROWS = {"side": ["unclear"]}


def classification_metrics(cfg: Config, versions: list[str],
                           fields=("side", "erosion", "photo_quality"),
                           common: bool = True, save: bool = True, quiet: bool = False,
                           mode_of: list[str] | None = None):
    """
    Por version y campo, contra la anotacion manual (fotos con 1 individuo anotado):
    - n: fotos comparadas. La persona debe haber decidido una clase.
    - side 'unclear' de la persona: fila extra en la matriz, fuera de las metricas.
      human_unclear_n = cuantas hay; human_unclear_model_unclear = en cuantas el modelo
      tambien dijo 'unclear'.
    - coverage: fraccion de esas fotos donde el modelo decidio una clase (no 'unclear').
    - accuracy: aciertos / n (una abstencion cuenta como error).
    - accuracy_when_decided: aciertos / fotos donde el modelo decidio.
    - por clase: precision = aciertos / veces que el modelo dijo esa clase;
      recall = aciertos / veces que la persona anoto esa clase (abstencion = no acierto);
      F1 = media armonica de ambas; support = veces que la persona anoto esa clase.
    - macro_f1 y balanced_accuracy (= promedio de recall): no dependen de cuantas fotos
      hay de cada clase.
    common=True: todas las versiones se evaluan sobre las mismas fotos (por campo).
    Las versiones sin ese campo (p. ej. v1 sin calidad) se saltan para ese campo.
    Devuelve (resumen, tabla por clase, {(version, campo): matriz}).
    save=True agrega una linea por version a outputs/metrics/photo_traits_classification_metrics.jsonl.
    """
    say = (lambda *a: None) if quiet else print
    if mode_of and MODE not in versions:
        versions = list(versions) + [MODE]       # mode_of: evaluar tambien la moda
    t = disagreement_table(cfg, versions, mode_of)
    summary, per_class, mats = [], [], {}
    for f in fields:
        classes = CLASSES[f]
        h = t[f"{f}__human"].map(_norm) if f"{f}__human" in t else pd.Series("", index=t.index)
        vs = [v for v in versions if f"{f}__{v}" in t]
        skipped = [v for v in versions if v not in vs]
        if skipped:
            say(f"{f}: sin resultados en {', '.join(skipped)} (se saltan)")
        rows_extra = HUMAN_EXTRA_ROWS.get(f, [])
        base = h.isin(classes + rows_extra)                  # filas de la matriz
        if common:
            for v in vs:
                m = t[f"{f}__{v}"].map(_norm)
                base &= (m != "") & (m != "not_applicable")
        for v in vs:
            m = t[f"{f}__{v}"].map(_norm)
            sub_all = base & (m != "") & (m != "not_applicable")
            extra = sorted(set(m[sub_all]) - set(classes))    # p. ej. 'unclear' del modelo
            cm_all = (pd.crosstab(h[sub_all], m[sub_all])
                      .reindex(index=classes + rows_extra, columns=classes + extra, fill_value=0)
                      .rename_axis(index="human", columns="model"))
            mats[(v, f)] = cm_all
            # metricas: solo fotos donde la persona decidio una clase
            sub = sub_all & h.isin(classes)
            hh, mm = h[sub], m[sub]
            n = int(len(hh))
            cm = cm_all.loc[classes]
            ok = hh == mm
            decided = mm.isin(classes)
            rec = {"prompt_version": v, "field": f, "n": n,
                   "coverage": float(decided.mean()) if n else None,
                   "accuracy": float(ok.mean()) if n else None,
                   "accuracy_when_decided": float(ok[decided].mean()) if decided.any() else None}
            if rows_extra:
                hu = sub_all & (h == "unclear")
                rec["human_unclear_n"] = int(hu.sum())
                rec["human_unclear_model_unclear"] = int((hu & (m == "unclear")).sum())
            f1s, recalls = [], []
            for c in classes:
                tp = int(cm.loc[c, c])
                pred, sup = int(cm[c].sum()), int(cm.loc[c].sum())
                prec = tp / pred if pred else None
                recl = tp / sup if sup else None
                f1 = (2 * prec * recl / (prec + recl)) if prec and recl else (0.0 if sup else None)
                per_class.append({"prompt_version": v, "field": f, "class": c, "precision": prec,
                                  "recall": recl, "f1": f1, "support": sup, "predicted": pred})
                if sup:
                    f1s.append(f1)
                    recalls.append(recl)
            rec["macro_f1"] = sum(f1s) / len(f1s) if f1s else None
            rec["balanced_accuracy"] = sum(recalls) / len(recalls) if recalls else None
            summary.append(rec)
            if save and n:
                save_metrics(cfg, "photo_traits_classification", {
                    "prompt_version": v if v != MODE else f"mode({','.join(mode_of)})",
                    "field": f,
                    "eval_set": ("common:" if common else "own:") + ",".join(vs),
                    "eval_set_sha": ids_sha(t.loc[sub, "photo_id"].astype(int)),
                    "metrics": {**{k: rec[k] for k in rec if k not in ("prompt_version", "field")},
                                "confusion": {r: {c: int(cm_all.loc[r, c]) for c in cm_all.columns}
                                              for r in cm_all.index},
                                "per_class": [x for x in per_class
                                              if x["prompt_version"] == v and x["field"] == f]}})
    summary = pd.DataFrame(summary)
    per_class = pd.DataFrame(per_class)
    if not quiet and len(summary):
        for f in fields:
            s = summary[summary["field"] == f].set_index("prompt_version").drop(columns="field")
            if s.empty:
                continue
            print(f"\n=== {f} ===")
            print(s.round(2).to_string())
            pc = per_class[per_class["field"] == f].pivot(index="prompt_version", columns="class",
                                                           values=["precision", "recall", "f1"])
            pc = pc.reindex(index=s.index).reindex(columns=CLASSES[f], level=1)
            print(pc.round(2).to_string())
            for v in s.index:
                print(f"\n{v} · confusion (rows = human, columns = model)")
                print(mats[(v, f)].to_string())
    return summary, per_class, mats


def count_agreement(cfg: Config, versions: list[str], quiet: bool = False,
                    mode_of: list[str] | None = None) -> pd.DataFrame:
    """
    Huesos visibles por foto: modelo vs 'Individuals in photo' de la persona, sobre las fotos
    anotadas con resultado en todas las versiones. Devuelve una fila por version
    (n, acuerdo, modelo cuenta de mas, modelo cuenta de menos) e imprime la tabla cruzada.
    """
    if mode_of and MODE not in versions:
        versions = list(versions) + [MODE]
    t = disagreement_table(cfg, versions, mode_of)
    h = pd.to_numeric(t["n_cuttlebones_visible__human"], errors="coerce")
    base = h.notna()
    cols = [f"n_cuttlebones_visible__{v}" for v in versions if f"n_cuttlebones_visible__{v}" in t]
    for c in cols:
        if c != f"n_cuttlebones_visible__{MODE}":   # la moda sin valor (empate) no saca fotos
            base &= pd.to_numeric(t[c], errors="coerce").notna()
    rows = []
    for c in cols:
        v = c.split("__", 1)[1]
        sub = base & pd.to_numeric(t[c], errors="coerce").notna()
        m = pd.to_numeric(t.loc[sub, c], errors="coerce")
        hh = h[sub]
        rows.append({"prompt_version": v, "n": int(len(hh)),
                     "agreement": float((m == hh).mean()) if len(hh) else None,
                     "model_more": int((m > hh).sum()), "model_fewer": int((m < hh).sum())})
        if not quiet:
            cap = lambda x: x.clip(upper=5).astype(int).astype(str).replace("5", "5+")
            print(f"\n{v} · bones visible (rows = human, columns = model)")
            print(pd.crosstab(cap(hh).rename("human"), cap(m).rename("model")).to_string())
    out = pd.DataFrame(rows).set_index("prompt_version")
    if not quiet:
        print("\n" + out.round(2).to_string())
    return out


def plot_confusion(mats: dict, field: str, versions: list[str] | None = None):
    """
    Matrices de confusion de un campo, una por version, lado a lado.
    Numero = fotos; color = fraccion de la fila (recall de esa clase anotada).
    """
    import matplotlib.pyplot as plt
    vs = versions or [v for (v, f) in mats if f == field]
    if not vs:
        print(f"Sin matrices para {field}")
        return None
    fig, axes = plt.subplots(1, len(vs), figsize=(3.2 * len(vs), 3.2), squeeze=False)
    for ax, v in zip(axes[0], vs):
        cm = mats[(v, field)]
        frac = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        ax.imshow(frac.values, cmap="Blues", vmin=0, vmax=1)     # un solo tono: claro -> oscuro
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm.iat[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if frac.iat[i, j] > 0.55 else "#222")
        ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=45, ha="right")
        ax.set_yticks(range(cm.shape[0]), cm.index)
        ax.set_xlabel("model")
        ax.set_title(v.replace("photo_traits_", ""), fontsize=11)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0][0].set_ylabel("human")
    note = (" · human 'unclear' row: not in the metrics" if HUMAN_EXTRA_ROWS.get(field) else "")
    fig.suptitle(f"{field}: rows = human, columns = model (color = row fraction){note}", fontsize=11)
    fig.tight_layout()
    return fig


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
                 order: str = "record", view: str = "all", read_only: bool = False,
                 image_width: str = "560px", thumb_px: str = "130px",
                 mode_of: list[str] | None = None):
        """
        view: all / any_model_differs / all_models_agree_differ / ask_martin (en estas vistas
        'unclear' cuenta como diferencia). read_only=True: sin veredictos ni botones de guardar.
        cfg: una corrida o una lista de corridas (p. ej. [cfg_a1, cfg_a2]). Con varias, se juntan
        solo en memoria para mirarlas: cada corrida sigue en su carpeta y los veredictos se
        guardan en la corrida de cada foto.
        """
        import ipywidgets as w
        cfgs = list(cfg) if isinstance(cfg, (list, tuple)) else [cfg]
        if mode_of and MODE not in versions:
            versions = list(versions) + [MODE]       # columna extra con la moda
        self.w, self.cfg, self.versions, self.annotator = w, cfgs[0], versions, annotator
        self.read_only = read_only
        self.runs = {Path(c.base_dir).name: c for c in cfgs}
        self.dirs = {}
        for name, c in self.runs.items():
            base = paths(c)["base"] / "photos"
            big = [base / "1024px", base / "original"]
            self.dirs[name] = (big, [base / "512px"] + big)
        self.thumb_px = thumb_px
        self.full = pd.concat([disagreement_table(c, versions, mode_of).assign(run=n)
                               for n, c in self.runs.items()], ignore_index=True)
        # todas las fotos (tambien sin anotar) para la tira de miniaturas
        self.all_photos = pd.concat([review_table(c, [v for v in versions if v != MODE])
                                     .assign(run=n)
                                     for n, c in self.runs.items()], ignore_index=True)
        self.individuals = self._load_individuals()
        # fotos cuya anotacion vino prellenada desde la asistida (no es a ciegas)
        ind = self.individuals
        self.assisted = (set(ind.loc[ind["prefilled_from"] == "assisted", "photo_id"].astype(int))
                         if "prefilled_from" in ind else set())
        self._cache = {}

        n = len(versions)
        self.field = w.Dropdown(options=["any"] + COMPARE_FIELDS + [COUNT_MISMATCH], value=field,
                                description="Field", layout=w.Layout(width="260px"))
        self.order = w.Dropdown(options=ORDERS, value=order, description="Order",
                                layout=w.Layout(width="260px"))
        self.min_dis = w.IntSlider(value=min(min_disagree, n), min=0, max=n,
                                   description=f"Models ≠ me (of {n})",
                                   style={"description_width": "initial"})
        self.hide = w.Checkbox(value=hide_reviewed, description="Hide reviewed", indent=False)
        self.view = w.Dropdown(options=VIEWS, value=view, description="View",
                               layout=w.Layout(width="320px"))
        self.run_sel = w.Dropdown(options=["all runs"] + list(self.runs), value="all runs",
                                  description="Run", layout=w.Layout(width="240px"))
        for c in (self.view, self.run_sel, self.field, self.order, self.min_dis, self.hide):
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
        if read_only:
            right = w.VBox([self.table, self.record])
            filters = w.HBox([self.view, self.run_sel, self.field, self.order])
        else:
            right = w.VBox([self.table, self.record, self.verdict_box,
                            w.HBox([save_b, save_next_b]), self.msg])
            filters = w.HBox([self.view, self.run_sel, self.field, self.order, self.min_dis,
                              self.hide])
        self.box = w.VBox([filters,
                           w.HBox([prev_b, next_b, self.jump, go_b]),
                           self.header,
                           w.HBox([left, right], layout=w.Layout(align_items="flex-start"))])
        self._apply_filter()

    # ---- datos ----
    def _load_individuals(self) -> pd.DataFrame:
        from .annotate import load_events
        parts = [load_events(c)["individual"] for c in self.runs.values()]
        parts = [p for p in parts if len(p)]
        return (pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame(columns=["photo_id", "individual"]))

    def _verdicts(self) -> pd.DataFrame:
        parts = [load_verdicts(c, self.annotator) for c in self.runs.values()]
        parts = [p for p in parts if len(p)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _apply_filter(self):
        f = self.field.value
        cols = COMPARE_FIELDS if f in ("any", COUNT_MISMATCH) else [f]
        rows = self.full
        if self.run_sel.value != "all runs":
            rows = rows[rows["run"] == self.run_sel.value]
        if f == COUNT_MISMATCH:
            rows = rows[rows["count_mismatch"]]
        view = self.view.value
        if view == "any_model_differs":
            keep = pd.Series(False, index=rows.index)
            for c in cols:
                keep |= rows[f"n_differ__{c}"] > 0
            rows = rows[keep]
        elif view == "all_models_agree_differ":
            keep = pd.Series(False, index=rows.index)
            for c in cols:
                keep |= rows[f"all_agree_differ__{c}"]
            rows = rows[keep]
        elif view == "ask_martin":
            rows = rows[rows["ask_martin"]]
        if self.min_dis.value > 0:          # 0 = todas las fotos anotadas
            keep = pd.Series(False, index=rows.index)
            for c in cols:
                keep |= rows[f"n_disagree__{c}"] >= self.min_dis.value
            rows = rows[keep]
        if self.order.value == "most_disagreement":
            key = f"n_disagree__{f}" if f in COMPARE_FIELDS else "n_disagree_max"
            rows = rows.sort_values([key, "observation_id", "position"],
                                    ascending=[False, True, True])
        if self.hide.value and not self.read_only:
            v = self._verdicts()
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
        if int(r["photo_id"]) in self.assisted:
            badge = ("background:#fde2c8", "annotation prefilled from the assisted review "
                     "(not blind)")
        else:
            badge = ("background:#d4edda", "blind annotation")
        self.header.value += (f"<div style='{badge[0]};color:#000;display:inline-block;"
                              f"padding:2px 8px;margin:2px 0'>run <b>{html.escape(r['run'])}</b>"
                              f" · {badge[1]}</div>")
        if r["count_mismatch"]:
            self.header.value += (
                "<div style='background:#fff3cd;color:#000;padding:4px 8px;margin:4px 0'>"
                f"⚠ My annotation: 'Individuals in photo' = "
                f"{_norm(r['n_cuttlebones_visible__human'])}, but "
                f"{int(r['n_individuals__human'])} individual(s) annotated in this photo</div>")
        if r.get("notes_all__human"):
            bg = "#d1ecf1" if r["ask_martin"] else "#eee"
            self.header.value += (
                f"<div style='background:{bg};color:#000;padding:4px 8px;margin:4px 0'>"
                f"<b>My notes:</b> {html.escape(r['notes_all__human'])}</div>")
        self._show_big(r)
        self._render_strip(r)
        self._render_table(r)
        self._render_record(r)
        if not self.read_only:
            self._render_verdicts(r)

    def _show_big(self, photo_row, reviewed_row=None):
        img = self._read(photo_row["file_name"], self.dirs[photo_row["run"]][0])
        ind = self.individuals
        mine = (ind[ind["photo_id"] == int(photo_row["photo_id"])].sort_values("individual")
                if len(ind) else ind)
        boxes = [(r["individual"], r["crop_box"]) for _, r in mine.iterrows()
                 if isinstance(r.get("crop_box"), dict)]
        self.image.value = draw_boxes(img, boxes)
        letters = list(mine["individual"]) if len(mine) else []
        no_box = [l for l in letters if l not in {b[0] for b in boxes}]
        txt = f"Photo {int(photo_row['position'])}"
        if letters:
            txt += " · individuals: " + ", ".join(
                f"<b>{html.escape(str(l))}</b>" + (" (no box)" if l in no_box else "")
                for l in letters)
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
        thumbs = self.dirs[r["run"]][1]
        items = []
        for _, p in rec.iterrows():
            current = int(p["photo_id"]) == int(r["photo_id"])
            img = w.Image(value=self._read(p["file_name"], thumbs), format="jpeg",
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
            several = ""
            if field in ("side", "erosion", "photo_quality", "notes") and not human:
                # con varios individuos no se compara, pero se muestran sus valores
                mine = self.individuals[self.individuals["photo_id"] == int(r["photo_id"])]
                if len(mine) > 1 and field in mine:
                    several = " · ".join(f"{i['individual']}: {_norm(i.get(field))}"
                                         for _, i in mine.sort_values("individual").iterrows())
            cells = ""
            for c, v in zip(cols, vals):
                st = "padding:3px 7px;vertical-align:top"
                if field in INFO_FIELDS:
                    st += ";font-size:85%;max-width:180px"
                elif c != "human" and human and v not in NOT_COMPARED and v != human:
                    st += ";background:#f8d7da;color:#000"           # decidio distinto
                elif c != "human" and human and v in ("unclear", "not_assessable") and v != human:
                    st += ";background:#d6d6d6;color:#000"           # no decidio (cuenta igual)
                elif c == "human":
                    st += ";font-weight:bold"
                if c == "human" and several:
                    st += ";color:#d97706;font-style:italic"
                    v = several
                cells += f"<td style='{st}'>{html.escape(v)}</td>"
            n = ""
            if field in COMPARE_FIELDS:
                unc = int(r[f"n_differ__{field}"]) - int(r[f"n_disagree__{field}"])
                extra = f" +{unc} unclear" if unc > 0 else ""
                n = (f" <small>({int(r[f'n_disagree__{field}'])}/{int(r[f'n_models__{field}'])}"
                     f"{extra})</small>")
            body += f"<tr><th style='text-align:right;padding:3px 7px'>{field}{n}</th>{cells}</tr>"
        legend = ("<small>red = decided something different from me · grey = did not decide "
                  "(unclear), counts as a difference in the views · (x/y +z unclear) = models "
                  "that decided differently / models that decided, + models unclear · orange = "
                  "several individuals in the photo: shown, not compared</small>")
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
        prev = load_verdicts(self.runs[r["run"]], self.annotator)
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
        run_cfg = self.runs[r["run"]]
        prev = load_verdicts(run_cfg, self.annotator)
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
            }, verdict_file(run_cfg, self.annotator))
            n += 1
        # la foto queda marcada como revisada aunque no haya ningun error
        append_jsonl({"observation_id": int(r["observation_id"]), "photo_id": int(r["photo_id"]),
                      "field": PHOTO_CHECKED, "verdict": "checked", "note": None,
                      "versions": self.versions, "reviewer": self.annotator, "saved_at": now},
                     verdict_file(run_cfg, self.annotator))
        self.msg.value = (f"<span style='color:#080'>Photo marked as reviewed · "
                          f"{n} verdict(s) saved</span>")

    def show(self):
        from IPython.display import display
        display(self.box)
