from pathlib import Path
"""
Pruebas sin internet: simulan la API de iNaturalist.
Correr con:  pytest -q
"""
import io
import json

import openpyxl
import pytest
from PIL import Image

import sepia
from sepia import inat, config

LAT, LON = 57.1900283333, -2.072345


def fake_obs(i):
    return {"id": i, "uuid": f"u{i}", "observed_on": "2026-02-22", "quality_grade": "research",
            "captive": False, "license_code": "cc0", "place_guess": "Bridge of Don, UK",
            "geojson": {"type": "Point", "coordinates": [LON, LAT]},
            "positional_accuracy": 10, "obscured": i == 3, "user": {"login": "x"},
            "taxon": {"id": 151429, "name": "Sepia officinalis"},
            "observation_photos": [
                {"position": 1, "photo": {"id": i * 10 + 2, "license_code": "cc0",
                 "url": f"https://inaturalist-open-data.s3.amazonaws.com/photos/{i*10+2}/square.jpg"}},
                {"position": 0, "photo": {"id": i * 10 + 1, "license_code": None,
                 "url": f"https://static.inaturalist.org/photos/{i*10+1}/square.jpeg?123"}}]}


ALL = [fake_obs(i) for i in range(1, 8)]


class R:
    def __init__(self, j=None, c=b""):
        self._j, self.content, self.status_code = j, c, 200
    def json(self):
        return self._j
    def raise_for_status(self):
        pass


@pytest.fixture
def api(monkeypatch):
    buf = io.BytesIO()
    Image.new("RGB", (2048, 1536), (200, 200, 200)).save(buf, "JPEG")
    jpg = buf.getvalue()

    def fake_get(self, url, params=None, timeout=None):
        if url == config.API_URL:
            if params.get("per_page") == 0:
                return R({"total_results": len(ALL), "results": []})
            if "id" in params:
                ids = {int(x) for x in params["id"].split(",")}
                return R({"results": [o for o in ALL if o["id"] in ids]})
            return R({"results": [o for o in ALL if o["id"] > params["id_above"]][:3]})
        assert "/original." in url
        return R(c=jpg)

    monkeypatch.setattr(inat.requests.Session, "get", fake_get)
    monkeypatch.setattr(inat, "API_PAUSE_S", 0)
    monkeypatch.setattr(inat, "PHOTO_PAUSE_S", 0)


@pytest.fixture
def cfg(tmp_path):
    return sepia.Config(base_dir=str(tmp_path), per_page=3)


def test_download_resume_and_tables(api, cfg):
    sepia.download_observations(cfg, max_pages=2)   # se corta a las 2 paginas
    sepia.download_observations(cfg)                # reanuda
    assert len(sepia.load_raw(cfg)) == 7
    obs, pho = sepia.build_tables(cfg)
    assert list(pho[pho.observation_id == 1].photo_id) == [11, 12]   # orden del observador
    assert pho.file_name[0] == "1_11.jpg"
    assert pho.download_url[0].endswith("/original.jpeg?123")
    back, _ = sepia.load_tables(cfg)
    assert back.latitude[0] == LAT and back.longitude[0] == LON       # precision completa


def test_photos_resume_and_resize(api, cfg):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg, limit=5)
    stats = sepia.download_photos(cfg)
    assert stats["skipped"] == 5 and stats["ok"] == 9
    sepia.make_resized(cfg, 512)
    small = next((sepia.paths(cfg)["base"] / "photos" / "512px").glob("*.jpg"))
    assert max(Image.open(small).size) == 512
    rep = sepia.check_download(cfg)
    assert rep["photos_missing"] == 0 and rep["coordinates_obscured"] == 1


def test_download_by_ids(api, cfg):
    assert sepia.download_by_ids(cfg, [1, 2, 99, 5]) == 3


def _excel_rows(path):
    """Filas del Excel: identification + traceability (mismas filas, mismo orden), unidas."""
    wb = openpyxl.load_workbook(path)
    out = []
    for name in ("identification", "traceability"):
        ws = wb[name]
        hdr = [c.value for c in ws[1]]
        rows = [dict(zip(hdr, r)) for r in ws.iter_rows(min_row=2, values_only=True)]
        out = rows if not out else [{**a, **b} for a, b in zip(out, rows)]
    return wb, out


def test_excel(api, cfg):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    ws = openpyxl.load_workbook(sepia.build_excel(cfg, "a.xlsx"))["identification"]
    assert ws.max_row == 1 + 14 and ws.cell(2, 5).value == LAT     # 7 registros x 2 fotos
    assert not any(isinstance(c.value, str) and c.value.startswith("=")
                   for row in ws.iter_rows() for c in row)

    # Registro 1: fotos 11 (dorsal) y 12 (ventral) del individuo a; en la foto 12
    # aparece ademas un segundo hueso b. El lado es atributo de la fila.
    f = sepia.paths(cfg)["annotations"] / "HZ.jsonl"
    sepia.append_jsonl({"observation_id": 1, "photo_id": 11, "individual": "a", "side": "dorsal",
                        "epibionts": "yes", "saved_at": "2026-10-28T10:00"}, f)
    sepia.append_jsonl({"observation_id": 1, "photo_id": 12, "individual": "a", "side": "ventral",
                        "bitemarks": "no", "saved_at": "2026-10-28T10:01"}, f)
    sepia.append_jsonl({"observation_id": 1, "photo_id": 12, "individual": "a", "side": "ventral",
                        "bitemarks": "yes", "saved_at": "2026-10-28T10:05"}, f)   # correccion
    sepia.append_jsonl({"observation_id": 1, "photo_id": 12, "individual": "b", "side": "dorsal",
                        "saved_at": "2026-10-28T10:06"}, f)
    sepia.append_jsonl({"observation_id": 1, "individual": "c", "side": "dorsal",
                        "saved_at": "2026-10-28T10:07"}, f)                       # sin photo_id: se ignora
    wb, rows = _excel_rows(sepia.build_excel(cfg, "b.xlsx"))
    r1 = [r for r in rows if r["ID"] == 1]
    assert len(rows) == 15 and len(r1) == 3
    assert [(r["photo_id"], r["individual (a,b,c,d,e)"]) for r in r1] == [(11, "a"), (12, "a"), (12, "b")]
    ventral = [r for r in r1 if r["side (cuttlebone)"] == "ventral"]
    assert len(ventral) == 1 and ventral[0]["¿bitemarks?"] == "yes"   # vale la ultima
    assert all(r["number individuals (cuttlebone)"] == 2 for r in r1)  # letras distintas, no filas
    assert all(r["latitude"] == LAT for r in r1)                       # coordenadas por join
    run = Path(cfg.base_dir).name
    assert wb.sheetnames == ["legend", "map", "identification", run, "traceability", "excluded"]
    rs = wb[run]
    rhdr = [c.value for c in rs[1]]
    assert rhdr[:4] == ["id", "uuid", "observed_on_string", "observed_on"]
    assert "private_latitude" in rhdr and rhdr[-1] == "excluded_reason"
    recs = [dict(zip(rhdr, r)) for r in rs.iter_rows(min_row=2, values_only=True)]
    assert len(recs) == 7 and recs[0]["latitude"] == LAT            # una fila por registro
    r1 = [r for r in recs if r["id"] == 1][0]
    assert r1["photos annotated"] == "2/2" and r1["number individuals (cuttlebone)"] is None
    assert r1["photo IDs"] == "11, 12"
    ident = [c.value for c in wb["identification"][1]]
    assert ident[:10] == ["n° observation", "ID", "photo ID", "site", "latitude", "longitude",
                          "place", "observed on", "season", "url"]      # Martin + photo ID
    assert "side (cuttlebone)" in ident and "dorsal side (cuttlebone)" not in ident
    assert ident[-3] == "possible genus/specie of balano"
    assert ident[-2:] == ["same photo as (photo ID)", "same bone as other record (ID:letter)"]
    assert all(r["season"] is None and r["site"] == "A" for r in rows)   # season pendiente; sector A
    assert all(r["label_source"] != "gpt" for r in rows)                  # solo anotacion manual


def test_config_yaml(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("base_dir: /x\nbbox: [48, -6, 59.5, 13]\nresize: [512]\n")
    c = sepia.load_config(str(y))
    assert c.bbox == (48, -6, 59.5, 13) and c.resize == (512,)
    assert c.query_params()["swlat"] == 48


def test_ids_file(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("# comentario\n339768773\n\n26739989\n339768773\n")
    assert sepia.ids_from_file(str(f)) == [26739989, 339768773]


def test_pilot_ids_in_repo():
    ids = sepia.ids_from_file("configs/pilot79_ids.txt")
    assert len(ids) == 79


class FakeProvider:
    name = "fake"
    def __init__(self):
        self.calls = []
    def classify(self, images, prompt):
        self.calls.append(len(images))
        res = {"photos": [{"photo_index": i + 1, "content": "cuttlebone", "n_cuttlebones_visible": 1}
                          for i in range(len(images))],
               "record_state": "cuttlebone", "n_cuttlebones": 1, "count_possible": True,
               "same_bone_in_several_photos": len(images) > 1, "confidence": "high", "notes": ""}
        sepia.triage.TriageResult.model_validate(res)   # cumple el esquema
        return res, json.dumps(res), {"input_tokens": 500, "output_tokens": 80}


def test_triage_resume(api, cfg, monkeypatch):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg)
    sepia.make_resized(cfg, 512)
    cfg.triage = {"provider": "fake", "model": "m1", "requests_per_minute": 100000}
    monkeypatch.setattr(sepia.triage.time, "sleep", lambda s: None)
    prov = FakeProvider()
    sepia.run_triage(cfg, provider=prov, limit=3)
    sepia.run_triage(cfg, provider=prov)              # reanuda: solo los 4 que faltan
    assert len(prov.calls) == 7 and set(prov.calls) == {2}   # todas las fotos del registro juntas
    t = sepia.load_triage(cfg, "fake", "m1")
    assert len(t) == 7 and t["n_cuttlebones"].eq(1).all()
    raw = sepia.read_jsonl(sepia.triage.output_path(cfg, "fake", "m1", "triage_v1"))
    assert raw["prompt_sha"].nunique() == 1 and raw["photo_ids"].iloc[0] == [11, 12]


def test_prompt_versioned():
    text, sha = sepia.load_prompt("triage_v1")
    assert "DISTINCT cuttlebones" in text and len(sha) == 12


def test_triage_stops_on_fatal(api, cfg, monkeypatch):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg)
    sepia.make_resized(cfg, 512)
    cfg.triage = {"provider": "fake", "model": "gone", "requests_per_minute": 100000}
    monkeypatch.setattr(sepia.triage.time, "sleep", lambda s: None)

    class Gone:
        calls = 0
        def classify(self, images, prompt):
            Gone.calls += 1
            raise sepia.triage.FatalProviderError("404 NOT_FOUND model gone")

    sepia.run_triage(cfg, provider=Gone())
    assert Gone.calls == 1                       # se detuvo al primer error de configuracion
    assert sepia.triage.done_ids(sepia.read_jsonl(sepia.triage.output_path(cfg, "fake", "gone", "triage_v1"))) == set()


def test_triage_retries_overload(api, cfg, monkeypatch):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg)
    sepia.make_resized(cfg, 512)
    cfg.triage = {"provider": "fake", "model": "busy", "requests_per_minute": 100000}
    monkeypatch.setattr(sepia.triage.time, "sleep", lambda s: None)
    ok = FakeProvider()

    class Busy:
        n = 0
        def classify(self, images, prompt):
            Busy.n += 1
            if Busy.n == 1:   # primer intento: modelo saturado
                raise sepia.triage.RateLimited("503 UNAVAILABLE high demand")
            return ok.classify(images, prompt)

    sepia.run_triage(cfg, provider=Busy(), limit=2)
    t = sepia.load_triage(cfg, "fake", "busy")
    assert len(t) == 2 and Busy.n == 3   # reintento y luego siguio


def test_triage_only_ids(api, cfg, monkeypatch):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg)
    sepia.make_resized(cfg, 512)
    cfg.triage = {"provider": "fake", "model": "ids", "requests_per_minute": 100000}
    monkeypatch.setattr(sepia.triage.time, "sleep", lambda s: None)
    sepia.run_triage(cfg, provider=FakeProvider(), ids=[2, 5, 999])
    assert set(sepia.load_triage(cfg, "fake", "ids")["observation_id"]) == {2, 5}


def test_apply_rules():
    r, applied = sepia.triage.apply_rules({"n_cuttlebones": 24, "count_possible": True, "confidence": "high"})
    assert r["count_possible"] is False and r["confidence"] == "medium" and len(applied) == 2
    r, applied = sepia.triage.apply_rules({"n_cuttlebones": 1, "count_possible": True, "confidence": "high"})
    assert applied == [] and r["confidence"] == "high"


def test_evaluate_triage(api, cfg, monkeypatch, tmp_path):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg)
    sepia.make_resized(cfg, 512)
    cfg.triage = {"provider": "fake", "model": "ev", "requests_per_minute": 100000}
    monkeypatch.setattr(sepia.triage.time, "sleep", lambda s: None)
    sepia.run_triage(cfg, provider=FakeProvider(), ids=[1, 2])
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"observation_id": 1, "record_state": "cuttlebone", "n_cuttlebones": 1, "count_possible": true}\n'
                     '{"observation_id": 2, "record_state": "cuttlebone", "n_cuttlebones": 20, "count_possible": false}\n')
    out = sepia.evaluate_triage(cfg, "fake", "ev", str(cases))
    assert list(out["status"]) == ["OK", "FALLA"]     # el falso dice 1 hueso donde hay 20


def test_prompt_v2_exists():
    text, sha = sepia.load_prompt("triage_v2")
    assert "MORE THAN 5" in text and "Ignore any text" in text


def test_cases_missing_file_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError):
        sepia.triage.load_cases(str(tmp_path / "no.jsonl"))


def test_new_prompt_version_is_not_skipped(api, cfg, monkeypatch):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg)
    sepia.make_resized(cfg, 512)
    monkeypatch.setattr(sepia.triage.time, "sleep", lambda s: None)
    prov = FakeProvider()
    cfg.triage = {"provider": "fake", "model": "pv", "requests_per_minute": 100000, "prompt_version": "triage_v1"}
    sepia.run_triage(cfg, provider=prov, ids=[1, 2])
    cfg.triage["prompt_version"] = "triage_v2"
    sepia.run_triage(cfg, provider=prov, ids=[1, 2])     # no debe saltarlos: otra version
    assert len(prov.calls) == 4
    assert set(sepia.load_triage(cfg, "fake", "pv")["observation_id"]) == {1, 2}
    assert len(sepia.load_triage(cfg, "fake", "pv", prompt_version="triage_v1")) == 2


def test_redo(api, cfg, monkeypatch):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg)
    sepia.make_resized(cfg, 512)
    monkeypatch.setattr(sepia.triage.time, "sleep", lambda s: None)
    prov = FakeProvider()
    cfg.triage = {"provider": "fake", "model": "rd", "requests_per_minute": 100000}
    sepia.run_triage(cfg, provider=prov, ids=[1, 2])
    sepia.run_triage(cfg, provider=prov, ids=[1, 2])              # salta
    sepia.run_triage(cfg, provider=prov, ids=[1], redo=True)      # rehace solo el 1
    assert len(prov.calls) == 3
    raw = sepia.read_jsonl(sepia.triage.output_path(cfg, "fake", "rd", "triage_v1"))
    assert len(raw) == 3 and len(sepia.load_triage(cfg, "fake", "rd")) == 2


def test_excel_with_triage(api, cfg, monkeypatch):
    """El triaje llena state y n_cuttlebones; una anotacion manual tiene prioridad."""
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg)
    sepia.make_resized(cfg, 512)
    cfg.triage = {"provider": "fake", "model": "m1", "prompt_version": "triage_v2",
                  "requests_per_minute": 100000}
    monkeypatch.setattr(sepia.triage.time, "sleep", lambda s: None)
    sepia.run_triage(cfg, provider=FakeProvider())
    f = sepia.paths(cfg)["annotations"] / "HZ.jsonl"
    sepia.append_jsonl({"observation_id": 2, "photo_id": 21, "individual": "a", "side": "dorsal",
                        "state": "egg case", "label_source": "annotator",
                        "saved_at": "2026-10-28T10:00"}, f)
    _, plain = _excel_rows(sepia.build_excel(cfg, "t0.xlsx"))            # por defecto: sin triaje
    assert all(r["state"] is None for r in plain if r["ID"] == 1)
    _, rows = _excel_rows(sepia.build_excel(cfg, "t.xlsx", use_triage=True))
    rows = {(r["ID"], r["photo_id"]): r for r in rows}
    r1, r2 = rows[(1, 11)], rows[(2, 21)]
    assert r1["state"] == "cuttlebone" and r1["number individuals (cuttlebone)"] == 1
    assert r1["label_source"] == "gpt"
    assert "model_version" not in r1 and "model_prob" not in r1      # no van al Excel
    assert r1["reviewed"] is False
    assert r2["state"] == "egg case" and r2["label_source"] == "annotator"   # gana la persona


# --------------------------------------------------------------------------
# Anotacion manual
# --------------------------------------------------------------------------

def _ready(cfg):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg)
    sepia.make_resized(cfg, 1024)


def test_annotation_events(api, cfg):
    """Dorsal y ventral del mismo individuo en fotos distintas conviven; corregir = la ultima."""
    from sepia import annotate as an
    _ready(cfg)
    q = an.work_queue(cfg)
    assert len(q) == 14 and an.first_pending(q, an.load_events(cfg)) == 0
    an.save_individual(cfg, "HZ", 1, 11, "a", {"state": "cuttlebone", "side": "dorsal",
                                               "notes": "same bone as in record 2?"})
    an.save_individual(cfg, "HZ", 1, 12, "a", {"state": "cuttlebone", "side": "dorsal"})
    an.save_individual(cfg, "HZ", 1, 12, "a", {"state": "cuttlebone", "side": "ventral"})  # correccion
    an.save_individual(cfg, "HZ", 1, 12, "b", {"state": "cuttlebone", "side": "dorsal"})
    an.delete_individual(cfg, "HZ", 1, 12, "b")                                           # error
    an.mark_empty_photo(cfg, "HZ", 2, 21)
    an.exclude_record(cfg, "HZ", 3, "captive animal")
    an.exclude_record(cfg, "HZ", 4, "duplicate")
    an.exclude_record(cfg, "HZ", 4, "")                                                   # deshacer
    ev = an.load_events(cfg)
    ind = ev["individual"].set_index(["photo_id", "individual"])["side"].to_dict()
    assert ind == {(11, "a"): "dorsal", (12, "a"): "ventral"}
    assert an.letters_in_record(ev, 1) == ["a"] and an.next_letter(["a", "b"]) == "c"
    assert list(ev["exclude_record"]["observation_id"]) == [3]
    # pendiente: foto 22 (la 21 esta vacia, el registro 1 completo)
    assert q.iloc[an.first_pending(q, ev)]["photo_id"] == 22
    line = sepia.read_jsonl(an.annotation_file(cfg, "HZ")).iloc[0]
    assert line["schema_version"] == 1 and line["annotator"] == "HZ" and line["saved_at"].endswith("+00:00")

    wb, rows = _excel_rows(sepia.build_excel(cfg, "ann.xlsx"))
    r1 = [r for r in rows if r["ID"] == 1]
    assert [(r["photo ID"], r["side (cuttlebone)"]) for r in r1] == [(11, "dorsal"), (12, "ventral")]
    assert r1[0]["notes"] == "same bone as in record 2?"
    assert all(r["number individuals (cuttlebone)"] == 1 and r["reviewed"] is True for r in r1)
    r21 = [r for r in rows if r["photo_id"] == 21][0]
    assert r21["reviewed"] is True and r21["individual (a,b,c,d,e)"] is None
    assert not any(r["ID"] == 3 for r in rows)
    ex = list(wb["excluded"].iter_rows(min_row=2, values_only=True))
    assert len(ex) == 1 and ex[0][1] == 3 and ex[0][-1] == "captive animal"   # ex[0][0] = run


def test_annotation_app_smoke(api, cfg):
    """La interfaz se construye y los botones escriben las lineas esperadas."""
    pytest.importorskip("ipywidgets")
    _ready(cfg)
    app = sepia.AnnotationApp(cfg, annotator="HZ", round_=1)
    assert app.row["photo_id"] == 11 and len(app.image.value) > 0
    assert len(app.strip.children) == 2                           # miniaturas de las 2 fotos
    app.strip.children[1].children[1].click()                     # clic en la miniatura 2
    assert app.row["photo_id"] == 12
    app.strip.children[0].children[1].click()
    assert app.row["photo_id"] == 11
    assert app.inputs["state"].value == "cuttlebone"              # estado por defecto
    assert app.n_photo.value == 1                                 # individuos en la foto: 1
    assert app.inputs["bitemarks"].value == "no" and app.inputs["epibionts"].value == "no"  # sugeridos
    assert app.inputs["side"].layout.display is None              # visible para huesos
    assert app.inputs["barnacles_n"].layout.display == "none"     # oculto sin epibiontes
    app.inputs["side"].value = "dorsal"
    app.inputs["epibionts"].value = "yes"
    app.inputs["barnacles_n"].value = "1-10"
    app.buttons.children[0].click()                               # "Save ▶": guarda y avanza
    assert "Saved" in app.msg.value and app.row["photo_id"] == 12
    assert app.letter.value == "b" and "a" in [o[1] for o in app.letter.options]
    app.letter.value = "a"                                        # mismo hueso, otra foto
    assert app.inputs["state"].value == "cuttlebone" and app.inputs["side"].value is None
    assert app.inputs["epibionts"].value == "no"                  # sugerido (nuevo en esta foto)
    app.inputs["side"].value = "ventral"
    app.inputs["epibionts"].value = "no"
    app._safe(app.on_save)
    app._safe(app.on_next)
    app._safe(app.on_empty)                                       # foto 21 sin individuos
    app.reason.value = "not Sepia"
    app._safe(app.on_exclude)                                     # excluye registro 2
    app.inputs["state"].value = None
    app._safe(app.on_save)                                        # sin estado: error visible
    assert "Error" in app.msg.value
    ev = sepia.load_events(cfg)
    ind = ev["individual"]
    assert len(ind) == 2 and ind["round"].eq(1).all()
    assert ind.set_index("photo_id").loc[11, "barnacles_n"] == "1-10"
    assert "barnacles_n" not in ind or pd_isna(ind.set_index("photo_id").loc[12].get("barnacles_n"))
    assert list(ev["exclude_record"]["observation_id"]) == [2]
    app2 = sepia.AnnotationApp(cfg, annotator="HZ")              # retoma donde quedo
    assert app2.row["observation_id"] == 3


def pd_isna(v):
    import pandas as pd
    return v is None or pd.isna(v)


def test_saved_blank_stays_blank(api, cfg):
    """Un 'no se puede evaluar' (vacio) ya guardado no se convierte en 'no' al volver a abrirlo."""
    pytest.importorskip("ipywidgets")
    from sepia import annotate as an
    _ready(cfg)
    an.save_individual(cfg, "HZ", 1, 11, "a", {"state": "cuttlebone", "bitemarks": None,
                                               "epibionts": "yes"})
    app = sepia.AnnotationApp(cfg, annotator="HZ")
    app._go(0)
    assert app.letter.value == "a"
    assert app.inputs["bitemarks"].value is None and app.inputs["epibionts"].value == "yes"


def test_crop_box_roundtrip(tmp_path):
    """Recuadro relativo: sirve igual en 1024 px y en el original (y respeta EXIF)."""
    from sepia import annotate as an
    box = an.to_crop_box({"x": 256, "y": 192, "width": 512, "height": 384}, 1024, 768)
    assert box == {"x0": 0.25, "y0": 0.25, "x1": 0.75, "y1": 0.75}
    # dibujado de derecha a izquierda / de abajo hacia arriba, y saliendose de la foto
    assert an.to_crop_box({"x": 768, "y": 576, "width": -512, "height": -384}, 1024, 768) == box
    assert an.to_crop_box({"x": 900, "y": -50, "width": 300, "height": 100}, 1024, 768)["x1"] == 1.0
    assert an.from_crop_box(box, 2048, 1536) == {"x": 512, "y": 384, "width": 1024, "height": 768}
    f = tmp_path / "o.jpg"
    Image.new("RGB", (2048, 1536)).save(f)
    assert an.crop_photo(f, box).size == (1024, 768)


def test_crop_box_text():
    from sepia import annotate as an
    box = {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.9}
    assert an.parse_crop_box(an.format_crop_box(box)) == box
    assert an.parse_crop_box("") is None
    with pytest.raises(ValueError):
        an.parse_crop_box("0.5, 0.2, 0.1, 0.9")                  # x0 > x1


def test_annotation_app_boxes(api, cfg):
    """Boton Draw box -> arrastrar -> coordenadas en el campo -> se guardan en crop_box."""
    pytest.importorskip("jupyter_bbox_widget")
    _ready(cfg)
    app = sepia.AnnotationApp(cfg, annotator="HZ")
    assert app.bbox is not None and app.bbox.view_only is True     # no se dibuja sin el boton
    W, H = app._box_size
    app.bbox.bboxes = [{"x": 0, "y": 0, "width": W / 3, "height": H / 3, "label": "a"}]
    assert app.crop.value == ""                                    # sin boton: se ignora
    app.draw_btn.value = True
    assert app.bbox.view_only is False and app.bbox.label == "a"
    app.bbox.bboxes = [{"x": 0, "y": 0, "width": W / 2, "height": H / 2, "label": "a"}]
    assert app.crop.value == "0.0000, 0.0000, 0.5000, 0.5000"      # al soltar: el campo
    assert app.draw_btn.value is False and app.bbox.view_only is True
    app.inputs["state"].value = "cuttlebone"
    app._safe(app.on_save)
    ind = sepia.load_events(cfg)["individual"]
    assert ind.iloc[0]["crop_box"] == {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5}
    assert len(app.bbox.bboxes) == 1 and app.crop.value.endswith("0.5000")   # recargado
    app.crop.value = "0.1, 0.1, 0.6, 0.6"                           # corregir a mano
    app._safe(app.on_save)
    assert sepia.load_events(cfg)["individual"].iloc[0]["crop_box"]["x1"] == 0.6
    app.crop.value = "nonsense"
    app._safe(app.on_save)
    assert "Error" in app.msg.value
    r = _excel_rows(sepia.build_excel(cfg, "box.xlsx"))[1][0]
    assert json.loads(r["crop_box"])["y1"] == 0.6



def test_photo_count(api, cfg):
    """Conteo por foto: se guarda con el individuo, 'solo conteo' cuenta como anotada, va al Excel."""
    pytest.importorskip("ipywidgets")
    _ready(cfg)
    app = sepia.AnnotationApp(cfg, annotator="HZ", boxes=False)
    app.n_photo.value = 2
    app._safe(app.on_save_next)                                  # a en foto 11, con 2 visibles
    app.n_photo.value = 27
    app._safe(app.on_count_only)                                 # foto 12: muchos huesos, solo conteo
    assert app.row["photo_id"] == 21
    ev = sepia.load_events(cfg)
    assert dict(zip(ev["photo_info"]["photo_id"], ev["photo_info"]["n_in_photo"])) == {11: 2, 12: 27}
    app._safe(app.on_empty)                                      # foto 21: sin individuos -> 0
    ev = sepia.load_events(cfg)
    assert ev["photo_info"].set_index("photo_id").loc[21, "n_in_photo"] == 0
    assert sepia.annotate.first_pending(app.queue, ev) == 3      # 11, 12, 21 hechas
    app._go(0)
    assert app.n_photo.value == 2                                # se recarga lo guardado
    app._safe(app.on_save)                                       # mismo conteo: no repite linea
    lines = sepia.read_jsonl(sepia.annotate.annotation_file(cfg, "HZ"))
    assert (lines["kind"] == "photo_info").sum() == 3
    app.n_photo.value = 0
    app._safe(app.on_save)
    assert "Note" in app.msg.value                               # 1 anotado > 0 en la foto
    rows = {r["photo_id"]: r for r in _excel_rows(sepia.build_excel(cfg, "cnt.xlsx"))[1]}
    assert rows[12]["number individuals in photo (cuttlebone)"] == 27 and rows[12]["reviewed"] is True
    assert rows[12]["individual (a,b,c,d,e)"] is None


class FakeTraits:
    name = "fake"
    def classify(self, images, prompt, schema=None):
        assert schema is sepia.photo_traits.PhotoTraits and len(images) == 1
        res = {"n_cuttlebones_visible": 1, "side": "ventral", "erosion": "2",
               "photo_quality": "medium", "notes": ""}
        return res, json.dumps(res), {"input_tokens": 300, "output_tokens": 20}


def test_photo_traits(api, cfg, monkeypatch):
    """Una consulta por foto, reanudable; regla n!=1; evaluacion contra la anotacion."""
    from sepia import annotate as an, photo_traits as pt
    _ready(cfg)
    cfg.photo_traits = {"provider": "fake", "model": "t1", "requests_per_minute": 100000}
    monkeypatch.setattr(pt.time, "sleep", lambda s: None)
    pt.run_photo_traits(cfg, provider=FakeTraits(), limit=4)
    pt.run_photo_traits(cfg, provider=FakeTraits())               # reanuda: las 10 que faltan
    t = pt.load_photo_traits(cfg)
    assert len(t) == 14 and t["side"].eq("ventral").all()
    r, rules = pt.apply_rules({"n_cuttlebones_visible": 3, "side": "dorsal", "erosion": "1"})
    assert r["side"] == r["erosion"] == r["photo_quality"] == "not_applicable" and len(rules) == 3
    # anotacion: foto 11 ventral/2 (acuerdo), foto 12 dorsal/1 (desacuerdo)
    an.save_individual(cfg, "HZ", 1, 11, "a", {"state": "cuttlebone", "side": "ventral", "erosion": 2})
    an.save_photo_count(cfg, "HZ", 1, 11, 1)
    an.save_individual(cfg, "HZ", 1, 12, "a", {"state": "cuttlebone", "side": "dorsal", "erosion": 1})
    an.save_photo_count(cfg, "HZ", 1, 12, 2)
    cmp = pt.evaluate_photo_traits(cfg)
    c = cmp.set_index("photo_id")
    assert c.loc[11, "side_human"] == "ventral" and c.loc[12, "side_model"] == "ventral"
    assert c.loc[12, "n_in_photo"] == 2 and c.loc[11, "photo_quality_model"] == "medium"



def test_possible_duplicate(api, cfg):
    """Mismo hueso en otro registro: se valida, se guarda y va al Excel."""
    pytest.importorskip("ipywidgets")
    from sepia import annotate as an
    assert an.parse_duplicates(" 405594162:A , 409584581") == "405594162:a, 409584581"
    assert an.parse_duplicates("337987515 :a") == "337987515:a"
    assert an.parse_duplicates("337987515 : a; 1") == "337987515:a, 1"
    assert an.parse_duplicates("") is None
    with pytest.raises(ValueError):
        an.parse_duplicates("record 4055")
    _ready(cfg)
    app = sepia.AnnotationApp(cfg, annotator="HZ", boxes=False)
    app.inputs["possible_duplicate_of"].value = "1:a"            # su propio registro: error
    app._safe(app.on_save)
    assert "same record" in app.msg.value
    app.inputs["possible_duplicate_of"].value = "5:b"
    app._safe(app.on_save)
    r = _excel_rows(sepia.build_excel(cfg, "dup.xlsx"))[1][0]
    assert r["same bone as other record (ID:letter)"] == "5:b"



def test_duplicate_photo(api, cfg):
    """Foto repetida: cuenta como revisada, sin individuos, y queda trazada en el Excel."""
    pytest.importorskip("ipywidgets")
    _ready(cfg)
    app = sepia.AnnotationApp(cfg, annotator="HZ", boxes=False)
    app._safe(app.on_save_next)                                  # foto 11: individuo a
    assert app.row["photo_id"] == 12 and app.dup_of.value == 11
    app._safe(app.on_duplicate)                                  # foto 12 = igual que la 11
    assert app.row["photo_id"] == 21 and "same as" in app.msg.value.lower()
    ev = sepia.load_events(cfg)
    assert sepia.annotate.first_pending(app.queue, ev) == 2
    app._go(1)
    assert "Same as photo 1" in app.table.value
    rows = {r["photo_id"]: r for r in _excel_rows(sepia.build_excel(cfg, "dupph.xlsx"))[1]}
    assert rows[12]["same photo as (photo ID)"] == 11 and rows[12]["reviewed"] is True
    assert rows[12]["individual (a,b,c,d,e)"] is None and rows[11]["number individuals (cuttlebone)"] == 1
    app._safe(app.on_save)                                       # anotar igual la foto 12: deshace
    assert len(sepia.load_events(cfg)["duplicate_photo"]) == 0



def test_keyboard_shortcuts(api, cfg):
    """Alt+S guarda y avanza; Alt+A guarda y se queda; Alt+P vuelve."""
    pytest.importorskip("anywidget")
    _ready(cfg)
    app = sepia.AnnotationApp(cfg, annotator="HZ", boxes=False)
    assert app.keys is not None and "KeyS" in app.keys.codes
    send = lambda code: app.keys._handle_custom_msg({"code": code}, [])
    send("KeyA")                                                 # guarda a, se queda en la foto 11
    assert app.row["photo_id"] == 11 and app.letter.value == "b"
    t0 = app.keys.tick
    send("KeyS")                                                 # guarda b y avanza
    assert app.row["photo_id"] == 12 and app.keys.tick == t0 + 1  # avisa: volver arriba
    send("KeyP")
    assert app.row["photo_id"] == 11
    send("KeyZ")                                                 # tecla sin accion: no hace nada
    assert len(sepia.load_events(cfg)["individual"]) == 2


# --------------------------------------------------------------------------
# Un archivo por version de prompt (30 oct 2026)
# --------------------------------------------------------------------------

def test_one_file_per_prompt_version(api, cfg, monkeypatch):
    """Cada version escribe su archivo; el archivo antiguo se sigue leyendo y se puede separar."""
    from sepia import io as sio
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    sepia.download_photos(cfg)
    sepia.make_resized(cfg, 512)
    monkeypatch.setattr(sepia.triage.time, "sleep", lambda s: None)
    prov = FakeProvider()
    cfg.triage = {"provider": "fake", "model": "pf", "requests_per_minute": 100000,
                  "prompt_version": "triage_v1"}
    sepia.run_triage(cfg, provider=prov, ids=[1, 2])
    cfg.triage["prompt_version"] = "triage_v2"
    sepia.run_triage(cfg, provider=prov, ids=[1])
    v1 = sepia.triage.output_path(cfg, "fake", "pf", "triage_v1")
    v2 = sepia.triage.output_path(cfg, "fake", "pf", "triage_v2")
    assert len(sepia.read_jsonl(v1)) == 2 and len(sepia.read_jsonl(v2)) == 1
    assert set(sepia.read_jsonl(v2)["prompt_version"]) == {"triage_v2"}

    # archivo antiguo (todas las versiones juntas): se lee sin separarlo...
    legacy = sio.legacy_output_path(cfg, "triage", "fake", "pf")
    old = sepia.read_jsonl(v1)
    old["observation_id"] = old["observation_id"] + 2               # registros 3 y 4
    v1.unlink()
    sio.write_jsonl(old, legacy)
    assert set(sepia.load_triage(cfg, "fake", "pf", "triage_v1")["observation_id"]) == {3, 4}
    n_calls = len(prov.calls)
    cfg.triage["prompt_version"] = "triage_v1"
    sepia.run_triage(cfg, provider=prov, ids=[3, 4])               # ya hechos en el antiguo: salta
    assert len(prov.calls) == n_calls
    # ...y al separarlo no se pierde ni se duplica nada
    assert sio.split_legacy_outputs(cfg, "triage", "fake", "pf") == {"triage_v1": 2}
    assert not legacy.exists() and legacy.with_suffix(".jsonl.legacy").exists()
    assert len(sepia.read_jsonl(v1)) == 2
    assert set(sepia.load_triage(cfg, "fake", "pf", "triage_v1")["observation_id"]) == {3, 4}


class FakeRecordTraits:
    """Responde por registro: foto 1 ventral directa, foto 2 dorsal inferida del mismo hueso."""
    name = "fake"
    def __init__(self, bad_index=False):
        self.calls, self.bad_index = [], bad_index
    def classify(self, images, prompt, schema=None):
        pt_ = sepia.photo_traits
        assert schema in (pt_.RecordPhotoTraits, pt_.RecordPhotoTraitsV7, pt_.RecordPhotoTraitsV9,
                          pt_.RecordPhotoTraitsV10, pt_.RecordPhotoTraitsV11)
        v7 = schema is not pt_.RecordPhotoTraits
        v10 = schema in (pt_.RecordPhotoTraitsV10, pt_.RecordPhotoTraitsV11)
        self.calls.append(len(images))
        photos = []
        for i in range(1, len(images) + 1):
            photos.append({"photo_index": 1 if self.bad_index else i,
                           "n_cuttlebones_visible": 1,
                           "side": "ventral" if i == 1 else "dorsal",
                           "side_basis": "direct" if i == 1 else "inferred_same_bone",
                           "erosion": "1", "photo_quality": "high", "notes": "",
                           **({"side_evidence": "porous surface"} if v7 else {}),
                           **({"side_signs": ["inner_cone_rim"] if i == 1 else ["granular_shield"]}
                              if v10 else {})})
        res = {"photos": photos}
        return res, json.dumps(res), {"input_tokens": 900, "output_tokens": 60}


def test_photo_traits_by_record(api, cfg, monkeypatch):
    """v5: una consulta por registro, una linea por foto, archivo propio, side_basis evaluado."""
    from sepia import annotate as an, photo_traits as pt
    _ready(cfg)
    cfg.photo_traits = {"provider": "fake", "model": "t2", "requests_per_minute": 100000,
                        "prompt_version": "photo_traits_v4"}
    monkeypatch.setattr(pt.time, "sleep", lambda s: None)
    prov = FakeRecordTraits()
    pt.run_photo_traits(cfg, provider=prov, limit=2, prompt_version="photo_traits_v5")
    pt.run_photo_traits(cfg, provider=prov, prompt_version="photo_traits_v5")     # reanuda
    assert prov.calls == [2] * 7                                   # 7 registros, 2 fotos cada uno
    out = pt.output_path(cfg, "fake", "t2", "photo_traits_v5")
    raw = sepia.read_jsonl(out)
    assert len(raw) == 14 and raw["call_id"].nunique() == 7 and set(raw["unit"]) == {"record"}
    assert not pt.output_path(cfg, "fake", "t2", "photo_traits_v4").exists()
    t = pt.load_photo_traits(cfg, "photo_traits_v5")
    assert set(t.loc[t["position"] == 1, "side"]) == {"ventral"}
    assert set(t.loc[t["position"] == 2, "side_basis"]) == {"inferred_same_bone"}
    assert pt.load_photo_traits(cfg).empty                         # el YAML sigue en v4

    # respuesta sin una entrada por foto -> error en todas las fotos del registro, se reintenta
    bad = pt.output_path(cfg, "fake", "t2", "photo_traits_v6")
    pt.run_photo_traits(cfg, provider=FakeRecordTraits(bad_index=True), ids=[1],
                        prompt_version="photo_traits_v6")
    b = sepia.read_jsonl(bad)
    assert len(b) == 2 and b["error"].notna().all()
    pt.run_photo_traits(cfg, provider=FakeRecordTraits(), ids=[1], prompt_version="photo_traits_v6")
    assert len(pt.load_photo_traits(cfg, "photo_traits_v6")) == 2

    # evaluacion: foto 11 ventral (acierto, direct), foto 12 ventral (falla, inferred_same_bone)
    for pid in (11, 12):
        an.save_individual(cfg, "HZ", 1, pid, "a", {"state": "cuttlebone", "side": "ventral", "erosion": 1})
        an.save_photo_count(cfg, "HZ", 1, pid, 1)
    c = pt.evaluate_photo_traits(cfg, "photo_traits_v5").set_index("photo_id")
    assert c.loc[11, "side_basis"] == "direct" and c.loc[12, "side_model"] == "dorsal"


def test_photo_traits_rules_side_basis():
    from sepia import photo_traits as pt
    r, rules = pt.apply_rules({"n_cuttlebones_visible": 1, "side": "unclear",
                               "side_basis": "inferred_other_bone", "erosion": "1"})
    assert r["side_basis"] == "unclear" and rules == ["side=unclear -> side_basis=unclear"]
    r, _ = pt.apply_rules({"n_cuttlebones_visible": 2, "side": "dorsal",
                           "side_basis": "direct", "erosion": "1"})
    assert r["side_basis"] == "not_applicable"
    for v in ("photo_traits_v5", "photo_traits_v6"):
        text, _ = sepia.load_prompt(v)
        assert "side_basis" in text and pt.PROMPT_UNIT[v] == "record"
    assert "LENGTHWISE" in sepia.load_prompt("photo_traits_v6")[0]
    assert "LENGTHWISE" not in sepia.load_prompt("photo_traits_v5")[0]


def test_metrics_saved_and_compare(api, cfg, monkeypatch):
    """Cada evaluacion agrega una linea de metricas; --compare usa las mismas fotos."""
    from sepia import annotate as an, photo_traits as pt
    _ready(cfg)
    cfg.photo_traits = {"provider": "fake", "model": "t3", "requests_per_minute": 100000,
                        "prompt_version": "photo_traits_v6"}
    monkeypatch.setattr(pt.time, "sleep", lambda s: None)
    pt.run_photo_traits(cfg, provider=FakeRecordTraits(), prompt_version="photo_traits_v6")
    pt.run_photo_traits(cfg, provider=FakeRecordTraits(), ids=[1], prompt_version="photo_traits_v7")
    t7 = pt.load_photo_traits(cfg, "photo_traits_v7")
    assert set(t7["side_evidence"]) == {"porous surface"} and len(t7) == 2
    for pid in (11, 12, 21):
        an.save_individual(cfg, "HZ", pid // 10, pid, "a",
                           {"state": "cuttlebone", "side": "ventral", "erosion": 1})
        an.save_photo_count(cfg, "HZ", pid // 10, pid, 1)
    t = pt.evaluate_photo_traits(cfg, "photo_traits_v6")
    m = sepia.load_metrics(cfg, "photo_traits")
    assert len(m) == 1 and m.iloc[0]["prompt_version"] == "photo_traits_v6"
    assert m.iloc[0]["metrics"]["side_n"] == 3 and m.iloc[0]["n_photos_compared"] == 3
    assert t.attrs["metrics"]["side_by_basis"]["direct"]["n"] == 2
    out = pt.compare_photo_traits(cfg, ["photo_traits_v6", "photo_traits_v7"])
    assert list(out["side_n"]) == [2, 2]                           # solo fotos del registro 1
    m = sepia.load_metrics(cfg, "photo_traits")
    assert len(m) == 3 and m.iloc[1]["eval_set_sha"] == m.iloc[2]["eval_set_sha"]
    assert "side_evidence" in sepia.load_prompt("photo_traits_v7")[0]
    v8 = sepia.load_prompt("photo_traits_v8")[0]
    assert "ridges" in v8 and "relief" in v8 and "parallel lines" not in v8
    assert pt.SCHEMA_BY_VERSION["photo_traits_v8"] is pt.RecordPhotoTraitsV7
    # visor: tabla por foto con cada version y la persona
    from sepia import review
    rt = review.review_table(cfg, ["photo_traits_v6", "photo_traits_v7"])
    assert len(rt) == 14 and "side_evidence__photo_traits_v7" in rt and "side__human" in rt
    assert len(review.filter_rows(rt, ["photo_traits_v6"], "annotated")) == 3
    wrong = review.filter_rows(rt, ["photo_traits_v6"], "model_wrong")
    assert set(wrong["photo_id"]) == {12}               # foto 2: dorsal para el modelo, ventral anotada
    assert len(review.filter_rows(rt, ["photo_traits_v6", "photo_traits_v7"], "versions_differ")) == 0
    vv = ["photo_traits_v6", "photo_traits_v7"]
    assert set(review.filter_rows(rt, vv, "versions_agree_human_differs")["photo_id"]) == {12}
    assert set(review.filter_rows(rt, vv, "versions_agree_human_agrees")["photo_id"]) == {11}
    v9 = sepia.load_prompt("photo_traits_v9")[0]
    assert "bands_rule" in v9 and "ACROSS" in v9
    pt.RecordPhotoTraitsV9.model_validate({"photos": [{
        "photo_index": 1, "n_cuttlebones_visible": 1, "side": "ventral", "side_basis": "bands_rule",
        "side_evidence": "curved transverse bands", "erosion": "1", "photo_quality": "high",
        "notes": ""}]})


def test_side_from_signs_and_v10(api, cfg, monkeypatch):
    """v10-v11: tabla fija rasgos -> lado, side_rule guardado y evaluado."""
    from sepia import annotate as an, photo_traits as pt
    f = pt.side_from_signs
    assert f(["inner_cone_rim"]) == "ventral"
    assert f(["striated_zone", "white_chalky_or_porous"]) == "ventral"
    assert f(["granular_shield"]) == "dorsal"
    assert f(["granular_shield", "median_groove"]) == "unclear"        # contradiccion debil
    assert f(["inner_cone_rim", "chitinous_posterior_cap"]) == "unclear"
    assert f(["white_chalky_or_porous", "transverse_lines"]) == "unclear"  # solo neutros
    assert f(["median_groove"]) == "unclear"                           # solo debiles
    assert f([]) == "unclear"
    assert f(["smooth_hard_uniform_face"], "high") == "dorsal"
    assert f(["smooth_hard_uniform_face"], "medium") == "unclear"
    assert f(["smooth_hard_uniform_face", "bite_marks"], "high") == "unclear"
    r, _ = pt.apply_rules({"n_cuttlebones_visible": 2, "side": "dorsal", "side_basis": "direct",
                           "side_signs": ["granular_shield"], "erosion": "1"})
    assert r["side_signs"] == [] and r["side_rule"] == "not_applicable"
    for v in ("photo_traits_v10", "photo_traits_v11"):
        text = sepia.load_prompt(v)[0]
        assert "inner_cone_rim" in text and "side_signs" in text and pt.PROMPT_UNIT[v] == "record"
        assert "lines_rule" not in text
    assert "laminated_surface" in sepia.load_prompt("photo_traits_v11")[0]
    assert "laminated_surface" not in sepia.load_prompt("photo_traits_v10")[0]

    _ready(cfg)
    cfg.photo_traits = {"provider": "fake", "model": "t4", "requests_per_minute": 100000,
                        "prompt_version": "photo_traits_v10"}
    monkeypatch.setattr(pt.time, "sleep", lambda s: None)
    pt.run_photo_traits(cfg, provider=FakeRecordTraits(), ids=[1, 2])
    t = pt.load_photo_traits(cfg)
    assert set(t.loc[t["position"] == 1, "side_rule"]) == {"ventral"}
    assert set(t.loc[t["position"] == 2, "side_rule"]) == {"dorsal"}
    for pid, side, ero in ((11, "ventral", 1), (12, "ventral", 3), (21, "ventral", 1)):
        an.save_individual(cfg, "HZ", pid // 10, pid, "a",
                           {"state": "cuttlebone", "side": side, "erosion": ero})
        an.save_photo_count(cfg, "HZ", pid // 10, pid, 1)
    e = pt.evaluate_photo_traits(cfg)
    m = e.attrs["metrics"]
    assert m["side_rule_n"] == 3 and m["side_rule_agreement"] == 2 / 3
    assert m["side_rule_by_erosion"]["3"]["agreement_when_decided"] == 0.0
    assert m["side_sign_reliability"]["inner_cone_rim"] == {"n": 2, "human_dorsal": 0, "human_ventral": 2}
    from sepia import review
    rt = review.review_table(cfg, ["photo_traits_v10"])
    assert "side_rule__photo_traits_v10" in rt and review._norm(["a", "b"]) == "a, b"


class FakeBitemarks:
    """Foto en posicion 1: muesca limpia (yes por regla). Posicion 2: muesca que podria ser
    borde roto (no por regla), aunque el modelo diga yes."""
    name = "fake"
    def __init__(self):
        self.calls = 0
    def classify(self, images, prompt, schema=None):
        assert schema is sepia.bitemarks.BitemarksResult and len(images) == 1
        self.calls += 1
        broken = self.calls % 2 == 0
        res = {"n_cuttlebones_visible": 1, "assessable": "yes", "bitemarks": "yes", "notes": "",
               "candidates": [{"location": "edge", "shape": "triangular_notch", "edges": "sharp",
                               "is_sand_or_debris": False, "is_broken_edge": broken,
                               "is_crack": False, "is_erosion_pit": False,
                               "is_shadow_or_reflection": False, "description": "V notch"}]}
        return res, json.dumps(res), {"input_tokens": 500, "output_tokens": 80}


def test_bitemarks(api, cfg, monkeypatch):
    """Paso aparte: regla fija desde las candidatas, archivo propio, evaluacion y metricas."""
    from sepia import annotate as an, bitemarks as bm
    f = bm.rule_from_candidates
    clean = {"shape": "triangular_notch", "edges": "sharp", **{k: False for k in bm.CHECKS}}
    assert f([clean], 1, "yes") == "yes"
    assert f([{**clean, "edges": "worn"}], 1, "yes") == "probable"
    assert f([{**clean, "is_crack": True}], 1, "yes") == "no"
    assert f([{**clean, "shape": "round_hole"}], 1, "yes") == "no"
    assert f([], 1, "yes") == "no"
    assert f([clean], 1, "no") == "not_assessable"
    assert f([clean], 2, "yes") == "not_applicable"
    r, _ = bm.apply_rules({"n_cuttlebones_visible": 3, "assessable": "yes", "bitemarks": "yes",
                           "candidates": [clean], "notes": ""})
    assert r["bitemarks"] == "not_applicable" and r["candidates"] == [] and r["n_candidates"] == 0
    assert "is_broken_edge" in sepia.load_prompt("bitemarks_v1")[0]

    _ready(cfg)
    monkeypatch.setattr(bm.time, "sleep", lambda s: None)
    prov = FakeBitemarks()
    bm.run_bitemarks(cfg, provider=prov, ids=[1, 2])
    bm.run_bitemarks(cfg, provider=prov, ids=[1, 2])                      # reanuda: nada nuevo
    assert prov.calls == 4
    out = bm.output_path(cfg, "gemini", "gemini-3.5-flash-lite", "bitemarks_v1")
    assert out.exists() and len(sepia.read_jsonl(out)) == 4
    t = bm.load_bitemarks(cfg)
    assert set(t.loc[t["position"] == 1, "bitemarks_rule"]) == {"yes"}
    assert set(t.loc[t["position"] == 2, "bitemarks_rule"]) == {"no"}
    # anotador: 11 yes (ventral), 12 no (ventral), 21 (yes) (ventral)
    for pid, bmv in ((11, "yes"), (12, "no"), (21, "(yes)")):
        an.save_individual(cfg, "HZ", pid // 10, pid, "a",
                           {"state": "cuttlebone", "side": "ventral", "erosion": 1, "bitemarks": bmv})
    e = bm.evaluate_bitemarks(cfg)
    m = e.attrs["metrics"]
    assert m["bitemarks_all"]["fp"] == 1 and m["bitemarks_all"]["tp"] == 2      # el modelo dice yes a todo
    assert m["bitemarks_rule_all"] == {**m["bitemarks_rule_all"], "tp": 2, "fp": 0, "tn": 1, "fn": 0}
    assert m["bitemarks_rule_all"]["precision"] == 1.0
    assert len(sepia.load_metrics(cfg, "bitemarks")) == 1
    from sepia import review
    rt = review.review_table(cfg, ["bitemarks_v1"])
    assert "bitemarks_rule__bitemarks_v1" in rt and "bitemarks__human" in rt
    assert "broken_edge" in review._norm({**clean, "is_broken_edge": True, "location": "edge",
                                          "description": "x"})


def test_v12_erosion_rule():
    """v12: v8 + 'si dudas, unclear' en el prompt; erosion 3 -> side unclear en el codigo."""
    from sepia import photo_traits as pt
    base = {"n_cuttlebones_visible": 1, "side": "ventral", "side_basis": "direct",
            "side_evidence": "rim", "erosion": "3", "photo_quality": "high", "notes": ""}
    r, rules = pt.apply_rules(base, "photo_traits_v12")
    assert r["side"] == "unclear" and r["side_raw"] == "ventral" and r["side_basis"] == "unclear"
    assert "erosion=3 -> side=unclear" in rules
    r, _ = pt.apply_rules({**base, "erosion": "2"}, "photo_traits_v12")
    assert r["side"] == "ventral" and r["side_raw"] == "ventral"
    r, _ = pt.apply_rules(base, "photo_traits_v8")                  # otras versiones: sin regla
    assert r["side"] == "ventral" and "side_raw" not in r
    v12 = sepia.load_prompt("photo_traits_v12")[0]
    v8 = sepia.load_prompt("photo_traits_v8")[0]
    assert "A wrong side is worse" in v12 and "A wrong side is worse" not in v8
    assert v12.replace(v8, "") != v12 or len(v12) > len(v8)
    assert pt.SCHEMA_BY_VERSION["photo_traits_v12"] is pt.RecordPhotoTraitsV7


def test_v13_side_text():
    """v13 = v12 sin 'chalky/porous' como señal ventral y con dorsal 'smoother, harder'."""
    from sepia import photo_traits as pt
    v12 = sepia.load_prompt("photo_traits_v12")[0]
    v13 = sepia.load_prompt("photo_traits_v13")[0]
    assert "chalky and porous-looking" in v12 and "chalky and porous-looking" not in v13
    assert '"porous chalky surface"' not in v13
    assert "smoother, harder" in v13
    assert pt.SCHEMA_BY_VERSION["photo_traits_v13"] is pt.RecordPhotoTraitsV7
    assert pt.PROMPT_UNIT["photo_traits_v13"] == "record"
    r, _ = pt.apply_rules({"n_cuttlebones_visible": 1, "side": "ventral", "side_basis": "direct",
                           "side_evidence": "rim", "erosion": "3", "photo_quality": "high",
                           "notes": ""}, "photo_traits_v13")
    assert r["side"] == "unclear" and r["side_raw"] == "ventral"


def test_v14_quality_rule():
    """v14: el modelo describe cover y focus; photo_quality la decide quality_from_cover."""
    from sepia import photo_traits as pt
    q = pt.quality_from_cover
    assert q("none", "sharp") == "high" and q("few_grains", "sharp") == "high"
    assert q("scattered_grains_or_small_objects", "sharp") == "medium"
    assert q("large_part_covered", "sharp") == "low" and q("none", "blurry") == "low"
    assert q("not_applicable", "not_applicable") == "not_applicable"
    base = {"photo_index": 1, "n_cuttlebones_visible": 1, "side": "dorsal", "side_basis": "direct",
            "side_evidence": "ridges", "erosion": "1", "cover": "scattered_grains_or_small_objects",
            "focus": "sharp", "quality_evidence": "many grains on tip", "notes": ""}
    pt.RecordPhotoTraitsV14(photos=[base])                       # el esquema acepta la respuesta
    assert "photo_quality" not in pt.PhotoTraitsInRecordV14.model_fields
    r, rules = pt.apply_rules(base, "photo_traits_v14")
    assert r["photo_quality"] == "medium" and r["side"] == "dorsal"
    r, _ = pt.apply_rules({**base, "n_cuttlebones_visible": 2}, "photo_traits_v14")
    assert r["photo_quality"] == "not_applicable" and r["cover"] == "not_applicable"
    r, _ = pt.apply_rules({**base, "erosion": "3"}, "photo_traits_v14")   # regla de erosion 3
    assert r["side"] == "unclear" and r["side_raw"] == "dorsal"
    v13 = sepia.load_prompt("photo_traits_v13")[0]
    v14 = sepia.load_prompt("photo_traits_v14")[0]
    assert v13.split('"erosion": how much')[0].replace(
        '"side_evidence", "erosion", "cover", "focus" and "quality_evidence"',
        '"erosion" and "photo_quality"') == v14.split('"erosion": how much')[0].replace(
        '"side_evidence", "erosion", "cover", "focus" and "quality_evidence"',
        '"erosion" and "photo_quality"')                         # el lado no cambia
    assert "can still be \"high\"" not in v14 and "few_grains" in v14


def test_classification_metrics_human_unclear(tmp_path):
    """side 'unclear' de la persona: fila extra en la matriz, fuera de las metricas."""
    import json
    from sepia.disagree import classification_metrics
    cfg = sepia.Config(base_dir=str(tmp_path))
    p = sepia.paths(cfg)
    photos = [(1, 11, 1), (2, 21, 1), (3, 31, 1)]
    human = {11: "dorsal", 21: "ventral", 31: "unclear"}
    model = {11: "dorsal", 21: "ventral", 31: "dorsal"}
    with open(p["tables"] / "photos.jsonl", "w") as f:
        for o, ph, pos in photos:
            f.write(json.dumps({"observation_id": o, "photo_id": ph, "position": pos,
                                "file_name": f"{o}_{ph}.jpg"}) + "\n")
    with open(p["annotations"] / "manual_T.jsonl", "w") as f:
        for o, ph, pos in photos:
            f.write(json.dumps({"kind": "photo_info", "observation_id": o, "photo_id": ph,
                                "n_in_photo": 1, "saved_at": "2026-11-01"}) + "\n")
            f.write(json.dumps({"kind": "individual", "observation_id": o, "photo_id": ph,
                                "individual": "a", "side": human[ph], "erosion": 1,
                                "photo_quality": "high", "saved_at": "2026-11-01"}) + "\n")
    out = p["base"] / "outputs"
    out.mkdir(exist_ok=True)
    with open(out / "photo_traits_gemini_gemini-3.5-flash-lite_photo_traits_v8.jsonl", "w") as f:
        for o, ph, pos in photos:
            f.write(json.dumps({"observation_id": o, "photo_id": ph, "position": pos, "error": None,
                                "saved_at": "2026-11-02", "prompt_sha": "x",
                                "result": {"n_cuttlebones_visible": 1, "side": model[ph],
                                           "erosion": "1", "photo_quality": "high"}}) + "\n")
    s, pc, m = classification_metrics(cfg, ["photo_traits_v8"], fields=("side",),
                                      save=False, quiet=True)
    cm = m[("photo_traits_v8", "side")]
    assert list(cm.index) == ["dorsal", "ventral", "unclear"] and cm.loc["unclear", "dorsal"] == 1
    r = s.iloc[0]
    assert r["n"] == 2 and r["accuracy"] == 1.0 and r["human_unclear_n"] == 1
    assert r["human_unclear_model_unclear"] == 0
    d = pc[pc["class"] == "dorsal"].iloc[0]
    assert d["precision"] == 1.0 and d["predicted"] == 1        # la fila unclear no cuenta


def test_martin_views(tmp_path):
    """Vistas para Martin: unclear cuenta como diferencia; notas que mencionan a Martin."""
    import json
    from sepia.disagree import disagreement_table, mentions_martin
    assert mentions_martin("Preguntar a Martín") and mentions_martin("hablar con MARTHIN")
    assert not mentions_martin("martes") and not mentions_martin(None)
    assert not mentions_martin("Martina dijo") and not mentions_martin("smartinez")
    assert mentions_martin("MARTIN?") and mentions_martin("hablar con Martin.")
    cfg = sepia.Config(base_dir=str(tmp_path))
    p = sepia.paths(cfg)
    photos = [(1, 11, 1), (2, 21, 1), (3, 31, 1)]
    human = {11: ("dorsal", ""), 21: ("unclear", "preguntar a martin"), 31: ("ventral", "")}
    model = {"v8": {11: "ventral", 21: "dorsal", 31: "unclear"},
             "v9": {11: "ventral", 21: "dorsal", 31: "ventral"}}
    with open(p["tables"] / "photos.jsonl", "w") as f:
        for o, ph, pos in photos:
            f.write(json.dumps({"observation_id": o, "photo_id": ph, "position": pos,
                                "file_name": f"{o}_{ph}.jpg"}) + "\n")
    with open(p["annotations"] / "manual_T.jsonl", "w") as f:
        for o, ph, pos in photos:
            f.write(json.dumps({"kind": "photo_info", "observation_id": o, "photo_id": ph,
                                "n_in_photo": 1, "saved_at": "2026-11-01"}) + "\n")
            f.write(json.dumps({"kind": "individual", "observation_id": o, "photo_id": ph,
                                "individual": "a", "side": human[ph][0], "erosion": 1,
                                "photo_quality": "high", "notes": human[ph][1],
                                "saved_at": "2026-11-01"}) + "\n")
    out = p["base"] / "outputs"
    out.mkdir(exist_ok=True)
    for v, sides in model.items():
        with open(out / f"photo_traits_gemini_gemini-3.5-flash-lite_photo_traits_{v}.jsonl", "w") as f:
            for o, ph, pos in photos:
                f.write(json.dumps({"observation_id": o, "photo_id": ph, "position": pos,
                                    "error": None, "saved_at": "2026-11-02", "prompt_sha": "x",
                                    "result": {"n_cuttlebones_visible": 1, "side": sides[ph],
                                               "erosion": "1", "photo_quality": "high"}}) + "\n")
    t = disagreement_table(cfg, ["photo_traits_v8", "photo_traits_v9"]).set_index("photo_id")
    assert t.loc[11, "all_agree_differ__side"] and t.loc[11, "n_differ__side"] == 2
    assert t.loc[21, "all_agree_differ__side"]             # persona unclear, los dos dorsal
    assert t.loc[31, "n_differ__side"] == 1 and t.loc[31, "n_disagree__side"] == 0   # unclear cuenta
    assert not t.loc[31, "all_agree_differ__side"]
    assert list(t.index[t["ask_martin"]]) == [21]


def test_skip_ids_from_other_run(api, tmp_path):
    """Registros que ya estan en otra corrida no entran a las tablas (ni se bajan sus fotos)."""
    other = tmp_path / "pilot"
    (other / "tables").mkdir(parents=True)
    sepia.write_jsonl([{"observation_id": 1}, {"observation_id": 2}],
                      other / "tables" / "observations.jsonl")
    cfg = sepia.Config(base_dir=str(tmp_path / "new_run"), per_page=3, skip_ids_from=["pilot"])
    sepia.download_observations(cfg)
    obs, pho = sepia.build_tables(cfg)
    assert len(sepia.load_raw(cfg)) == 7                       # la copia cruda no se toca
    assert len(obs) == 5 and not obs["observation_id"].isin([1, 2]).any()
    assert not pho["observation_id"].isin([1, 2]).any()
    log = sepia.read_jsonl(sepia.paths(cfg)["logs"] / "skipped_existing.jsonl")
    assert sorted(log["observation_id"]) == [1, 2] and set(log["in_run"]) == {"pilot"}
    sepia.download_photos(cfg)
    names = {f.name for f in sepia.paths(cfg)["photos"].iterdir()}
    assert not any(n.startswith(("1_", "2_")) for n in names)
    bad = sepia.Config(base_dir=str(tmp_path / "x"), skip_ids_from=["missing_run"])
    sepia.download_observations(bad)
    with pytest.raises(FileNotFoundError):
        sepia.build_tables(bad)


def test_duplicate_candidates(tmp_path):
    """Pares del mismo dia con su distancia; grupos (trios); mismo usuario y otro usuario."""
    from sepia.duplicates import load_runs, candidate_pairs, candidate_groups
    def run(name, rows):
        d = tmp_path / name / "tables"
        d.mkdir(parents=True)
        sepia.write_jsonl(rows, d / "observations.jsonl")
    base = {"place_guess": "x", "n_photos": 1, "url": "u", "coordinates_obscured": False}
    run("r1", [{**base, "observation_id": 1, "observed_on": "2026-02-14", "time_observed_at":
                "2026-02-14T12:00:00+00:00", "latitude": 56.0, "longitude": -2.0, "user_login": "ana"},
               {**base, "observation_id": 2, "observed_on": "2026-02-14", "time_observed_at":
                "2026-02-14T12:20:00+00:00", "latitude": 56.005, "longitude": -2.0, "user_login": "ben"},
               {**base, "observation_id": 3, "observed_on": "2026-02-15", "time_observed_at": None,
                "latitude": 56.0, "longitude": -2.0, "user_login": "ana"}])          # otro dia
    run("r2", [{**base, "observation_id": 4, "observed_on": "2026-02-14", "time_observed_at": None,
                "latitude": 56.01, "longitude": -2.0, "user_login": "ana"},
               {**base, "observation_id": 5, "observed_on": "2026-02-14", "time_observed_at": None,
                "latitude": 57.0, "longitude": -2.0, "user_login": "cam"},          # lejos
               {**base, "observation_id": 1, "observed_on": "2026-02-14", "time_observed_at": None,
                "latitude": 56.0, "longitude": -2.0, "user_login": "ana"}])          # repetido
    cfg = sepia.Config(base_dir=str(tmp_path / "r1"))
    obs = load_runs(cfg, ["r1", "r2"])
    assert len(obs) == 5 and obs.loc[obs["observation_id"] == 1, "run"].item() == "r1"
    p = candidate_pairs(obs, max_km=2)
    assert {(a, b) for a, b in zip(p["id_a"], p["id_b"])} == {(1, 2), (1, 4), (2, 4)}
    r12 = p[(p["id_a"] == 1) & (p["id_b"] == 2)].iloc[0]
    assert abs(r12["distance_km"] - 0.556) < 0.01 and r12["minutes_apart"] == 20
    assert not r12["same_user"] and p[(p["id_a"] == 1) & (p["id_b"] == 4)]["same_user"].item()
    assert p["distance_km"].is_monotonic_increasing
    assert len(candidate_pairs(obs, max_km=None)) == 6        # todos los pares del dia 14
    g = candidate_groups(p)
    assert len(g) == 1 and g.iloc[0]["ids"] == [1, 2, 4] and g.iloc[0]["n_users"] == 2


def test_assisted_rules(tmp_path):
    """Anotacion asistida: manual > clic > todos coinciden; archivo fuera de annotations/."""
    from sepia.assisted import resolve, validate, default_bitemarks, assisted_file
    m = {"v1": "dorsal", "v2": "dorsal"}
    assert resolve(m) == ("dorsal", "all_agree")
    assert resolve({"v1": "unclear", "v2": "unclear"}) == ("unclear", "all_agree")
    assert resolve({"v1": "dorsal", "v2": "ventral"}) == ("", "")          # empate: elegir
    assert resolve({"v1": "dorsal", "v2": ""}) == ("dorsal", "mode:1/1")   # falta un prompt
    assert resolve({"a": "dorsal", "b": "dorsal", "c": "dorsal", "d": "ventral"}) == ("dorsal", "mode:3/4")
    assert resolve({"a": "1", "b": "1", "c": "2", "d": "2"}) == ("", "")    # empate 2-2
    assert resolve({"a": "1", "b": "1", "c": "2", "d": "3"}) == ("1", "mode:2/4")
    assert resolve({"a": "1", "b": "1", "c": "2"}, click="c") == ("2", "click:c")
    assert resolve({"v1": "dorsal", "v2": "ventral"}, click="v2") == ("ventral", "click:v2")
    assert resolve(m, click="v1", manual=" ventral ") == ("ventral", "manual")
    assert validate("side", "dorsal") is None and validate("side", "dorsl")
    assert validate("n_cuttlebones_visible", "2") is None and validate("n_cuttlebones_visible", "x")
    assert default_bitemarks("dorsal") == "no" and default_bitemarks("ventral") == "yes"
    assert default_bitemarks("unclear") == ""
    cfg = sepia.Config(base_dir=str(tmp_path))
    f = assisted_file(cfg, "HZ")
    assert f.parent.name == "annotations_assisted" and f.parent != sepia.paths(cfg)["annotations"]


def test_annotation_prefill_from_assisted(api, cfg):
    """AnnotationApp(prefill='assisted'): parte con lo de la asistida; marca prefilled_from."""
    pytest.importorskip("ipywidgets")
    from sepia.assisted import assisted_file
    _ready(cfg)
    f = assisted_file(cfg, "HZ")
    sepia.append_jsonl({"observation_id": 1, "photo_id": 11, "position": 1, "individual": "a",
                        "n_cuttlebones_visible": 1, "state": "cuttlebone", "side": "ventral",
                        "erosion": 2, "photo_quality": "medium", "bitemarks": "yes",
                        "epibionts": "no", "certainty": 1, "notes": "",
                        "saved_at": "2026-11-05T10:00:00+00:00"}, f)
    sepia.append_jsonl({"observation_id": 1, "photo_id": 12, "position": 2, "individual": None,
                        "n_cuttlebones_visible": 3, "needs_manual_annotation": True,
                        "saved_at": "2026-11-05T10:01:00+00:00"}, f)
    app = sepia.AnnotationApp(cfg, annotator="HZ", boxes=False, prefill="assisted")
    assert app.row["photo_id"] == 11
    v = app.inputs
    assert (v["side"].value, v["erosion"].value, v["photo_quality"].value,
            v["bitemarks"].value) == ("ventral", 2, "medium", "yes")
    assert "Prefilled" in app.table.value
    app._safe(app.on_save_next)
    lines = sepia.read_jsonl(sepia.annotate.annotation_file(cfg, "HZ"))
    ind = lines[lines["kind"] == "individual"].iloc[-1]
    assert ind["side"] == "ventral" and ind["prefilled_from"] == "assisted"
    assert app.row["photo_id"] == 12 and app.n_photo.value == 3        # solo el conteo
    assert app.inputs["side"].value is None                            # el resto a mano
    plain = sepia.AnnotationApp(cfg, annotator="HZ", boxes=False)      # sin prefill: como antes
    plain._go(1)
    assert plain.n_photo.value == 1


def test_disagreement_app_two_runs(tmp_path):
    """Dos corridas juntas solo en memoria: fotos de cada carpeta, anotacion asistida marcada."""
    pytest.importorskip("ipywidgets")
    import json
    from PIL import Image
    from sepia.disagree import DisagreementApp

    def make_run(name, oid, pid, side_h, prefilled):
        cfg = sepia.Config(base_dir=str(tmp_path / name))
        p = sepia.paths(cfg)
        sepia.write_jsonl([{"observation_id": oid, "photo_id": pid, "position": 1,
                            "file_name": f"{oid}_{pid}.jpg"}], p["tables"] / "photos.jsonl")
        d = p["base"] / "photos" / "1024px"
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), (pid % 255, 0, 0)).save(d / f"{oid}_{pid}.jpg")
        extra = {"prefilled_from": "assisted"} if prefilled else {}
        with open(p["annotations"] / "manual_HZ.jsonl", "w") as f:
            f.write(json.dumps({"kind": "photo_info", "observation_id": oid, "photo_id": pid,
                                "n_in_photo": 1, "saved_at": "2026-11-01", **extra}) + "\n")
            f.write(json.dumps({"kind": "individual", "observation_id": oid, "photo_id": pid,
                                "individual": "a", "side": side_h, "erosion": 1,
                                "photo_quality": "high", "saved_at": "2026-11-01", **extra}) + "\n")
        out = p["base"] / "outputs"
        out.mkdir(exist_ok=True)
        with open(out / "photo_traits_gemini_gemini-3.5-flash-lite_photo_traits_v8.jsonl", "w") as f:
            f.write(json.dumps({"observation_id": oid, "photo_id": pid, "position": 1,
                                "error": None, "saved_at": "2026-11-02", "prompt_sha": "x",
                                "result": {"n_cuttlebones_visible": 1, "side": "ventral",
                                           "erosion": "1", "photo_quality": "high"}}) + "\n")
        return cfg

    a1 = make_run("pilot_a1", 1, 11, "dorsal", False)
    a2 = make_run("a2_test", 2, 21, "ventral", True)
    app = DisagreementApp([a1, a2], ["photo_traits_v8"], read_only=True)
    assert list(app.rows["run"]) == ["pilot_a1", "a2_test"]
    assert "blind annotation" in app.header.value and app.image.value
    app._go(1)
    assert "a2_test" in app.header.value and "not blind" in app.header.value
    app.run_sel.value = "pilot_a1"
    assert list(app.rows["photo_id"]) == [11]
    app.run_sel.value = "all runs"
    app.view.value = "any_model_differs"
    assert list(app.rows["photo_id"]) == [11]                 # solo A1 difiere en el lado
    single = DisagreementApp(a1, ["photo_traits_v8"], read_only=True)    # una corrida: igual
    assert len(single.rows) == 1


def test_mode_of_top_prompts(tmp_path):
    """Moda de varios prompts: mayoria; empate -> unclear (conteo: mediana); metricas de la moda."""
    import json
    from sepia.disagree import mode_value, classification_metrics, count_agreement, MODE
    assert mode_value(["dorsal", "dorsal", "ventral"]) == "dorsal"
    assert mode_value(["dorsal", "ventral", "unclear"]) == "unclear"           # empate
    assert mode_value(["dorsal", "not_applicable", ""]) == "dorsal"           # vota solo uno
    assert mode_value(["dorsal", "ventral", "not_applicable"]) == "unclear"
    assert mode_value(["unclear", "unclear", "dorsal"]) == "unclear"
    assert mode_value(["1", "2", "3"], numeric=True) == ""                    # conteo: se salta
    assert mode_value(["1", "1", "2"], numeric=True) == "1"
    assert mode_value(["", "not_applicable"]) == ""
    cfg = sepia.Config(base_dir=str(tmp_path))
    p = sepia.paths(cfg)
    photos = [(1, 11, 1), (2, 21, 1)]
    with open(p["tables"] / "photos.jsonl", "w") as f:
        for o, ph, pos in photos:
            f.write(json.dumps({"observation_id": o, "photo_id": ph, "position": pos,
                                "file_name": f"{o}_{ph}.jpg"}) + "\n")
    with open(p["annotations"] / "manual_T.jsonl", "w") as f:
        for o, ph, pos in photos:
            f.write(json.dumps({"kind": "photo_info", "observation_id": o, "photo_id": ph,
                                "n_in_photo": 1, "saved_at": "2026-11-01"}) + "\n")
            f.write(json.dumps({"kind": "individual", "observation_id": o, "photo_id": ph,
                                "individual": "a", "side": "dorsal", "erosion": 1,
                                "photo_quality": "high", "saved_at": "2026-11-01"}) + "\n")
    sides = {"v8": {11: "dorsal", 21: "ventral"}, "v9": {11: "dorsal", 21: "ventral"},
             "v10": {11: "ventral", 21: "dorsal"}}
    out = p["base"] / "outputs"
    out.mkdir(exist_ok=True)
    for v, sd in sides.items():
        with open(out / f"photo_traits_gemini_gemini-3.5-flash-lite_photo_traits_{v}.jsonl", "w") as f:
            for o, ph, pos in photos:
                f.write(json.dumps({"observation_id": o, "photo_id": ph, "position": pos,
                                    "error": None, "saved_at": "2026-11-02", "prompt_sha": "x",
                                    "result": {"n_cuttlebones_visible": 1, "side": sd[ph],
                                               "erosion": "1", "photo_quality": "high"}}) + "\n")
    top3 = ["photo_traits_v8", "photo_traits_v9", "photo_traits_v10"]
    s, pc, m = classification_metrics(cfg, top3, fields=("side",), save=False, quiet=True,
                                      mode_of=top3)
    cm = m[(MODE, "side")]
    assert cm.loc["dorsal", "dorsal"] == 1 and cm.loc["dorsal", "ventral"] == 1   # moda = v8/v9
    assert s.set_index("prompt_version").loc[MODE, "accuracy"] == 0.5
    c = count_agreement(cfg, top3, quiet=True, mode_of=top3)
    assert c.loc[MODE, "agreement"] == 1.0


def test_ask_martin_editing_and_boxes(api, cfg):
    """AnnotationApp(photo_ids=...) recorre solo esas fotos; draw_boxes dibuja el recuadro."""
    pytest.importorskip("ipywidgets")
    import io
    from PIL import Image
    from sepia.disagree import ask_martin_photos, draw_boxes
    _ready(cfg)
    f = sepia.annotate.annotation_file(cfg, "HZ")
    sepia.append_jsonl({"kind": "individual", "observation_id": 1, "photo_id": 12, "individual": "a",
                        "state": "cuttlebone", "notes": "preguntar a Martín",
                        "saved_at": "2026-11-05T10:00:00+00:00"}, f)
    sepia.append_jsonl({"kind": "individual", "observation_id": 2, "photo_id": 21, "individual": "a",
                        "state": "cuttlebone", "notes": "ok", "saved_at": "2026-11-05T10:01:00+00:00"}, f)
    ids = ask_martin_photos(cfg)
    assert ids == [12]
    app = sepia.AnnotationApp(cfg, annotator="HZ", boxes=False, photo_ids=ids)
    assert app.row["photo_id"] == 12
    assert set(app.queue["observation_id"]) == {1}           # el registro completo en miniaturas
    app.on_next()
    assert app.row["photo_id"] == 12                         # no hay otra elegida: se queda
    app.on_prev()
    assert app.row["photo_id"] == 12
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), (255, 255, 255)).save(buf, "JPEG")
    out = draw_boxes(buf.getvalue(), [("a", {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.8})])
    im = Image.open(io.BytesIO(out)).convert("RGB")
    r, g, b = im.getpixel((20, 50))                           # borde izquierdo del recuadro
    assert r > 200 and g < 200 and b < 100                    # naranjo
    assert draw_boxes(buf.getvalue(), []) == buf.getvalue()


def test_excel_two_runs(api, tmp_path):
    """Un solo Excel con dos corridas: filas una debajo de otra, run y prefilled_from."""
    a1 = sepia.Config(base_dir=str(tmp_path / "proj" / "data" / "r1"), per_page=3)
    sepia.download_observations(a1)
    sepia.build_tables(a1)
    a2 = sepia.Config(base_dir=str(tmp_path / "proj" / "data" / "r2"), per_page=3,
                      skip_ids_from=["r1"])
    # r2: mismos datos de la API, pero sin los registros de r1 -> se arma a mano con 1 registro
    sepia.download_observations(a2)
    obs, pho = sepia.load_tables(a1)
    p2 = sepia.paths(a2)
    sepia.write_jsonl(obs[obs["observation_id"] == 1].assign(observation_id=101),
                      p2["tables"] / "observations.jsonl")
    sepia.write_jsonl(pho[pho["observation_id"] == 1].assign(observation_id=101,
                                                            photo_id=pho["photo_id"] + 1000),
                      p2["tables"] / "photos.jsonl")
    sepia.append_jsonl({"kind": "individual", "observation_id": 101, "photo_id": 1011,
                        "individual": "a", "side": "ventral", "prefilled_from": "assisted",
                        "saved_at": "2026-11-05T10:00:00+00:00"},
                       p2["annotations"] / "manual_HZ.jsonl")
    out = sepia.build_excel([a1, a2], "both.xlsx")
    assert out.parent == tmp_path / "proj" / "exports"
    wb, rows = _excel_rows(out)
    runs = [r["run"] for r in rows]
    assert runs == sorted(runs, key=["r1", "r2"].index)          # primero r1, despues r2
    r101 = [r for r in rows if r["ID"] == 101]
    assert len(r101) == 2 and {r["prefilled_from"] for r in r101} == {"assisted", None}
    assert [r["n° observation"] for r in rows if r["ID"] == 101][0] == 8   # correlativo total
    assert [c.value for c in wb["excluded"][1]][0] == "run"
    # un registro en las dos corridas -> error
    sepia.write_jsonl(obs[obs["observation_id"] == 1], p2["tables"] / "observations.jsonl")
    sepia.write_jsonl(pho[pho["observation_id"] == 1], p2["tables"] / "photos.jsonl")
    with pytest.raises(ValueError):
        sepia.build_excel([a1, a2], "dup.xlsx")
    assert sepia.build_excel(a1, "one.xlsx").parent == sepia.paths(a1)["exports"]



def test_excel_counts_only_cuttlebones(api, cfg):
    """'number individuals (cuttlebone)' cuenta solo letras con estado hueso."""
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    f = sepia.paths(cfg)["annotations"] / "HZ.jsonl"
    sepia.append_jsonl({"observation_id": 1, "photo_id": 11, "individual": "a", "state": "cuttlebone",
                        "saved_at": "2026-10-28T10:00"}, f)
    sepia.append_jsonl({"observation_id": 1, "photo_id": 12, "individual": "b", "state": "live animal",
                        "saved_at": "2026-10-28T10:01"}, f)
    sepia.append_jsonl({"kind": "photo_info", "observation_id": 2, "photo_id": 21, "n_in_photo": 5,
                        "prefilled_from": "assisted", "saved_at": "2026-10-28T10:02"}, f)
    _, rows = _excel_rows(sepia.build_excel(cfg, "c.xlsx"))
    r1 = [r for r in rows if r["ID"] == 1]
    assert all(r["number individuals (cuttlebone)"] == 1 for r in r1)
    r21 = [r for r in rows if r["photo ID"] == 21][0]
    assert r21["prefilled_from"] == "assisted" and r21["number individuals in photo (cuttlebone)"] == 5


# --------------------------------------------------------------------------
# Sectores A-E (configs/sectors.yaml)
# --------------------------------------------------------------------------

def test_sectors_towns():
    """Localidades conocidas caen en el sector que muestran las diapositivas de Martin."""
    from sepia.sectors import load_sectors, assign_sector, overlaps, in_polygon
    S = load_sectors()
    assert [s["name"] for s in S] == ["A", "B", "C", "D", "E"]
    site = lambda lat, lon: assign_sector(lat, lon, S)["site"]
    reason = lambda lat, lon: assign_sector(lat, lon, S)["sector_reason"]
    assert site(57.15, -2.09) == "A"             # Aberdeen
    assert reason(53.41, -2.98) == "excluded_zone:A:irish_sea"      # Liverpool
    assert reason(58.59, -3.52) == "outside_sectors"                 # Thurso (pendiente)
    assert site(52.79, 1.60) == "A"              # Sea Palling: en A y C, C lo excluye
    assert site(51.13, 1.31) == "B"              # Dover
    assert site(50.72, -1.88) == "B"             # Bournemouth (C excluye el lado ingles)
    assert site(50.95, 1.86) == "C"              # Calais (B excluye la costa francesa)
    assert site(49.64, -1.62) == "C"             # Cherbourg
    assert reason(49.51, -1.80) == "excluded_zone:C:west_cotentin"  # Les Pieux
    assert site(52.96, 4.76) == "D"              # Den Helder
    assert site(53.86, 8.69) == "E"              # Cuxhaven
    assert site(57.72, 10.59) == "E"             # Skagen
    assert reason(None, None) == "no_coordinates"
    assert {(a, b) for a, b, _ in overlaps(S)} == {("A", "C"), ("B", "C"), ("D", "E")}
    sq = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    assert in_polygon(0.5, 0.5, sq) and not in_polygon(1.5, 0.5, sq)


def test_sector_config_and_excel(api, cfg, tmp_path):
    """sector: en el YAML da el rectangulo; la descarga no filtra; el sector se asigna en el Excel."""
    y = tmp_path / "c.yaml"
    y.write_text(f"base_dir: {tmp_path / 'run'}\nsector: B\n")
    assert sepia.load_config(str(y)).bbox == sepia.sectors.get_sector("B")["bbox"]
    sepia.download_observations(cfg)
    obs, pho = sepia.build_tables(cfg)
    assert len(obs) == 7 and len(pho) == 14 and "site" not in obs     # nada se filtra al bajar
    raw = inat.load_raw(cfg)
    raw[1]["geojson"]["coordinates"] = [-2.98, 53.41]                   # registro 2 en Liverpool
    (sepia.paths(cfg)["raw"] / "page_0001.json").write_text(json.dumps({"results": raw}))
    sepia.build_tables(cfg)
    wb, rows = _excel_rows(sepia.build_excel(cfg, "s.xlsx"))
    assert {r["site"] for r in rows} == {"A"} and not any(r["ID"] == 2 for r in rows)
    ex = list(wb["excluded"].iter_rows(min_row=2, values_only=True))
    assert [(e[1], e[-1]) for e in ex] == [(2, "excluded_zone:A:irish_sea")]


# --------------------------------------------------------------------------
# Recuadro del hueso (bonebox_v1) y AnnotationApp
# --------------------------------------------------------------------------

class FakeBoneBox:
    name = "fake"
    def __init__(self):
        self.calls = 0
    def classify(self, images, prompt, schema=None):
        assert schema is sepia.bonebox.BoneBoxResult and len(images) == 1
        self.calls += 1
        res = {"one_cuttlebone": True, "box_2d": [100, 200, 600, 900], "notes": ""}
        return res, json.dumps(res), {"input_tokens": 300, "output_tokens": 20}


def test_bonebox_conversion():
    from sepia.bonebox import box2d_to_crop, apply_rules
    assert box2d_to_crop([100, 200, 600, 900]) == {"x0": 0.2, "y0": 0.1, "x1": 0.9, "y1": 0.6}
    assert box2d_to_crop([600, 900, 100, 200]) == {"x0": 0.2, "y0": 0.1, "x1": 0.9, "y1": 0.6}
    assert box2d_to_crop([0, 0, 0, 0]) is None and box2d_to_crop([1, 2]) is None
    assert box2d_to_crop([-50, 0, 1200, 500])["y1"] == 1.0          # recorta a la foto
    r, _ = apply_rules({"one_cuttlebone": False, "box_2d": [1, 2, 300, 400], "notes": ""})
    assert r["crop_box"] is None
    assert "box_2d" in sepia.load_prompt("bonebox_v1")[0]


def test_bonebox_run_and_annotation(api, cfg, monkeypatch):
    """bonebox escribe su archivo; AnnotationApp parte con el recuadro y guarda box_source."""
    pytest.importorskip("jupyter_bbox_widget")
    from sepia import annotate as an, bonebox as bb
    assert an.box_source(None, None) is None
    box = {"x0": 0.2, "y0": 0.1, "x1": 0.9, "y1": 0.6}
    assert an.box_source(box, None) == "manual" and an.box_source(box, box) == "model"
    assert an.box_source({**box, "x1": 0.95}, box) == "model_edited"

    _ready(cfg)
    monkeypatch.setattr(bb.time, "sleep", lambda s: None)
    prov = FakeBoneBox()
    with pytest.raises(RuntimeError):                     # sin photo_traits_v14 corrido
        bb.run_bonebox(cfg, provider=prov)
    bb.run_bonebox(cfg, provider=prov, photo_ids=[11, 12])
    bb.run_bonebox(cfg, provider=prov, photo_ids=[11, 12])   # reanuda: nada nuevo
    assert prov.calls == 2
    assert bb.model_boxes(cfg) == {11: box, 12: box}

    blind = sepia.AnnotationApp(cfg, annotator="HZ")            # por defecto: sin modelos
    assert blind.model_boxes == {} and blind.crop.value == ""
    app = sepia.AnnotationApp(cfg, annotator="HZ", model_boxes="bonebox_v1")
    assert app.row["photo_id"] == 11 and app.crop.value == an.format_crop_box(box)
    assert len(app.bbox.bboxes) == 1 and "model" in app.box_note.value
    app._safe(app.on_save_next)                            # aceptado tal cual
    assert app.row["photo_id"] == 12
    app.crop.value = "0.1, 0.1, 0.95, 0.6"                 # ajustado
    app._safe(app.on_save)
    ind = sepia.load_events(cfg)["individual"].set_index("photo_id")
    assert ind.loc[11, "box_source"] == "model" and ind.loc[11, "crop_box"] == box
    assert ind.loc[12, "box_source"] == "model_edited" and ind.loc[12, "model_box"] == box

    # Fotos anotadas antes de los recuadros: lista para pasarlas con el recuadro del modelo
    an.save_individual(cfg, "HZ", 2, 21, "a", {"state": "cuttlebone"})
    bb.run_bonebox(cfg, provider=prov, photo_ids=[21])
    assert an.photos_without_box(cfg) == [21]
    app = sepia.AnnotationApp(cfg, annotator="HZ", photo_ids=an.photos_without_box(cfg),
                              model_boxes="bonebox_v1")
    assert app.row["photo_id"] == 21 and app.crop.value == an.format_crop_box(box)
    app._safe(app.on_save)
    assert an.photos_without_box(cfg) == []


def test_bonebox_iou_and_zai_json():
    from sepia.bonebox import iou
    from sepia.triage import extract_json
    a = {"x0": 0, "y0": 0, "x1": 0.5, "y1": 0.5}
    assert iou(a, a) == 1 and iou(a, {"x0": 0.5, "y0": 0.5, "x1": 1, "y1": 1}) == 0
    assert abs(iou(a, {"x0": 0.25, "y0": 0, "x1": 0.75, "y1": 0.5}) - 1 / 3) < 1e-9
    txt = '<think>hmm</think>Here:\n```json\n{"one_cuttlebone": true, "box_2d": [1, 2, 3, 4], "notes": ""}\n```'
    r = sepia.bonebox.BoneBoxResult.model_validate_json(extract_json(txt))
    assert r.box_2d == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        sepia.triage.make_provider("otro", "m")


def test_zai_provider(monkeypatch):
    """GLM por la API de Z.ai: fotos como data URL, esquema al final del prompt, respuesta validada."""
    import requests
    sent = {}

    class Resp:
        status_code = 200
        text = ""
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": '```json\n{"one_cuttlebone": true, '
                                             '"box_2d": [10, 20, 30, 40], "notes": ""}\n```'}}],
                    "usage": {"prompt_tokens": 900, "completion_tokens": 30}}

    def fake_post(self, url, json=None, timeout=None):
        sent.update(url=url, body=json)
        return Resp()

    monkeypatch.setenv("ZAI_API_KEY", "k")
    monkeypatch.setattr(requests.Session, "post", fake_post)
    prov = sepia.triage.make_provider("zai", "glm-4.6v-flash", thinking_level=None)
    res, raw, usage = prov.classify([b"\xff\xd8jpg"], "find the bone",
                                    schema=sepia.bonebox.BoneBoxResult)
    assert res["box_2d"] == [10, 20, 30, 40] and usage["input_tokens"] == 900
    content = sent["body"]["messages"][0]["content"]
    assert sent["url"].endswith("/chat/completions") and sent["body"]["thinking"]["type"] == "disabled"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "box_2d" in content[-1]["text"] and content[-1]["text"].startswith("find the bone")


# --------------------------------------------------------------------------
# v15: triaje + rasgos + recuadro, varios registros por consulta
# --------------------------------------------------------------------------

class FakePack:
    """Responde cada registro de la consulta. broken = etiqueta de un registro que vuelve sin
    una de sus fotos (la primera vez): se repite solo."""
    name = "fake"
    def __init__(self, broken=None):
        self.calls, self.packs, self.broken = 0, [], broken
    def classify(self, images, prompt, schema=None, labels=None):
        from sepia.assist import PackV15
        assert schema is PackV15 and len(labels) == len(images)
        self.calls += 1
        recs = {}
        for lab in labels:
            pl = lab.splitlines()[-1].rstrip(":")
            recs.setdefault(pl.split("-")[0], []).append(pl)
        self.packs.append(len(recs))
        photo = lambda pl, i: {
            "photo_label": pl, "content": "cuttlebone", "n_cuttlebones_visible": 1,
            "box_2d": [100, 200, 600, 900], "side_evidence": "U-shaped rim",
            "side": "ventral" if i == 0 else "dorsal", "side_basis": "direct", "erosion": "1",
            "quality_evidence": "nothing covers the bone", "cover": "none", "focus": "sharp",
            "notes": ""}
        out = []
        for rl, pls in recs.items():
            if rl == self.broken and len(recs) > 1:
                pls = pls[:1]                               # falta una foto
            out.append({"record_label": rl, "photos": [photo(pl, i) for i, pl in enumerate(pls)],
                        "record_state": "cuttlebone", "n_cuttlebones": 1, "count_possible": True,
                        "same_bone_in_several_photos": True, "fragments_present": False,
                        "confidence": "high", "record_notes": ""})
        res = PackV15.model_validate({"records": out}).model_dump()
        return res, json.dumps(res), {"input_tokens": 9000, "output_tokens": 2000}


def test_assist_packs(api, cfg, monkeypatch):
    from sepia import assist as asx
    packs = asx.make_packs(list(range(1, 8)), 3)
    assert [sorted(p) for p in packs] == [[1, 2, 3], [4, 5, 6], [7]]
    assert asx.make_packs(list(range(1, 8)), 3) == packs                 # misma semilla
    _ready(cfg)
    monkeypatch.setattr(asx.time, "sleep", lambda s: None)
    prov = FakePack(broken="R2")
    stats = asx.run_assist(cfg, provider=prov, pack_size=5, suffix="_t")
    # 7 registros: paquetes de 5 y 2; en cada uno R2 falla y se repite solo
    assert stats["records_ok"] == 7 and stats["records_retried"] == 2 and prov.calls == 4
    assert asx.run_assist(cfg, provider=prov, pack_size=5, suffix="_t")["calls"] == 0  # reanuda
    pt = sepia.load_photo_traits(cfg, "photo_traits_v15_t")
    assert len(pt) == 14 and set(pt["photo_quality"]) == {"high"}        # regla de calidad v14
    assert set(pt.loc[pt["position"] == 1, "side"]) == {"ventral"}
    tri = sepia.triage.load_triage(cfg, "gemini", "gemini-3.5-flash-lite", "triage_v15_t")
    assert len(tri) == 7 and set(tri["n_cuttlebones"]) == {1}
    assert len(tri.iloc[0]["photos"]) == 2
    boxes = sepia.bonebox.model_boxes(cfg, "bonebox_v15_t")
    assert len(boxes) == 14 and boxes[11] == {"x0": 0.2, "y0": 0.1, "x1": 0.9, "y1": 0.6}
    raw = sepia.read_jsonl(sepia.io.model_output_path(cfg, "assist", "gemini", "gemini-3.5-flash-lite",
                                                      "assist_v15_t"))
    assert len(raw) == 4 and raw["input_tokens"].sum() == 36000
    assert "NEW RECORD" in sepia.load_prompt("assist_v15")[0]


def test_assist_no_box_when_not_one():
    """Con 0 o varios huesos, v15 va sin recuadro (null) y no se guarda recuadro."""
    from sepia.assist import PhotoV15
    ph = PhotoV15.model_validate({"photo_label": "R1-P1", "content": "cuttlebone",
                                  "n_cuttlebones_visible": 3, "side_evidence": "not_applicable",
                                  "side": "not_applicable", "side_basis": "not_applicable",
                                  "erosion": "not_applicable", "quality_evidence": "not_applicable",
                                  "cover": "not_applicable", "focus": "not_applicable", "notes": ""})
    assert ph.box_2d is None


def test_assisted_draws_model_box(api, cfg, monkeypatch):
    """AssistedApp(model_boxes=...) dibuja el recuadro del modelo sobre la foto (solo para ver)."""
    pytest.importorskip("ipywidgets")
    from sepia import assist as asx
    _ready(cfg)
    sepia.make_resized(cfg, 512)
    monkeypatch.setattr(asx.time, "sleep", lambda s: None)
    asx.run_assist(cfg, provider=FakePack(), pack_size=5, suffix="")
    from sepia.assisted import AssistedApp
    plain = AssistedApp(cfg, ["photo_traits_v15"])
    app = AssistedApp(cfg, ["photo_traits_v15"], model_boxes="bonebox_v15")
    assert len(app.model_boxes) == 14
    if app.bbox is not None:                                          # con el widget: editable
        assert len(app.bbox.bboxes) == 1
        app.model_boxes = dict(app.model_boxes)
    # sin el widget (o para comprobar el dibujo): recuadro pintado en rojo sobre la foto
    data = app._with_box(plain.image.value, int(app.rows.loc[app.i, "photo_id"]))
    assert data != plain.image.value
    im = Image.open(io.BytesIO(data)).convert("RGB")
    W, H = im.size
    r, g, b = im.getpixel((int(0.2 * W) + 1, int(0.35 * H)))           # borde izquierdo del recuadro
    assert r > 150 and g < 80


def test_assisted_shows_manual(api, cfg, monkeypatch):
    """La asistida muestra al lado la anotacion manual de la foto, si existe."""
    pytest.importorskip("ipywidgets")
    from sepia import assist as asx, annotate as an
    _ready(cfg)
    sepia.make_resized(cfg, 512)
    monkeypatch.setattr(asx.time, "sleep", lambda s: None)
    asx.run_assist(cfg, provider=FakePack(), pack_size=5, suffix="")
    an.save_individual(cfg, "HZ", 1, 11, "a", {"state": "cuttlebone", "side": "dorsal", "erosion": 2})
    from sepia.assisted import AssistedApp
    app = AssistedApp(cfg, ["photo_traits_v15"])
    assert app.rows.loc[app.i, "photo_id"] == 11
    assert app._manual_values(1, 11) == {"n_cuttlebones_visible": 1, "side": "dorsal", "erosion": 2}
    assert app._manual_values(1, 12) == {}
    html_all = " ".join(getattr(c, "value", "") for row in app.grid.children
                        for c in getattr(row, "children", []))
    assert "manual" in html_all and "dorsal" in html_all and "final" in html_all


def test_assisted_redraw_box_goes_to_annotation(api, cfg, monkeypatch):
    """Recuadro rehecho en la asistida -> se guarda -> AnnotationApp(prefill) parte con ese."""
    pytest.importorskip("jupyter_bbox_widget")
    from sepia import assist as asx, annotate as an
    from sepia.assisted import AssistedApp, load_assisted
    _ready(cfg)
    sepia.make_resized(cfg, 512)
    monkeypatch.setattr(asx.time, "sleep", lambda s: None)
    asx.run_assist(cfg, provider=FakePack(), pack_size=5, suffix="")
    app = AssistedApp(cfg, ["photo_traits_v15"], model_boxes="bonebox_v15")
    assert app.bbox is not None and len(app.bbox.bboxes) == 1         # recuadro del modelo
    W, H = app._box_size
    app.redraw_btn.value = True
    app.bbox.bboxes = [{"x": 0, "y": 0, "width": W / 2, "height": H / 2, "label": "bone"}]
    assert app.box_value == {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5}
    assert "redrawn" in app.box_info.value
    app._save(go_next=True)                                           # foto 11
    app._set_box(None)                                                # foto 12: sin recuadro
    app._save()
    a = load_assisted(cfg, "HZ").set_index("photo_id")
    assert a.loc[11, "box_source"] == "model_edited" and a.loc[11, "crop_box"]["x1"] == 0.5
    assert a.loc[12, "box_reviewed"] and a.loc[12, "box_source"] != a.loc[12, "box_source"]  # NaN

    ann = sepia.AnnotationApp(cfg, "HZ", prefill="assisted", model_boxes="bonebox_v15")
    assert ann.row["photo_id"] == 11 and ann.crop.value.endswith("0.5000")
    assert "assisted" in ann.box_note.value
    ann._safe(ann.on_save_next)
    assert ann.row["photo_id"] == 12 and ann.crop.value == ""        # quitado en la asistida
    ind = sepia.load_events(cfg)["individual"].set_index("photo_id")
    assert ind.loc[11, "box_source"] == "model_edited"


def test_assisted_box_after_old_save(api, cfg, monkeypatch):
    """Foto guardada en la asistida ANTES de los recuadros: igual muestra el recuadro del modelo."""
    pytest.importorskip("jupyter_bbox_widget")
    from sepia import assist as asx
    from sepia.assisted import AssistedApp, assisted_file
    _ready(cfg)
    sepia.make_resized(cfg, 512)
    monkeypatch.setattr(asx.time, "sleep", lambda s: None)
    asx.run_assist(cfg, provider=FakePack(), pack_size=5, suffix="")
    AssistedApp(cfg, ["photo_traits_v15"])._save()                  # guardada sin recuadros
    app = AssistedApp(cfg, ["photo_traits_v15"], model_boxes="bonebox_v15")
    app._set_box({"x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.4})       # otra foto con recuadro guardado
    app._go(1); app._save(); app._go(0)
    assert app.saved is not None and app.box_value == app.box_model and len(app.bbox.bboxes) == 1


def test_assisted_resumes_at_pending(api, cfg, monkeypatch):
    """La asistida abre en la primera foto pendiente (ni guardada aqui ni anotada a mano)."""
    pytest.importorskip("ipywidgets")
    from sepia import assist as asx, annotate as an
    from sepia.assisted import AssistedApp
    _ready(cfg)
    sepia.make_resized(cfg, 512)
    monkeypatch.setattr(asx.time, "sleep", lambda s: None)
    asx.run_assist(cfg, provider=FakePack(), pack_size=5, suffix="")
    app = AssistedApp(cfg, ["photo_traits_v15"])
    assert app.i == 0
    app._save(go_next=True)                                          # foto 1 guardada
    an.save_individual(cfg, "HZ", 1, 12, "a", {"state": "cuttlebone"})  # foto 2 anotada a mano
    again = AssistedApp(cfg, ["photo_traits_v15"])
    assert again.i == 2 and again._next_pending(3) == 3
