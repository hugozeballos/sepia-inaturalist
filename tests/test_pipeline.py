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


def test_excel(api, cfg):
    sepia.download_observations(cfg)
    sepia.build_tables(cfg)
    ws = openpyxl.load_workbook(sepia.build_excel(cfg, "a.xlsx"))["identification"]
    assert ws.max_row == 8 and ws.cell(2, 4).value == LAT
    assert not any(isinstance(c.value, str) and c.value.startswith("=")
                   for row in ws.iter_rows() for c in row)

    f = sepia.paths(cfg)["annotations"] / "HZ.jsonl"
    sepia.append_jsonl({"observation_id": 1, "individual": "a", "side": "dorsal",
                        "epibionts": "yes", "saved_at": "2026-10-28T10:00"}, f)
    sepia.append_jsonl({"observation_id": 1, "individual": "a", "side": "ventral",
                        "bitemarks": "no", "saved_at": "2026-10-28T10:01"}, f)
    sepia.append_jsonl({"observation_id": 1, "individual": "a", "side": "ventral",
                        "bitemarks": "yes", "saved_at": "2026-10-28T10:05"}, f)   # correccion
    wb = openpyxl.load_workbook(sepia.build_excel(cfg, "b.xlsx"))
    ws = wb["identification"]
    hdr = [c.value for c in ws[1]]
    rows = [dict(zip(hdr, [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
    r1 = [r for r in rows if r["ID"] == 1]
    assert len(rows) == 8 and len(r1) == 2
    ventral = [r for r in r1 if r["ventral side (cuttlebone)"] == 1]
    assert ventral[0]["¿bitemarks?"] == "yes"                    # vale la ultima
    assert all(r["latitude"] == LAT for r in r1)                  # coordenadas repetidas por join
    assert wb.sheetnames == ["identification", "legend", "excluded"]


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
    raw = sepia.read_jsonl(sepia.triage.output_path(cfg, "fake", "m1"))
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
    assert sepia.triage.done_ids(sepia.triage.output_path(cfg, "fake", "gone")) == set()


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
    raw = sepia.read_jsonl(sepia.triage.output_path(cfg, "fake", "rd"))
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
    sepia.append_jsonl({"observation_id": 2, "individual": "a", "side": "dorsal",
                        "state": "egg case", "label_source": "annotator",
                        "saved_at": "2026-10-28T10:00"}, f)
    ws = openpyxl.load_workbook(sepia.build_excel(cfg, "t.xlsx"))["identification"]
    hdr = [c.value for c in ws[1]]
    rows = {r[1]: dict(zip(hdr, r)) for r in ws.iter_rows(min_row=2, values_only=True)}
    r1, r2 = rows[1], rows[2]
    assert r1["state"] == "cuttlebone" and r1["number individuals (cuttlebone)"] == 1
    assert r1["label_source"] == "gpt" and r1["model_version"] == "m1/triage_v2"
    assert json.loads(r1["model_prob"])["confidence"] == "high"
    assert r1["reviewed"] is False
    assert r2["state"] == "egg case" and r2["label_source"] == "annotator"   # gana la persona
