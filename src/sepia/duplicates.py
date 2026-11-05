"""
Posibles registros duplicados: el mismo hueso subido dos veces a iNaturalist (por el mismo
usuario o por usuarios distintos, como 409584581 y 405594162 en la guia de Martin).

Criterio (Martin, 28 oct): mismo dia y lugar cercano. Primero se agrupa por dia y solo
despues se calculan distancias, dentro de cada dia (mucho mas rapido que todos contra todos).
Solo propone candidatos: decidir si es el mismo hueso lo hace una persona mirando las fotos.

Uso (Colab):
    from sepia.duplicates import load_runs, candidate_pairs, candidate_groups
    obs = load_runs(cfg, ["pilot_a1", "a2_test"])     # corridas = carpetas en data/
    pairs = candidate_pairs(obs, max_km=2)
    groups = candidate_groups(pairs)
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import pandas as pd

from .config import Config
from .io import read_jsonl

COLS = ["observation_id", "observed_on", "time_observed_at", "latitude", "longitude",
        "user_login", "place_guess", "n_photos", "url", "coordinates_obscured",
        "positional_accuracy"]


def load_runs(cfg: Config, runs: list[str]) -> pd.DataFrame:
    """
    Registros de varias corridas juntas, con la columna 'run'. Cada corrida es una carpeta
    junto a cfg.base_dir (p. ej. 'pilot_a1') o una ruta completa. Si un registro aparece en
    dos corridas, queda una sola vez (la primera de la lista).
    """
    parts = []
    for name in runs:
        d = Path(name) if Path(name).is_absolute() else Path(cfg.base_dir).parent / name
        f = d / "tables" / "observations.jsonl"
        if not f.exists():
            raise FileNotFoundError(f"no existe {f}")
        o = read_jsonl(f)
        parts.append(o[[c for c in COLS if c in o]].assign(run=name))
    obs = pd.concat(parts, ignore_index=True).drop_duplicates("observation_id", keep="first")
    return obs.reset_index(drop=True)


def _km(lat1, lon1, lat2, lon2) -> float:
    """Distancia en km sobre la esfera (haversine)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def candidate_pairs(obs: pd.DataFrame, max_km: float | None = 2.0) -> pd.DataFrame:
    """
    Pares de registros del MISMO dia, con la distancia entre ellos, ordenados de mas cerca a
    mas lejos. max_km: deja solo los pares a esa distancia o menos (None = todos los del dia).
    Columnas utiles:
      distance_km     distancia entre las coordenadas publicas
      minutes_apart   diferencia de hora (vacio si alguno no tiene hora)
      same_user       mismo usuario (puede ser el mismo hueso subido dos veces o huesos distintos
                      del mismo paseo)
      obscured_any    alguno tiene coordenadas ocultas: la distancia no es confiable
    """
    o = obs.dropna(subset=["observed_on", "latitude", "longitude"]).copy()
    o["day"] = pd.to_datetime(o["observed_on"], errors="coerce").dt.date
    o["t"] = pd.to_datetime(o.get("time_observed_at"), errors="coerce", utc=True)
    rows = []
    for day, g in o.groupby("day"):                    # 1. filtrar: mismo dia
        if len(g) < 2:
            continue
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):   # 2. operar
            d = _km(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
            if max_km is not None and d > max_km:
                continue
            mins = (abs((a["t"] - b["t"]).total_seconds()) / 60
                    if pd.notna(a["t"]) and pd.notna(b["t"]) else None)
            rows.append({
                "observed_on": day, "id_a": int(a["observation_id"]),
                "id_b": int(b["observation_id"]), "distance_km": round(d, 3),
                "minutes_apart": None if mins is None else round(mins, 1),
                "same_user": a.get("user_login") == b.get("user_login"),
                "user_a": a.get("user_login"), "user_b": b.get("user_login"),
                "n_photos_a": a.get("n_photos"), "n_photos_b": b.get("n_photos"),
                "obscured_any": bool(a.get("coordinates_obscured") or b.get("coordinates_obscured")),
                "run_a": a["run"], "run_b": b["run"],
                "place_a": a.get("place_guess"), "place_b": b.get("place_guess"),
                "url_a": a.get("url"), "url_b": b.get("url"),
            })
    cols = ["observed_on", "id_a", "id_b", "distance_km", "minutes_apart", "same_user",
            "user_a", "user_b", "n_photos_a", "n_photos_b", "obscured_any", "run_a", "run_b",
            "place_a", "place_b", "url_a", "url_b"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["distance_km", "minutes_apart"], na_position="last").reset_index(drop=True)


def candidate_groups(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Une los pares en grupos (pares, trios, ...): si A esta cerca de B y B de C, el grupo es
    A-B-C. Una fila por grupo: dia, registros, tamano, distancia maxima entre pares del grupo.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["id_a"], pairs["id_b"]):
        parent[find(a)] = find(b)
    if not parent:
        return pd.DataFrame(columns=["group", "observed_on", "size", "ids", "max_km",
                                     "n_users", "any_obscured"])
    members = {}
    for x in parent:
        members.setdefault(find(x), []).append(x)
    rows = []
    for i, ids in enumerate(sorted(members.values(), key=lambda v: (-len(v), min(v))), 1):
        sub = pairs[pairs["id_a"].isin(ids) & pairs["id_b"].isin(ids)]
        users = set(sub["user_a"]) | set(sub["user_b"])
        rows.append({"group": i, "observed_on": sub["observed_on"].iloc[0], "size": len(ids),
                     "ids": sorted(ids), "max_km": sub["distance_km"].max(),
                     "n_users": len(users), "any_obscured": bool(sub["obscured_any"].any())})
    return pd.DataFrame(rows)
