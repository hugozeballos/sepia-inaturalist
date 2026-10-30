"""
Separa los archivos antiguos de salidas de modelos (todas las versiones de prompt juntas)
en un archivo por version de prompt. Se corre UNA vez por corrida de datos.

    !python scripts/split_outputs.py --config configs/pilot_a1.yaml

No borra nada: copia cada linea al archivo de su version y renombra el archivo antiguo
a *.jsonl.legacy (se conserva como respaldo, pero el codigo ya no lo lee).
Sin correr este script todo sigue funcionando: el codigo lee tambien el archivo antiguo.
"""
import argparse

import sepia
from sepia.io import split_legacy_outputs

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()

cfg = sepia.load_config(args.config)
for kind, section in (("triage", cfg.triage), ("photo_traits", cfg.photo_traits)):
    if not section:
        continue
    provider, model = section.get("provider", "gemini"), section["model"]
    counts = split_legacy_outputs(cfg, kind, provider, model)
    if counts:
        print(f"{kind} {provider}/{model}: " + ", ".join(f"{v}: {n} lineas" for v, n in sorted(counts.items())))
    else:
        print(f"{kind} {provider}/{model}: no hay archivo antiguo")
