"""
v15: triaje + rasgos por foto + recuadro en UNA consulta, con varios registros por consulta
(ver src/sepia/assist.py). Usa la seccion 'assist' del YAML (o proveedor y modelo de 'photo_traits').

Uso (en Colab, con GEMINI_API_KEY en el entorno):
    !python scripts/assist.py --config configs/pilot_a1.yaml --limit 1                 # 1 consulta de prueba
    !python scripts/assist.py --config configs/sector_b.yaml                           # todo, 5 registros por consulta
    !python scripts/assist.py --config configs/pilot_a1.yaml --pack-size 1 --suffix _p1   # prueba de a 1

Escribe triage_v15, photo_traits_v15 y bonebox_v15 (+ sufijo) en outputs/, que las interfaces leen
como cualquier otra version. Reanudable.
"""
import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

import sepia

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--pack-size", type=int, default=None, help="registros por consulta (por defecto 5)")
ap.add_argument("--suffix", default="", help="etiqueta de prueba en las versiones, p. ej. _p5")
ap.add_argument("--ids", default=None, help="solo estos registros: IDs separados por comas")
ap.add_argument("--limit", type=int, default=None, help="solo N consultas")
ap.add_argument("--redo", action="store_true", help="rehacer aunque ya esten hechos (agrega, no borra)")
args = ap.parse_args()

cfg = sepia.load_config(args.config)
ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
sepia.run_assist(cfg, pack_size=args.pack_size, suffix=args.suffix, ids=ids, limit=args.limit,
                 redo=args.redo)
