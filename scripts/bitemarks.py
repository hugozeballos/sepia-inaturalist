"""
Marcas de picoteo por foto (zero-shot), paso aparte de photo_traits. Usa la seccion
'bitemarks' del YAML (o, si no existe, proveedor y modelo de 'photo_traits').

Uso (en Colab, con GEMINI_API_KEY en el entorno):
    !python scripts/bitemarks.py --config configs/pilot_a1.yaml --limit 5      # prueba
    !python scripts/bitemarks.py --config configs/pilot_a1.yaml --annotated    # solo fotos anotadas
    !python scripts/bitemarks.py --config configs/pilot_a1.yaml                # todas las fotos
    !python scripts/bitemarks.py --config configs/pilot_a1.yaml --evaluate     # vs anotacion manual

Una linea por foto en outputs/bitemarks_<proveedor>_<modelo>_<version>.jsonl. Reanudable.
--evaluate agrega una linea a outputs/metrics/bitemarks_metrics.jsonl.
"""
import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

import sepia

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=None, help="solo N fotos")
ap.add_argument("--ids", default=None, help="solo fotos de estos registros: IDs separados por comas")
ap.add_argument("--annotated", action="store_true",
                help="solo fotos con un unico individuo anotado (las que se pueden evaluar)")
ap.add_argument("--redo", action="store_true", help="rehacer aunque ya esten hechas (agrega, no borra)")
ap.add_argument("--evaluate", action="store_true", help="comparar con la anotacion manual y salir")
ap.add_argument("--prompt-version", default=None, help="p. ej. bitemarks_v1")
args = ap.parse_args()

cfg = sepia.load_config(args.config)
if args.evaluate:
    sepia.evaluate_bitemarks(cfg, prompt_version=args.prompt_version)
    sys.exit(0)
ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
photo_ids = None
if args.annotated:
    ind = sepia.load_events(cfg)["individual"]
    photo_ids = list(ind.groupby("photo_id").filter(lambda g: len(g) == 1)["photo_id"].astype(int).unique())
    print(f"{len(photo_ids)} fotos con un unico individuo anotado")
sepia.run_bitemarks(cfg, limit=args.limit, ids=ids, redo=args.redo,
                    prompt_version=args.prompt_version, photo_ids=photo_ids)
