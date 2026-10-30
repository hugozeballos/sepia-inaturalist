"""
Rasgos por foto con un modelo de vision: huesos visibles, lado y erosion (sin ejemplos).
Paso aparte del triaje: no cambia el conteo por registro. Usa la seccion 'photo_traits' del YAML.

Uso (en Colab, con GEMINI_API_KEY en el entorno):
    !python scripts/photo_traits.py --config configs/pilot_a1.yaml --limit 5     # prueba
    !python scripts/photo_traits.py --config configs/pilot_a1.yaml               # todas las fotos
    !python scripts/photo_traits.py --config configs/pilot_a1.yaml --evaluate    # vs anotacion manual
    !python scripts/photo_traits.py --config configs/pilot_a1.yaml --prompt-version photo_traits_v5

Cada version de prompt escribe en su propio archivo (outputs/photo_traits_..._<version>.jsonl).
Reanudable: las fotos ya hechas con el mismo prompt se saltan. --redo las rehace (agrega lineas).
"""
import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

import sepia

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=None, help="solo N fotos (o N registros con prompts por registro, v5-)")
ap.add_argument("--ids", default=None, help="solo estos registros: IDs separados por comas")
ap.add_argument("--redo", action="store_true", help="rehacer aunque ya esten hechas (agrega, no borra)")
ap.add_argument("--evaluate", action="store_true", help="comparar con la anotacion manual y salir")
ap.add_argument("--prompt-version", default=None,
                help="usar esta version de prompt en vez de la del YAML (p. ej. photo_traits_v5)")
args = ap.parse_args()

cfg = sepia.load_config(args.config)
if args.evaluate:
    sepia.evaluate_photo_traits(cfg, prompt_version=args.prompt_version)
    sys.exit(0)
if not cfg.photo_traits:
    ap.error("la configuracion no tiene seccion 'photo_traits'")
ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
sepia.run_photo_traits(cfg, limit=args.limit, ids=ids, redo=args.redo,
                       prompt_version=args.prompt_version)
