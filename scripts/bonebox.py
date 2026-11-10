"""
Recuadro del hueso por foto (bonebox_v1), paso aparte de photo_traits. Solo fotos con 1 hueso
segun la version de photo_traits de 'select_from' (seccion 'bonebox' del YAML; por defecto
photo_traits_v14, que tiene que estar corrida antes).

Uso (en Colab, con GEMINI_API_KEY en el entorno):
    !python scripts/bonebox.py --config configs/pilot_a1.yaml --limit 5    # prueba
    !python scripts/bonebox.py --config configs/pilot_a1.yaml              # todas las fotos con 1 hueso

Una linea por foto en outputs/bonebox_<proveedor>_<modelo>_<version>.jsonl. Reanudable.
AnnotationApp muestra ese recuadro ya dibujado: se acepta, se ajusta o se dibuja otro.
"""
import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

import sepia

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=None, help="solo N fotos")
ap.add_argument("--ids", default=None, help="solo fotos de estos registros: IDs separados por comas")
ap.add_argument("--redo", action="store_true", help="rehacer aunque ya esten hechas (agrega, no borra)")
ap.add_argument("--prompt-version", default=None, help="p. ej. bonebox_v1")
args = ap.parse_args()

cfg = sepia.load_config(args.config)
ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
sepia.run_bonebox(cfg, limit=args.limit, ids=ids, redo=args.redo,
                  prompt_version=args.prompt_version)
