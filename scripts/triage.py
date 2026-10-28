"""
Triaje por registro con un modelo de vision (estado y numero de huesos).

Uso (en Colab, con la clave en los secretos):
    import os
    from google.colab import userdata
    os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
    !python scripts/triage.py --config configs/pilot_a1.yaml --limit 5   # prueba
    !python scripts/triage.py --config configs/pilot_a1.yaml             # todo
    !python scripts/triage.py --config configs/pilot_a1.yaml --ids configs/triage_hard_cases.jsonl

Reanudable: los registros ya clasificados con el mismo prompt se saltan.
Con --redo se vuelven a clasificar (se agregan lineas nuevas; vale la ultima).
"""
import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)  # en Colab, mostrar el avance al instante

import sepia

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=None, help="clasificar solo N registros (prueba)")
ap.add_argument("--redo", action="store_true",
                help="volver a clasificar aunque ya esten hechos con este prompt (agrega, no borra)")
ap.add_argument("--ids", default=None,
                help="clasificar solo estos IDs: lista separada por comas, .txt (uno por linea) o .jsonl de casos")
args = ap.parse_args()

cfg = sepia.load_config(args.config)
if not cfg.triage:
    ap.error("la configuracion no tiene seccion 'triage'")
ids = None
if args.ids:
    if args.ids.endswith(".txt"):
        if not os.path.exists(args.ids):
            ap.error(f"no existe el archivo {args.ids}")
        ids = sepia.ids_from_file(args.ids)
    elif args.ids.endswith(".jsonl"):
        ids = [int(i) for i in sepia.triage.load_cases(args.ids)["observation_id"]]
    else:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
sepia.run_triage(cfg, limit=args.limit, ids=ids, redo=args.redo)
