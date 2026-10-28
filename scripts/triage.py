"""
Triaje por registro con un modelo de vision (estado y numero de huesos).

Uso (en Colab, con la clave en los secretos):
    import os
    from google.colab import userdata
    os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
    !python scripts/triage.py --config configs/pilot_a1.yaml --limit 5   # prueba
    !python scripts/triage.py --config configs/pilot_a1.yaml             # todo

Reanudable: los registros ya clasificados se saltan.
"""
import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)  # en Colab, mostrar el avance al instante

import sepia

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=None, help="clasificar solo N registros (prueba)")
args = ap.parse_args()

cfg = sepia.load_config(args.config)
if not cfg.triage:
    ap.error("la configuracion no tiene seccion 'triage'")
sepia.run_triage(cfg, limit=args.limit)
