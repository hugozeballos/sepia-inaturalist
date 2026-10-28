"""
Genera el Excel de entrega con el formato de Martin, en cualquier momento.

Uso:
    python scripts/build_excel.py --config configs/pilot79.yaml
"""
import argparse

import sepia

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--name", default=None, help="nombre del archivo (por defecto, con fecha y hora)")
args = ap.parse_args()
sepia.build_excel(sepia.load_config(args.config), out_name=args.name)
