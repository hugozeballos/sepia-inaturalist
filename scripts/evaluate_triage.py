"""
Compara el triaje con los casos de prueba de respuesta conocida.

Uso:
    !python scripts/evaluate_triage.py --config configs/pilot_a1.yaml --cases configs/triage_hard_cases.jsonl
"""
import argparse
import sys

import sepia

sys.stdout.reconfigure(line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--cases", default="configs/triage_hard_cases.jsonl")
args = ap.parse_args()

cfg = sepia.load_config(args.config)
res = sepia.triage.evaluate_triage(cfg, cfg.triage["provider"], cfg.triage["model"], args.cases)
print(res.to_string(index=False))
