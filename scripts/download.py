"""
Fase 1: descarga de iNaturalist a Drive.

Uso:
    python scripts/download.py --config configs/pilot79.yaml
    python scripts/download.py --config configs/full.yaml --steps photos,resize,check

Pasos (en orden): metadata, tables, photos, resize, check, excel.
Todo es reanudable: si se corta, se vuelve a correr el mismo comando.
"""
import argparse

import sepia

STEPS = ["metadata", "tables", "photos", "resize", "check", "excel"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="archivo YAML en configs/")
    ap.add_argument("--steps", default=",".join(STEPS), help=f"pasos separados por coma: {STEPS}")
    ap.add_argument("--count-only", action="store_true", help="solo muestra cuantos registros devuelve la consulta")
    args = ap.parse_args()

    cfg = sepia.load_config(args.config)
    if args.count_only:
        print(sepia.count_results(cfg), "registros en la consulta")
        return

    steps = [s.strip() for s in args.steps.split(",")]
    unknown = set(steps) - set(STEPS)
    if unknown:
        ap.error(f"pasos desconocidos: {unknown}")

    if "metadata" in steps:
        if cfg.ids_excel:
            sepia.download_by_ids(cfg, sepia.ids_from_excel(cfg.ids_excel), tag="ids")
        else:
            if cfg.bbox is None:
                ap.error("falta bbox en la configuracion (descarga completa sin rectangulo)")
            sepia.download_observations(cfg)
    if "tables" in steps:
        sepia.build_tables(cfg)
    if "photos" in steps:
        sepia.download_photos(cfg)
    if "resize" in steps:
        for side in cfg.resize:
            sepia.make_resized(cfg, max_side=side)
    if "check" in steps:
        sepia.check_download(cfg)
    if "excel" in steps:
        sepia.build_excel(cfg)


if __name__ == "__main__":
    main()
