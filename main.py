import sys
import argparse
import yaml

def parse_args() :
    parser = argparse.ArgumentParser(description="Offline microscopy perception tool for cell tracking in Sun Lab experiments")
    parser.add_argument("-c",required=True, help="Config file path")
    return parser.parse_args()

def load_config(path: str) -> dict :
    with open(path) as f :
        return yaml.safe_load(f)

def main() :
    args = parse_args()
    cfg = load_config(args.c)

    ilastik_exe = cfg.get("ilastik_exe", "")
    if not ilastik_exe :
        sys.exit(f"[main] ERROR: 'ilastik_exe' not set in {args.c}")

        


if __name__ == "__main__" :
    main()
