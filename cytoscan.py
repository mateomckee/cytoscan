import os
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

    #--- config setup ----
    ilastik_exe = cfg.get("ilastik_exe", "")
    if not ilastik_exe :
        sys.exit(f"[main] ERROR: 'ilastik_exe' not set in {args.c}")
    
    ilastik_model = cfg.get("ilastik_model", "")
    if not ilastik_model :
        sys.exit(f"[main] ERROR: 'ilastik_model' not set in {args.c}")
    
    experiment_dir_str = cfg.get("experiment", "")
    if not experiment_dir_str :
        sys.exit(f"[main] ERROR: 'experiment' not set in {args.c}")

    #--- STEP 1 ---
    #gather all tif frames to analyze in experiment

    frames: Dict[int, str] = {} #frame_index -> .tif file name

    experiment_dir = os.fsencode(experiment_dir_str)
    frame_index = 0
    for file in os.listdir(experiment_dir) :
        filename = os.fsdecode(file)
        if filename.endswith(".tif") :
            frame_path = os.path.join(experiment_dir_str, filename)
            frames[frame_index] = frame_path
        frame_index += 1

    print(frames)

    #--- STEP 2 ---
    #run ilastik on .tif files using the given .ilp model

    #ilasitk_runner

if __name__ == "__main__" :
    main()
