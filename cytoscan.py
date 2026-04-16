import os
import sys
import argparse
import yaml
import tempfile

#temp
import matplotlib.pyplot as plt

from src.ilastik_runner import IlastikRunner
from src.cell_detector import detect_cells

import matplotlib.pyplot as plt

def visualize_detections(image_path: str, detections: list):
    img = plt.imread(image_path)
    
    plt.imshow(img)
    for d in detections:
        plt.plot(d.centroid_x, d.centroid_y, 'b+', markersize=10, markeredgewidth=1.0)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("detections_out.png", dpi=150)
    plt.show()

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
    
    #--- init ---
    runner = IlastikRunner(ilastik_model, ilastik_exe=ilastik_exe)
    frames: Dict[int, str] = {} 
    n_channels = 3   # BGR from OpenCV

    #--- STEP 1 ---
    #gather all tif frames to analyze in experiment

    experiment_dir = os.fsencode(experiment_dir_str)
    frame_index = 0
    for file in os.listdir(experiment_dir) :
        filename = os.fsdecode(file)
        if filename.endswith(".tif") :
            frame_path = os.path.join(experiment_dir_str, filename)
            frames[frame_index] = frame_path
            frame_index += 1

    print(f"Running ilastik on {len(frames)} frame(s)...")

    with tempfile.TemporaryDirectory() as tmp_dir :
        #--- STEP 2 ---
        #run ilastik on a subprocess, outputting HDF5 probability maps into a temp dir in disk
        
        runner.run_on_frames(list(frames.values()), tmp_dir, n_channels)
        print("ilastik done")

        #--- STEP 3 ---
        #load probability maps from disk
        
        for fi, fp in frames.items() :
            #convert original .tif filename to an ilastik output .h5 filename
            base = os.path.splitext(os.path.basename(fp))[0]
            filename = base + "_Probabilities.h5"

            prob_path = os.path.join(tmp_dir, filename)
            cur_prob = runner.read_prob_map(prob_path)

            #--- STEP 4 ---
            detections = detect_cells(cur_prob)
            
            visualize_detections(fp, detections)




if __name__ == "__main__" :
    main()
