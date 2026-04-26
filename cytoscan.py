import os
import sys
import argparse
import yaml
import tempfile
import cv2
import numpy as np
from typing import Dict

from dataclasses import dataclass

#temp
import matplotlib.pyplot as plt

from src.ilastik_runner import IlastikRunner
from src.cell_detector import detect_cells, visualize_detections
from src.channel_detector import detect_walls, detect_interface, visualize_curves

import matplotlib.pyplot as plt

@dataclass
class FrameDetections:
    left_wall:       np.ndarray   #coeffs
    right_wall:      np.ndarray   #coeffs
    interface:       np.ndarray   #coeffs
    cells:           list         #list of Detection
    is_valid:        bool         #interface stability flag

def visualize(experiment_dir_str: str, br_frame: str, frame_det: FrameDetections):
    os.makedirs(f"{experiment_dir_str}/output", exist_ok=True)

    img = plt.imread(br_frame)
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img)

    # walls and interface
    for coeffs, color in [
        (frame_det.left_wall,  'lime'),
        (frame_det.right_wall, 'red'),
        (frame_det.interface,  'cyan')
    ]:
        if coeffs is None:
            continue
        xs = [np.polyval(coeffs, y) for y in range(h)]
        ys = list(range(h))
        ax.plot(xs, ys, color=color, linewidth=1)

    # cells
    for d in frame_det.cells:
        ax.plot(d.centroid_x, d.centroid_y, 'b+', markersize=10, markeredgewidth=1.0)
        ax.text(d.centroid_x + 5, d.centroid_y, f"({d.centroid_x:.0f}, {d.centroid_y:.0f})",
                color='cyan', fontsize=5)

    basename = os.path.splitext(os.path.basename(br_frame))[0]
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{experiment_dir_str}/output/{basename}_output.png", dpi=150)

def count_cells_per_region(cells: list, interface_coeffs: np.ndarray) -> dict:
    counts = {"left": 0, "right": 0}
    for cell in cells:
        interface_x = np.polyval(interface_coeffs, cell.centroid_y)
        if cell.centroid_x < interface_x:
            counts["left"] += 1
        else:
            counts["right"] += 1
    return counts

#reads input frame pairs (brightfield, fluorescent) and outputs them as a dictionary
def load_frame_pairs(experiment_dir: str) -> Dict[int, tuple[str, str]]:
    bf_dir = os.path.join(experiment_dir, "brightfield")
    fl_dir = os.path.join(experiment_dir, "fluorescent")

    def load_dir(path: str) -> list[str]:
        encoded = os.fsencode(path)
        files = []
        for file in sorted(os.listdir(encoded)):
            filename = os.fsdecode(file)
            if filename.endswith(".tif") or filename.endswith(".tiff"):
                files.append(os.path.join(path, filename))
        return files

    bf_files = load_dir(bf_dir)
    fl_files = load_dir(fl_dir)

    if len(bf_files) != len(fl_files):
        raise RuntimeError(f"frame count mismatch: {len(bf_files)} bf vs {len(fl_files)} fl")

    return {i: (bf, fl) for i, (bf, fl) in enumerate(zip(bf_files, fl_files))}

#config and args
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
        sys.exit(f"[cytoscan] ERROR: 'ilastik_exe' not set in {args.c}")
    
    ilastik_model = cfg.get("ilastik_model", "")
    if not ilastik_model :
        sys.exit(f"[cytoscan] ERROR: 'ilastik_model' not set in {args.c}")
    
    experiment_dir_str = cfg.get("experiment", "")
    if not experiment_dir_str :
        sys.exit(f"[cytoscan] ERROR: 'experiment' not set in {args.c}")
    
    runner = IlastikRunner(ilastik_model, ilastik_exe=ilastik_exe)
    n_channels = 3 #BGR from OpenCV

    #

    #--- STEP 1 ---
    #gather all tif frames to analyze in experiment
    frame_pairs = load_frame_pairs(experiment_dir_str) 

    #dict that stores all frames with detections and the list of their detections
    detections: Dict[int, FrameDetections] = {}

    with tempfile.TemporaryDirectory() as tmp_dir :
        #--- STEP 2 ---
        #run ilastik on a subprocess, outputting HDF5 probability maps into a temp dir in disk
        
        print(f"[ilastik_runner] running ilastik on {len(frame_pairs)} frame(s)...")
        #run only on fluorescent frames
        fl_frames = [fl for _, fl in frame_pairs.values()]
        runner.run_on_frames(fl_frames, tmp_dir, n_channels)
        print("[ilastik_runner] ilastik done")

        #--- STEP 3 ---
        #load probability maps from disk
        for fi, (br, fl) in frame_pairs.items() :
            #get the name of the ilastik output file, which is the original filename + _Probabilities.h5, within the temp_dir
            base = os.path.splitext(os.path.basename(fl))[0] #both br and fl should have same base filename
            output_filename = base + "_Probabilities.h5"
            
            prob_path = os.path.join(tmp_dir, output_filename)
            cur_prob = runner.read_prob_map(prob_path)

            #--- STEP 4 ---
            #gather detections from the probability maps, output to detections dict if anything is detected
            dets = detect_cells(cur_prob)
        
            if(len(dets) >= 1) :
                detections[fi] = FrameDetections(cells = dets, left_wall = None, right_wall = None, interface = None, is_valid = True)

    print("[cytoscan] REPORT:")

    #--- STEP 5 ---
    #on all frames that had detections, detect the channel walls and interface (brightfield only)
    for fi in detections.keys() :
        frame_dets = detections[fi]
        br_frame, fl_frame = frame_pairs[fi]

        #calculate channel walls + interface mathematical curve
        left_coeffs, right_coeffs = detect_walls(br_frame);
        interface_coeffs = detect_interface(br_frame, left_coeffs, right_coeffs)

        #store in frame_dets
        frame_dets.left_wall = left_coeffs
        frame_dets.right_wall = right_coeffs
        frame_dets.interface = interface_coeffs

        #calculations
        counts = count_cells_per_region(frame_dets.cells, frame_dets.interface)
        print(f"{os.path.basename(br_frame)}: {len(frame_dets.cells)} cells\n\tleft: {counts['left']}\n\tright: {counts['right']}") 

        #visualization
        visualize(experiment_dir_str, br_frame, frame_dets)

if __name__ == "__main__" :
    main()
