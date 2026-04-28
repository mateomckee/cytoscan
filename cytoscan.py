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
from src.channel_detector import detect_walls, detect_interface

import matplotlib.pyplot as plt

@dataclass
class FrameDetections:
    left_coeffs:        np.ndarray   #coeffs
    left_centers:       np.ndarray   #raw points of wall contour center, (y, x_mean) vertical function f(y) = x

    right_coeffs:       np.ndarray   #coeffs
    right_centers:      np.ndarray   #raw points

    wall_inset:         int

    interface_coeffs:   np.ndarray   #coeffs
    interface_centers:  np.ndarray   #raw points

    cells:              list         #list of Detection
    is_valid:           bool         #interface stability flag

def visualize(experiment_dir_str: str, br_frame: str, frame_det: FrameDetections):
    os.makedirs(f"{experiment_dir_str}/output", exist_ok=True)

    img = plt.imread(br_frame)
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img)

    # inset wall coeffs (shifted inward by frame_det.wall_inset)
    left_in  = frame_det.left_coeffs.copy()
    right_in = frame_det.right_coeffs.copy()
    left_in[-1]  += frame_det.wall_inset
    right_in[-1] -= frame_det.wall_inset

    # walls, inset walls, and interface
    for coeffs, color in [
        (frame_det.left_coeffs,      'lime'),
        (frame_det.right_coeffs,     'red'),
        (left_in,                    'yellow'),
        (right_in,                   'yellow'),
        (frame_det.interface_coeffs, 'cyan'),
    ]:
        if coeffs is None:
            continue
        ys = np.arange(h)
        xs = np.polyval(coeffs, ys)
        ax.plot(xs, ys, color=color, linewidth=1)

    # cells
    for d in frame_det.cells:
        ax.plot(d.centroid_x, d.centroid_y, 'b+', markersize=10, markeredgewidth=1.0)
        ax.text(d.centroid_x + 5, d.centroid_y, f"({d.centroid_x:.0f}, {d.centroid_y:.0f})",
                color='cyan', fontsize=5)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
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

#reads input frame pairs (brightfield, fluorescent, mixed) and outputs them as a dictionary
def load_frames(experiment_dir: str) -> Dict[int, tuple[str, str]]:
    br_dir = os.path.join(experiment_dir, "brightfield")
    fl_dir = os.path.join(experiment_dir, "fluorescent")
    mx_dir = os.path.join(experiment_dir, "mixed")

    def load_dir(path: str) -> list[str]:
        encoded = os.fsencode(path)
        files = []
        for file in sorted(os.listdir(encoded)):
            filename = os.fsdecode(file)
            if filename.endswith(".tif") or filename.endswith(".tiff"):
                files.append(os.path.join(path, filename))
        return files

    br_files = load_dir(br_dir)
    fl_files = load_dir(fl_dir)
    mx_files = load_dir(mx_dir)

    if not (len(br_files) == len(fl_files) == len(mx_files)):
        raise RuntimeError(f"frame count mismatch: {len(br_files)} br, {len(fl_files)} fl, {len(mx_files)} mx")

    return {i: (br, fl, mx) for i, (br, fl, mx) in enumerate(zip(br_files, fl_files, mx_files))}

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

    print(f"[cytoscan] experiment: {os.path.basename(experiment_dir_str)}, ilastik_model: {os.path.basename(ilastik_model)}")

    #--- STEP 1 ---
    #gather all tif frames to analyze in experiment
    frames = load_frames(experiment_dir_str) 

    #dict that stores all frames with detections and the list of their detections
    detections: Dict[int, FrameDetections] = {}

    with tempfile.TemporaryDirectory() as tmp_dir :
        #--- STEP 2 ---
        #run ilastik on a subprocess, outputting HDF5 probability maps into a temp dir in disk
        
        print(f"[ilastik_runner] running ilastik on {len(frames)} frame(s)...")
        #run only on fluorescent frames
        fl_frames = [fl for _, fl, _ in frames.values()]
        runner.run_on_frames(fl_frames, tmp_dir, n_channels)

        print(f"[channel_detector] detecting channel walls and interface on {len(frames)} frame(s)...")
        #--- STEP 3 ---
        #load probability maps from disk
        for fi, (br, fl, mx) in frames.items() :
            #get the name of the ilastik output file, which is the original filename + _Probabilities.h5, within the temp_dir
            base = os.path.splitext(os.path.basename(fl))[0] #both br and fl should have same base filename
            output_filename = base + "_Probabilities.h5"
            
            prob_path = os.path.join(tmp_dir, output_filename)
            cur_prob = runner.read_prob_map(prob_path)

            #--- STEP 4 ---
            #gather detections from the probability maps, output to detections dict if anything is detected
            dets = detect_cells(cur_prob)

            #skip frames where no detections ocurred
            if(len(dets) < 1) : continue

            left_centers, left_coeffs, right_centers, right_coeffs, suggested_inset = detect_walls(br)
            interface_centers, interface_coeffs = detect_interface(br, left_coeffs, right_coeffs, suggested_inset)

            detections[fi] = FrameDetections(cells = dets, left_centers = left_centers, left_coeffs = left_coeffs, right_centers = right_centers, right_coeffs = right_coeffs, wall_inset = suggested_inset, interface_centers = interface_centers, interface_coeffs = interface_coeffs, is_valid = True)

    print("[cytoscan] REPORT:")

    #--- STEP 5 ---
    for fi, (br, fl, mx) in frames.items() :
        #skip zero-det frames
        if(fi not in detections) :
            print(f"{os.path.basename(mx)}: no detections")
            continue
        
        frame_dets = detections[fi]

        #calculations
        counts = count_cells_per_region(frame_dets.cells, frame_dets.interface_coeffs)
        print(f"{os.path.basename(mx)}: {len(frame_dets.cells)} cells\n\tleft: {counts['left']}\n\tright: {counts['right']}") 

        #visualization
        visualize(experiment_dir_str, br, frame_dets)

if __name__ == "__main__" :
    main()
