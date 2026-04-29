import os
import sys
import shutil
import argparse
import yaml
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt

from cytoscan.config import * 
from cytoscan.detections import FrameDetections
from cytoscan.ilastik_runner import IlastikRunner
from cytoscan.cell_detector import detect_cells
from cytoscan.channel_detector import detect_walls, detect_interface
from cytoscan.analysis import count_cells_per_region
from cytoscan.visualization import export_visuals

#reads input frame (.tif/.tiff) triples (brightfield, fluorescent, mixed) and outputs them as a dictionary
def load_frames(experiment_dir: str) -> Dict[int, tuple[Path, Path, Path]]:
    br_dir = os.path.join(experiment_dir, "brightfield")
    fl_dir = os.path.join(experiment_dir, "fluorescent")
    mx_dir = os.path.join(experiment_dir, "mixed")

    def load_dir(path: str) -> list[Path]:
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

def parse_args() :
    parser = argparse.ArgumentParser(description="Offline microscopy perception tool for cell tracking in Sun Lab experiments")
    parser.add_argument("-c",required=False, default="configs/default.yaml", help="Config file path")
    return parser.parse_args()

def load_config(path: str) -> dict :
    with open(path) as f :
        return yaml.safe_load(f)

def main() :
    args = parse_args()
    cfg = Config.load(args.c)

    cfg_name = Path(args.c).name

    runner = IlastikRunner(cfg.ilastik_model, cfg.ilastik_exe)
    n_channels = 3 #BGR from OpenCV

    print(f"[cytoscan] config: {cfg_name}, experiment: {os.path.basename(cfg.experiment)}, ilastik_model: {os.path.basename(cfg.ilastik_model)}")

    #key data structures for this program
    frames: Dict[int, (Path, Path, Path)] = load_frames(cfg.experiment) 
    detections: Dict[int, FrameDetections] = {}

    # Stage 1: the expensive loop, performs all detections (cell + channel)
    with tempfile.TemporaryDirectory() as tmp_dir :
        print(f"[ilastik_runner] running ilastik on {len(frames)} frame(s)...")
        fl_frames = [fl for _, fl, _ in frames.values()]
        runner.run_on_frames(fl_frames, tmp_dir, n_channels)

        print(f"[cytoscan] detecting cells and channels on {len(frames)} frame(s)...")
        for fi, (br, fl, mx) in frames.items() :
            #get the name of the ilastik output file, which is the original filename + _Probabilities.h5, within the temp_dir
            base = os.path.splitext(os.path.basename(fl))[0]
            output_filename = base + "_Probabilities.h5"

            prob_path = os.path.join(tmp_dir, output_filename)
            cur_prob = runner.read_prob_map(prob_path)

            #gather detections from the probability maps, output to detections dict if anything is detected
            dets = detect_cells(cur_prob)

            left_centers, left_coeffs, right_centers, right_coeffs, suggested_inset = detect_walls(br)
            interface_points, interface_curve = detect_interface(br, left_coeffs, right_coeffs, suggested_inset)

            #store detections for this frame
            detections[fi] = FrameDetections(br = br, fl = fl, mx = mx, cells = dets, left_centers = left_centers, left_coeffs = left_coeffs, right_centers = right_centers, right_coeffs = right_coeffs, wall_inset = suggested_inset, interface_points = interface_points, interface_curve = interface_curve, is_valid = True)

    # Stage 2: each is a one-shot pass over results
    export_visuals(cfg.output.export_visuals, cfg.experiment, detections)
    
    print("[cytoscan] finished execution")
