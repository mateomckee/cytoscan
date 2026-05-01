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
from cytoscan.flagging import compute_flags_all, print_flags
from cytoscan.analysis import analyze
from cytoscan.export import export_visuals, export_data, export_all

def parse_args() :
    parser = argparse.ArgumentParser(description="Offline microscopy perception tool for cell tracking in Sun Lab experiments")
    parser.add_argument("-c",required=False, default="configs/default.yaml", help="Config file path")
    return parser.parse_args()

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

def run_detections(cfg: Config, frames: Dict[int, tuple[Path, Path, Path]]) -> Dict[int, FrameDetections]:
    detections: Dict[int, FrameDetections] = {}

    runner = IlastikRunner(cfg.ilastik.model, cfg.ilastik.exe)
    n_channels = 3 #BGR from OpenCV

    with tempfile.TemporaryDirectory() as tmp_dir :
        fl_frames = [fl for _, fl, _ in frames.values()]
        runner.run_on_frames(fl_frames, tmp_dir, n_channels)

        for fi, (br, fl, mx) in frames.items() :
            print(f"\r[cytoscan] running channel detections: frame {fi+1}/{len(frames)}", end="", flush=True)

            #get the name of the ilastik output file, which is the original filename + _Probabilities.h5, within the temp_dir
            base = os.path.splitext(os.path.basename(fl))[0]
            output_filename = base + "_Probabilities.h5"

            prob_path = os.path.join(tmp_dir, output_filename)
            cur_prob = runner.read_prob_map(prob_path)

            #gather detections from the probability maps, output to detections dict if anything is detected
            dets = detect_cells(cur_prob)

            left_centers, left_coeffs, right_centers, right_coeffs, suggested_inset = detect_walls(cfg.detection, br)
            interface_points, interface_curve = detect_interface(cfg.detection, br, left_coeffs, right_coeffs, suggested_inset)

            #store detections for this frame
            detections[fi] = FrameDetections(br = br, fl = fl, mx = mx, cells = dets, left_centers = left_centers, left_coeffs = left_coeffs, right_centers = right_centers, right_coeffs = right_coeffs, wall_inset = suggested_inset, interface_points = interface_points, interface_curve = interface_curve, flags = None)
            
            fd = detections[fi]

        print(" done.")
    return detections

def main() :
    args = parse_args()
    cfg = Config.load(args.c)

    print(f"[cytoscan] config: {Path(args.c).name}, experiment: {os.path.basename(cfg.experiment.dir)}, ilastik_model: {os.path.basename(cfg.ilastik.model)}")

    frames = load_frames(cfg.experiment.dir) 

    # stage 1: the expensive loop, performs all detections (cell + channel)
    detections = run_detections(cfg, frames)

    # stage 2: each is a one-shot pass over results
    compute_flags_all(cfg.flagging, cfg.experiment, detections)

    #DEBUG print flags
    print_flags(detections)

    findings = analyze(cfg.output.export_data, cfg.experiment.dir, detections)
    export_all(cfg.output, cfg.experiment.dir, detections, findings)
    
    print("[cytoscan] completed successfully")

