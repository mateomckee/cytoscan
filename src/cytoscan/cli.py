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
import cv2
import matplotlib.pyplot as plt

from cytoscan.config import * 
from cytoscan.preprocessing import load_frames, preprocess_frames
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

            #detect walls and interface, store coeffs/curve
            left_centers, left_coeffs, right_centers, right_coeffs, suggested_inset = detect_walls(cfg.detection, cfg.experiment, br)
            interface_points, interface_curve = detect_interface(cfg.detection, br, left_coeffs, right_coeffs, suggested_inset)

            #get brightfield frame dimensions (after preprocessing) for later pixel to um conversion
            image_h_px, image_w_px = cv2.imread(str(br)).shape[:2]

            #store all detection data for this frame
            detections[fi] = FrameDetections(br = br, fl = fl, mx = mx, cells = dets, left_centers = left_centers, left_coeffs = left_coeffs, right_centers = right_centers, right_coeffs = right_coeffs, wall_inset = suggested_inset, interface_points = interface_points, interface_curve = interface_curve, image_w_px = image_w_px, image_h_px = image_h_px, flags = None)
        print(" done.")
    return detections

def main() :
    args = parse_args()
    cfg = Config.load(args.c)

    print(f"[cytoscan] config: {Path(args.c).name}, experiment: {os.path.basename(cfg.experiment.dir)}, ilastik_model: {os.path.basename(cfg.ilastik.model)}")

    frames = load_frames(cfg.experiment.dir) 

    preprocess_frames(cfg.preprocessing, cfg.experiment, frames) #updates frames map to point fi -> preprocessed frames path in a separate dir. becomes new canonical frame for rest of pipeline

    detections = run_detections(cfg, frames)

    compute_flags_all(cfg.flagging, cfg.detection, cfg.experiment, detections)

    #DEBUG print flags
    #print_flags(detections)

    findings = analyze(cfg.analysis, cfg.experiment, detections)
    export_all(cfg.output, cfg.experiment.dir, detections, findings)
    
    print("[cytoscan] completed successfully")

