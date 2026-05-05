import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict
import cv2
from importlib.resources import files

from cytoscan.config import Config, ResearchConfig, CellDetectionConfig, ChannelDetectionConfig
from cytoscan.preprocessing import load_frames, preprocess_frames, scaffold_experiment
from cytoscan.detections import FrameDetections
from cytoscan.cell_detector import detect_cells
from cytoscan.channel_detector import detect_walls, detect_interface
from cytoscan.flagging import compute_flags_all, print_flags
from cytoscan.analysis import analyze
from cytoscan.export import export_all


LOGO = r"""
                     d8                                            
 e88'888 Y8b Y888P  d88    e88 88e   dP"Y  e88'888  ,"Y88b 888 8e  
d888  '8  Y8b Y8P  d88888 d888 888b C88b  d888  '8 "8" 888 888 88b 
Y888   ,   Y8b Y    888   Y888 888P  Y88D Y888   , ,ee 888 888 888 
 "88,e8'    888     888    "88 88"  d,dP   "88,e8' "88 888 888 888 
            888                                                    
"""

GREEN  = "\033[32m"
RESET  = "\033[0m"

#get cytoscan version
try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("cytoscan")
except Exception:
    _VERSION = "unknown"

def _read_default_template() -> str:
    return files("cytoscan").joinpath("templates/default.yaml").read_text()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline microscopy perception tool for cell tracking in Sun Lab experiments"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    p_init = sub.add_parser("init", help="scaffold a new cytoscan experiment directory")
    p_init.add_argument("dir", help="path to experiment directory to create")

    p_run = sub.add_parser("run", help="run the full cytoscan pipeline on an experiment")
    p_run.add_argument("dir", help="experiment directory (must contain config.yaml)")

    p_val = sub.add_parser("validate", help="run detection + flagging only and print flag table")
    p_val.add_argument("dir", help="experiment directory (must contain config.yaml)")

    sub.add_parser("version", help="print cytoscan version and environment info")

    return parser.parse_args()

def _load_config(exp_dir: str) -> Config:
    cfg_path = Path(exp_dir) / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"[cytoscan] config.yaml not found in {exp_dir}. Run 'cytoscan init {exp_dir}' first")
    return Config.load(str(cfg_path))

def run_detections(r_cfg: ResearchConfig, celld_cfg: CellDetectionConfig, channeld_cfg: ChannelDetectionConfig, frames: Dict[int, tuple]) -> Dict[int, FrameDetections]:
    detections: Dict[int, FrameDetections] = {}

    for fi, (br, fl, mx) in frames.items():
        print(f"\r[cytoscan] running detections: frame {fi+1}/{len(frames)}", end="", flush=True)

        cell_dets = detect_cells(r_cfg, celld_cfg, fl)

        left_centers, left_coeffs, right_centers, right_coeffs, suggested_inset = detect_walls(r_cfg, channeld_cfg, br)
        interface_points, interface_curve = detect_interface(channeld_cfg, br, left_coeffs, right_coeffs, suggested_inset)

        image_h_px, image_w_px = cv2.imread(str(br)).shape[:2]

        detections[fi] = FrameDetections(
            br=br, fl=fl, mx=mx,
            cells=cell_dets,
            left_centers=left_centers, left_coeffs=left_coeffs,
            right_centers=right_centers, right_coeffs=right_coeffs,
            wall_inset=suggested_inset,
            interface_points=interface_points, interface_curve=interface_curve,
            image_w_px=image_w_px, image_h_px=image_h_px,
            flags=None,
        )
    print(" done.")
    return detections

def cmd_init(args):
    experiment_dir = Path(args.dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    cfg_dest = experiment_dir / "config.yaml"
    if cfg_dest.exists():
        print(f"[cytoscan] found config.yaml at {cfg_dest}. skipping")
    else:
        cfg_dest.write_text(_read_default_template())
        print(f"[cytoscan] created config.yaml at {cfg_dest}")

    scaffold_experiment(experiment_dir)

    print(f"[cytoscan] experiment directory ready: {experiment_dir}")
    print(f"[cytoscan] next: edit {cfg_dest}, then run 'cytoscan run {experiment_dir}'")

def cmd_run(args):
    cfg = _load_config(args.dir)
    experiment_dir = Path(args.dir)

    #print logo cuz its cool
    if cfg.export_visuals.enabled and cfg.export_visuals.print_logo :
        print(f"{LOGO}\t\t --- microfluidic cell perception ---\tv{_VERSION}\n")

    print(f"[cytoscan] experiment: {os.path.basename(experiment_dir)}")

    frames = load_frames(experiment_dir)
    preprocess_frames(cfg.research, cfg.preprocessing, experiment_dir, frames)
    detections = run_detections(cfg.research, cfg.cell_detection, cfg.channel_detection, frames)
    compute_flags_all(cfg.research, cfg.flagging, cfg.channel_detection, detections)
    findings = analyze(cfg.research, cfg.analysis, detections)
    export_all(cfg.export_visuals, cfg.export_data, experiment_dir, detections, findings)

    print(f"{GREEN}[cytoscan]{RESET} completed successfully")

def cmd_validate(args):
    cfg = _load_config(args.dir)
    experiment_dir = Path(args.dir)

    print(f"[cytoscan] validate: {os.path.basename(experiment_dir)}")

    frames = load_frames(experiment_dir)
    preprocess_frames(cfg.research, cfg.preprocessing, experiment_dir, frames)
    detections = run_detections(cfg.research, cfg.cell_detection, cfg.channel_detection, frames)
    compute_flags_all(cfg.research, cfg.flagging, cfg.channel_detection, detections)
    print_flags(detections)

def cmd_version(args):
    import platform
    print(f"cytoscan {_VERSION}")
    print(f"python   {platform.python_version()}")
    print(f"opencv   {cv2.__version__}")
    print(f"numpy    {np.__version__}")

def main():
    args = parse_args()
    dispatch = {
        "init":     cmd_init,
        "run":      cmd_run,
        "validate": cmd_validate,
        "version":  cmd_version,
    }
    dispatch[args.command](args)
