import sys
import os
import logging
import cv2
import numpy as np

from cytoscan import _logging
from cytoscan.config import Config, ResearchConfig, CellDetectionConfig, ChannelDetectionConfig
from cytoscan.preprocessing import load_frames, preprocess_frames
from cytoscan.detections import FrameDetections
from cytoscan.cell_detector import detect_cells
from cytoscan.channel_detector import detect_walls, detect_interface
from cytoscan.flagging import compute_flags_all
from cytoscan.analysis import analyze
from cytoscan.export import export_all

from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

def run_pipeline(cfg: Config, experiment_dir: Path) :
    log.info("experiment: %s", os.path.basename(experiment_dir))

    frames = load_frames(experiment_dir)
    preprocess_invalid = preprocess_frames(cfg.research, cfg.preprocessing, experiment_dir, frames)
    detections = run_detections(cfg.research, cfg.cell_detection, cfg.channel_detection, frames)
    compute_flags_all(cfg.research, cfg.flagging, cfg.channel_detection, detections)
    findings = analyze(cfg.research, cfg.analysis, detections, preprocess_invalid)
    export_all(cfg.export_visuals, cfg.export_data, experiment_dir, detections, findings)

    log.info("\033[32mcompleted successfully\033[0m" if sys.stderr.isatty() else "completed successfully")

def run_detections(r_cfg: ResearchConfig, celld_cfg: CellDetectionConfig, channeld_cfg: ChannelDetectionConfig, frames: Dict[int, tuple]) -> Dict[int, FrameDetections]:
    detections: Dict[int, FrameDetections] = {}

    log.info("running detections on %d frames", len(frames))
    for fi, (br, fl, mx) in _logging.progress(frames.items(), "detecting", total=len(frames)):
        cell_dets = detect_cells(r_cfg, celld_cfg, fl)

        left_centers, left_coeffs, right_centers, right_coeffs, suggested_inset = detect_walls(r_cfg, channeld_cfg, br)
        interface_points, interface_curve = detect_interface(channeld_cfg, br, left_coeffs, right_coeffs, suggested_inset)

        # drop cells whose centroid falls outside the detected channel walls
        n_before = len(cell_dets)
        cell_dets = [
            c for c in cell_dets
            if np.polyval(left_coeffs, c.centroid_y) <= c.centroid_x <= np.polyval(right_coeffs, c.centroid_y)
        ]
        n_dropped = n_before - len(cell_dets)

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
        log.debug("frame %d: %d cells (dropped %d outside walls), wall_inset=%d", fi, len(cell_dets), n_dropped, suggested_inset)
    return detections

