import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict
import cv2
from importlib.resources import files

from cytoscan import _logging
from cytoscan.config import Config, ResearchConfig, CellDetectionConfig, ChannelDetectionConfig
from cytoscan.preprocessing import load_frames, preprocess_frames, scaffold_experiment
from cytoscan.detections import FrameDetections
from cytoscan.cell_detector import detect_cells
from cytoscan.channel_detector import detect_walls, detect_interface
from cytoscan.flagging import compute_flags_all, print_flags
from cytoscan.analysis import analyze
from cytoscan.export import export_all

log = logging.getLogger(__name__)


LOGO = r"""
          |                             
,---.,   .|--- ,---.,---.,---.,---.,---.
|    |   ||    |   |`---.|    ,---||   |
`---'`---|`---'`---'`---'`---'`---^`   '
     `---'
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
    # Shared parent parser holds the global flags. Adding it as a parent on each
    # subparser AND on the top-level parser lets users put them in either spot:
    #   cytoscan -v run exp/      ← before the subcommand
    #   cytoscan run exp/ -v      ← after the subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true",
                        help="enable DEBUG-level logging (algorithm internals, per-frame diagnostics)")
    common.add_argument("-q", "--quiet", action="store_true",
                        help="suppress INFO logs; only WARNING and ERROR are shown")
    common.add_argument("--log-file", metavar="PATH",
                        help="also write a full DEBUG-level log to this file (uncolored, with timestamps)")

    parser = argparse.ArgumentParser(
        description="Offline microscopy perception tool for cell tracking in Sun Lab experiments",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    p_run = sub.add_parser("run", parents=[common],
                           help="scaffold an experiment and run the full cytoscan pipeline")
    p_run.add_argument("dir", help="experiment directory (must contain config.yaml)")

    sub.add_parser("version", parents=[common],
                   help="print cytoscan version and environment info")

    return parser.parse_args()

def _load_config(exp_dir: str) -> Config:
    cfg_path = Path(exp_dir) / "config.yaml"
    if not cfg_path.exists():
        log.error("config.yaml not found in %s. Run 'cytoscan init %s' first", exp_dir, exp_dir)
        sys.exit(1)
    return Config.load(str(cfg_path))

def run_detections(r_cfg: ResearchConfig, celld_cfg: CellDetectionConfig, channeld_cfg: ChannelDetectionConfig, frames: Dict[int, tuple]) -> Dict[int, FrameDetections]:
    detections: Dict[int, FrameDetections] = {}

    log.info("running detections on %d frames", len(frames))
    for fi, (br, fl, mx) in _logging.progress(frames.items(), "detecting", total=len(frames)):
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
        log.debug("frame %d: %d cells, wall_inset=%d", fi, len(cell_dets), suggested_inset)
    return detections

def cmd_init(args):
    experiment_dir = Path(args.dir)

    # create experiment dir if doesnt exist
    experiment_dir.mkdir(parents=True, exist_ok=True)

    cfg_dest = experiment_dir / "config.yaml"
    
    # create config if doesnt exist
    if cfg_dest.exists():
        log.info("found config.yaml at %s, skipping", cfg_dest)
    else:
        cfg_dest.write_text(_read_default_template())
        log.info("created config.yaml at %s", cfg_dest)

    # setup experiment structure and organize any loose frames into correct category
    scaffold_experiment(experiment_dir)

    log.info("experiment directory ready: %s", experiment_dir)

def cmd_run(args):
    cmd_init(args)

    cfg = _load_config(args.dir)
    experiment_dir = Path(args.dir)

    #print logo cuz its cool. printed directly (not logged) — it's UI banner, not a log record.
    if cfg.export_visuals.enabled and cfg.export_visuals.print_logo:
        sys.stderr.write(f"{LOGO}   --- microfluidic cell perception ---   v{_VERSION}\n\n")

    log.info("experiment: %s", os.path.basename(experiment_dir))

    frames = load_frames(experiment_dir)
    preprocess_frames(cfg.research, cfg.preprocessing, experiment_dir, frames)
    detections = run_detections(cfg.research, cfg.cell_detection, cfg.channel_detection, frames)
    compute_flags_all(cfg.research, cfg.flagging, cfg.channel_detection, detections)
    findings = analyze(cfg.research, cfg.analysis, detections)
    export_all(cfg.export_visuals, cfg.export_data, experiment_dir, detections, findings)

    log.info("\033[32mcompleted successfully\033[0m" if sys.stderr.isatty() else "completed successfully")

def cmd_version(args):
    # Plain stdout — `cytoscan version` is a data command (greppable, parseable).
    import platform
    print(f"cytoscan {_VERSION}")
    print(f"python   {platform.python_version()}")
    print(f"opencv   {cv2.__version__}")
    print(f"numpy    {np.__version__}")

def main():
    args = parse_args()
    _logging.setup(
        verbose=args.verbose,
        quiet=args.quiet,
        log_file=Path(args.log_file) if args.log_file else None,
    )
    dispatch = {
        "run":      cmd_run,
        "version":  cmd_version,
    }
    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        log.warning("interrupted by user")
        sys.exit(130)
    except Exception:
        # Full traceback in --log-file (FileHandler captures DEBUG); user sees a clean line on stderr.
        log.exception("unhandled error")
        sys.exit(1)
