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
from cytoscan.config import Config
from cytoscan.preprocessing import scaffold_experiment
from cytoscan.pipeline import run_pipeline

log = logging.getLogger(__name__)


LOGO = r"""
          |                             
,---.,   .|--- ,---.,---.,---.,---.,---.
|    |   ||    |   |`---.|    ,---||   |
`---'`---|`---'`---'`---'`---'`---^`   '
     `---'
"""

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("cytoscan")
except Exception:
    _VERSION = "unknown"

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
        log.error("config.yaml not found in %s", exp_dir)
        sys.exit(1)
    return Config.load(str(cfg_path))

def cmd_init(args):
    scaffold_experiment(Path(args.dir))

def cmd_run(args):
    cmd_init(args)
    cfg = _load_config(args.dir)

    # logo printed directly (not logged), its a UI banner, not a log record
    if cfg.export_visuals.enabled and cfg.export_visuals.print_logo:
        sys.stderr.write(f"{LOGO}   --- microfluidic cell perception ---   v{_VERSION}\n\n")

    run_pipeline(cfg, Path(args.dir))

def cmd_version(args):
    # plain stdout, `cytoscan version` is a data command (greppable, parseable)
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
