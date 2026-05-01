import numpy as np
from pathlib import Path

from cytoscan.config import ExportDataConfig 
from cytoscan.detections import FrameDetections
from cytoscan.findings import *

def analyze(ed_cfg: ExportDataConfig, experiment_dir: Path, detections: FrameDetections) -> ExperimentFindings:
    for fi, fd in detections.items():
        print(f"\r[cytoscan] analyzing and quantifying detections: frame {fi+1}/{len(detections)}", end="", flush=True)
        #
    print(f" done.")
    return None


def count_cells_per_region(cells: list, interface_coeffs: np.ndarray) -> dict:
    counts = {"left": 0, "right": 0}
    for cell in cells:
        interface_x = np.polyval(interface_coeffs, cell.centroid_y)
        if cell.centroid_x < interface_x:
            counts["left"] += 1
        else:
            counts["right"] += 1
    return counts
