from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass
class FrameDetections:
    br:                 Path
    fl:                 Path
    mx:                 Path

    left_coeffs:        np.ndarray   #coeffs
    left_centers:       np.ndarray   #raw points of wall contour center, (y, x_mean) vertical function f(y) = x

    right_coeffs:       np.ndarray   #coeffs
    right_centers:      np.ndarray   #raw points

    wall_inset:         int

    interface_coeffs:   np.ndarray   #coeffs
    interface_centers:  np.ndarray   #raw points

    cells:              list         #list of Detection
    is_valid:           bool         #interface stability flag

@dataclass
class CellDetection :
    centroid_x: float
    centroid_y: float
    area: int           #pixels
    label: int
