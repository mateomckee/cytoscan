from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from pathlib import Path

@dataclass
class FrameFlags:
    # walls
    left_spans_full:        bool
    right_spans_full:       bool

    # channel width
    mean_channel_width_um:  float
    channel_width_valid:    bool

    # interface
    interface_curve_amplitude: float
    interface_valid:        bool

    @property
    def walls_valid(self) -> bool:
        return self.left_spans_full and self.right_spans_full

    @property
    def frame_valid(self) -> bool:
        return self.walls_valid and self.channel_width_valid and self.interface_valid

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

    interface_curve:    Optional[Callable[[np.ndarray], np.ndarray]]
    interface_points:   np.ndarray   

    cells:              list         #list of Detection
    
    flags:              Optional[FrameFlags]

@dataclass
class CellDetection :
    centroid_x: float
    centroid_y: float
    area: int           #pixels
    label: int
