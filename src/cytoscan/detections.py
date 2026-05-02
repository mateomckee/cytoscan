from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from pathlib import Path

@dataclass
class FrameFlags:
    # wall metrics
    right_wall_anchor_strength: float
    left_wall_anchor_strength:  float

    # interface metrics
    interface_signal_ratio:     float
    interface_residual_mad_px:  float

    # diagnostic (no validity gate)
    mean_channel_width_um:      float

    # thresholds carried in so derived booleans don't need config access
    wall_anchor_strength_min:        float
    interface_signal_ratio_min:      float
    interface_residual_mad_max_px:   float

    @property
    def walls_valid(self) -> bool:
        return (self.right_wall_anchor_strength >= self.wall_anchor_strength_min
                and self.left_wall_anchor_strength  >= self.wall_anchor_strength_min)

    @property
    def interface_valid(self) -> bool:
        return (self.interface_signal_ratio    >= self.interface_signal_ratio_min
                and self.interface_residual_mad_px <= self.interface_residual_mad_max_px)

    @property
    def frame_valid(self) -> bool:
        return self.walls_valid and self.interface_valid

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
